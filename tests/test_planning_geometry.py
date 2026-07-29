"""Geometry and Jacobian tests for the planning layer.

The optimiser in ``planning/path_optimizer.py`` is a Gauss-Newton least
squares solve, so it is only as trustworthy as two things: the signed
distances and gradients returned by ``planning/obstacles.py``, and the
claim that the project's minimum-jerk spline is LINEAR in its knot
positions.  Both are asserted here directly rather than inferred from a
converged plan, because a wrong gradient does not make the optimiser
fail loudly -- it makes it converge to the wrong path.

Three properties carry the most weight:

- every obstacle's analytic gradient matches a central finite difference
  of its own distance function (the optimiser's obstacle Jacobian block
  is built from nothing else),
- the spline basis reproduces the real ``CartesianWaypointTrajectory``
  to machine precision, which is what makes an analytic Jacobian legal
  at all,
- and the optimiser's assembled Jacobian matches a finite difference of
  its own residual vector.

Points used for finite differencing are chosen away from the kinks that
these distance functions legitimately have (the cylinder axis, the
radial/vertical changeover, box edges and faces), because a finite
difference is meaningless where the derivative does not exist.
"""

import unittest

import numpy as np

from controller.human_safety import finite_cylinder_distance_gradient
from controller.state import Pose, TargetFrame
from controller.trajectory import (
    CartesianWaypoint,
    CartesianWaypointTrajectory,
    TrajectoryLimits,
)
from controller.transforms import rotation_from_rpy
from planning import path_optimizer
from planning.obstacles import (
    Box,
    HumanCylinder,
    ObstacleSet,
    Plane,
    Sphere,
    world_plane_in_torso,
)
from planning.path_optimizer import (
    OptimizerParams,
    _spline_basis,
    optimize_path,
)
from runtime_config import HumanSafetyConfig


_FD_STEP_M = 1e-6
_FD_TOLERANCE = 1e-6
_JACOBIAN_TOLERANCE = 1e-5


def _human_config(**changes):
    values = {
        "enabled": True,
        "center_xy_torso_m": (0.0, 0.0),
        "radius_m": 0.25,
        "z_min_torso_m": -1.10,
        "z_max_torso_m": 0.70,
        "clearance_m": 0.02,
        "control_margin_m": 0.02,
        "activation_distance_m": 0.10,
        "recovery_gain_s_inv": 4.0,
        "approach_velocity_damping": 0.0,
        "projection_iterations": 50,
        "constraint_tolerance_m_s": 1e-6,
    }
    values.update(changes)
    return HumanSafetyConfig(**values)


def _distance(obstacle, point):
    distance, _ = obstacle.distance_and_gradient(np.asarray([point]))
    return float(distance[0])


def _finite_difference_gradient(obstacle, point, step_m=_FD_STEP_M):
    point = np.asarray(point, dtype=float)
    gradient = np.zeros(3)
    for axis in range(3):
        offset = np.zeros(3)
        offset[axis] = step_m
        gradient[axis] = (
            _distance(obstacle, point + offset)
            - _distance(obstacle, point - offset)
        ) / (2.0 * step_m)
    return gradient


def _finite_difference_jacobian(residuals, point, step=_FD_STEP_M):
    point = np.asarray(point, dtype=float)
    columns = []
    for index in range(point.size):
        offset = np.zeros(point.size)
        offset[index] = step
        columns.append(
            (residuals(point + offset) - residuals(point - offset))
            / (2.0 * step)
        )
    return np.stack(columns, axis=1)


def _limits():
    return TrajectoryLimits(0.2, 0.5, 0.5, 1.0)


def _crossing_scene():
    """A straight path that passes through the wearer envelope."""
    torso_pose_world = Pose(np.zeros(3), np.eye(3))
    obstacles = ObstacleSet((HumanCylinder(_human_config()),))
    start = Pose(np.array([0.0, -0.55, 0.20]), np.eye(3))
    goal = Pose(np.array([0.0, 0.55, 0.20]), np.eye(3))
    return start, goal, torso_pose_world, obstacles


def _solve_capturing_closures(*args, **kwargs):
    """Run ``optimize_path`` while intercepting its scipy call.

    ``residuals`` and ``jacobian`` are closures built inside
    ``optimize_path`` and never exposed, so substituting the module-level
    ``least_squares`` name it looks up is the only way to test the REAL
    closures instead of a re-implementation of them.  Recording the LAST
    call also matters: the optimiser may re-solve on the delivered
    timing, and the closures then read the re-timed spline basis, so the
    residual and the Jacobian stay consistent with each other only if
    both come from the same (final) call.
    """
    captured = {}
    real_least_squares = path_optimizer.least_squares

    def recording(fun, x0, jac=None, **options):
        solution = real_least_squares(fun, x0, jac=jac, **options)
        captured["residuals"] = fun
        captured["jacobian"] = jac
        captured["initial"] = np.asarray(x0, dtype=float).copy()
        captured["solution"] = np.asarray(solution.x, dtype=float).copy()
        return solution

    path_optimizer.least_squares = recording
    try:
        result = optimize_path(*args, **kwargs)
    finally:
        path_optimizer.least_squares = real_least_squares
    return result, captured


class HumanCylinderAgreementTest(unittest.TestCase):
    """The planner and the real-time filter must see the same wearer."""

    def test_distance_matches_safety_filter_minus_clearance(self):
        config = _human_config()
        obstacle = HumanCylinder(config)
        points = np.array([
            [0.10, 0.00, 0.00],
            [0.05, -0.08, -0.60],
            [0.60, 0.10, 0.00],
            [-0.40, 0.30, 0.20],
            [0.05, 0.00, 1.20],
            [-0.02, 0.04, 0.90],
            [0.05, 0.02, -1.50],
            [0.00, -0.10, -2.00],
            [0.50, 0.00, 1.00],
            [-0.30, -0.30, -1.40],
            [0.25, 0.00, 0.70],
        ])
        distance, gradient = obstacle.distance_and_gradient(points)

        for index, point in enumerate(points):
            reference_distance, reference_gradient = (
                finite_cylinder_distance_gradient(point, config)
            )
            self.assertAlmostEqual(
                distance[index],
                reference_distance - config.clearance_m,
                places=12,
                msg=f"distance disagrees at point {index}: {point}",
            )
            np.testing.assert_allclose(
                gradient[index],
                reference_gradient,
                rtol=0.0,
                atol=1e-12,
                err_msg=f"gradient disagrees at point {index}: {point}",
            )

    def test_clearance_offset_is_the_filter_activation_boundary(self):
        config = _human_config(clearance_m=0.07)
        obstacle = HumanCylinder(config)
        surface_point = np.array([config.radius_m, 0.0, 0.0])
        self.assertAlmostEqual(
            _distance(obstacle, surface_point),
            -config.clearance_m,
            places=12,
        )


class ObstacleGradientFiniteDifferenceTest(unittest.TestCase):
    """The single most load-bearing property in the obstacle module."""

    def _check(self, obstacle, points, label):
        points = np.asarray(points, dtype=float)
        _, gradient = obstacle.distance_and_gradient(points)
        for index, point in enumerate(points):
            numeric = _finite_difference_gradient(obstacle, point)
            np.testing.assert_allclose(
                gradient[index],
                numeric,
                rtol=0.0,
                atol=_FD_TOLERANCE,
                err_msg=(
                    f"{label} gradient != finite difference at "
                    f"point {index}: {point}"
                ),
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(gradient[index])),
                1.0,
                places=9,
                msg=f"{label} gradient is not a unit vector at {point}",
            )

    def test_sphere_gradient(self):
        obstacle = Sphere((0.30, 0.10, 0.90), 0.15)
        self._check(
            obstacle,
            [
                [0.80, 0.10, 0.90],
                [0.30, 0.55, 0.90],
                [0.10, -0.20, 1.30],
                [0.36, 0.13, 0.94],
                [0.24, 0.06, 0.86],
            ],
            "Sphere",
        )

    def test_box_gradient(self):
        obstacle = Box((0.0, 0.0, 0.0), (0.12, 0.18, 0.28))
        self._check(
            obstacle,
            [
                [0.50, 0.00, 0.00],
                [0.00, -0.60, 0.00],
                [0.00, 0.00, 0.90],
                [0.30, 0.40, 0.50],
                [-0.35, -0.45, 0.55],
                [0.05, 0.02, 0.10],
                [-0.09, 0.05, -0.20],
            ],
            "Box",
        )

    def test_plane_gradient(self):
        obstacle = Plane((0.2, -0.5, 0.84), -0.30)
        self._check(
            obstacle,
            [
                [0.00, 0.00, 0.00],
                [1.00, -1.00, 0.50],
                [-0.40, 0.25, -0.90],
            ],
            "Plane",
        )

    def test_human_cylinder_gradient(self):
        obstacle = HumanCylinder(_human_config())
        self._check(
            obstacle,
            [
                [0.60, 0.10, 0.00],
                [-0.45, 0.20, -0.30],
                [0.10, 0.00, 0.00],
                [-0.08, 0.06, -0.50],
                [0.05, 0.00, 1.20],
                [0.05, 0.02, -1.50],
                [0.50, 0.00, 1.00],
                [-0.40, -0.30, -1.45],
            ],
            "HumanCylinder",
        )


class BoxDistanceTest(unittest.TestCase):
    def test_exterior_distance_is_exact(self):
        center = np.array([0.05, -0.10, 0.20])
        half = np.array([0.12, 0.18, 0.28])
        obstacle = Box(tuple(center), tuple(half))
        points = np.array([
            [0.90, -0.10, 0.20],
            [0.05, 0.80, 0.20],
            [0.05, -0.10, -0.90],
            [0.60, 0.40, 0.90],
            [-0.50, -0.70, -0.40],
        ])
        distance, _ = obstacle.distance_and_gradient(points)
        # Exact exterior distance: the nearest point of an axis-aligned
        # box is the query point clamped into the box.
        nearest = np.clip(points, center - half, center + half)
        np.testing.assert_allclose(
            distance,
            np.linalg.norm(points - nearest, axis=1),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertTrue(np.all(distance > 0.0))

    def test_interior_distance_is_negative_face_depth(self):
        center = np.array([0.05, -0.10, 0.20])
        half = np.array([0.12, 0.18, 0.28])
        obstacle = Box(tuple(center), tuple(half))
        points = np.array([
            [0.05, -0.10, 0.20],
            [0.10, -0.02, 0.30],
            [0.00, -0.20, 0.10],
            [-0.05, -0.24, 0.42],
        ])
        distance, _ = obstacle.distance_and_gradient(points)
        self.assertTrue(np.all(distance < 0.0))
        # Interior value is minus the distance to the nearest face.
        face_depth = np.min(half - np.abs(points - center), axis=1)
        np.testing.assert_allclose(
            distance, -face_depth, rtol=0.0, atol=1e-12
        )


class PlaneTest(unittest.TestCase):
    def test_sign_convention_and_normalisation(self):
        obstacle = Plane((0.0, 0.0, 3.0), 0.5)
        np.testing.assert_allclose(
            np.asarray(obstacle.normal_torso),
            [0.0, 0.0, 1.0],
            rtol=0.0,
            atol=1e-15,
        )
        points = np.array([
            [1.0, -2.0, 1.5],
            [0.0, 0.0, 0.5],
            [3.0, 4.0, -0.25],
        ])
        distance, gradient = obstacle.distance_and_gradient(points)
        # Positive above the plane (free space), zero on it, negative
        # inside the half-space the obstacle occupies.
        np.testing.assert_allclose(
            distance, [1.0, 0.0, -0.75], rtol=0.0, atol=1e-15
        )
        np.testing.assert_allclose(
            gradient, np.tile([0.0, 0.0, 1.0], (3, 1)), rtol=0.0, atol=1e-15
        )

    def test_rejects_zero_normal(self):
        with self.assertRaises(ValueError):
            Plane((0.0, 0.0, 0.0), 0.0)


class ObstacleSetTest(unittest.TestCase):
    def test_returns_minimum_distance_and_nearest_gradient(self):
        sphere = Sphere((0.40, 0.00, 0.20), 0.10)
        floor = Plane((0.0, 0.0, 1.0), -0.50)
        obstacle_set = ObstacleSet((sphere, floor))
        points = np.array([
            [0.90, 0.00, 0.20],
            [0.90, 0.00, -0.40],
            [0.40, 0.00, 0.70],
            [-0.60, 0.30, -0.45],
            [0.40, 0.25, 0.20],
        ])

        distance, gradient = obstacle_set.distance_and_gradient(points)
        sphere_distance, sphere_gradient = sphere.distance_and_gradient(
            points
        )
        floor_distance, floor_gradient = floor.distance_and_gradient(points)

        expected_distance = np.minimum(sphere_distance, floor_distance)
        np.testing.assert_allclose(
            distance, expected_distance, rtol=0.0, atol=1e-15
        )

        sphere_is_nearest = sphere_distance <= floor_distance
        expected_gradient = np.where(
            sphere_is_nearest[:, None], sphere_gradient, floor_gradient
        )
        np.testing.assert_allclose(
            gradient, expected_gradient, rtol=0.0, atol=1e-15
        )
        # The batch must actually exercise both branches, or the test
        # would pass on a set that always returned member zero.
        self.assertTrue(np.any(sphere_is_nearest))
        self.assertFalse(np.all(sphere_is_nearest))

    def test_empty_set_is_unbounded_clearance(self):
        distance, gradient = ObstacleSet(()).distance_and_gradient(
            np.zeros((2, 3))
        )
        self.assertTrue(np.all(np.isinf(distance)))
        np.testing.assert_allclose(gradient, np.zeros((2, 3)))


class WorldPlaneInTorsoTest(unittest.TestCase):
    def test_world_floor_maps_to_the_correct_torso_half_space(self):
        torso_pose_world = Pose(
            np.array([0.35, -0.20, 1.05]),
            rotation_from_rpy((0.20, -0.30, 0.50)),
        )
        plane = world_plane_in_torso(
            torso_pose_world, (0.0, 0.0, 1.0), 0.0
        )

        rng = np.random.default_rng(20260728)
        points_world = rng.uniform(-1.5, 1.5, size=(24, 3))
        points_torso = (
            points_world - torso_pose_world.position_m
        ) @ torso_pose_world.rotation

        distance, gradient = plane.distance_and_gradient(points_torso)
        # The torso-frame plane must report exactly the world height of
        # the same physical point.
        np.testing.assert_allclose(
            distance, points_world[:, 2], rtol=0.0, atol=1e-12
        )
        # And its outward direction must be world +z once rotated back.
        np.testing.assert_allclose(
            gradient @ torso_pose_world.rotation.T,
            np.tile([0.0, 0.0, 1.0], (len(points_world), 1)),
            rtol=0.0,
            atol=1e-12,
        )

        # Transform the other way as well: torso-frame points pushed out
        # to the world must keep the same signed height.
        points_torso_direct = rng.uniform(-1.0, 1.0, size=(12, 3))
        back_to_world = (
            points_torso_direct @ torso_pose_world.rotation.T
            + torso_pose_world.position_m
        )
        direct_distance, _ = plane.distance_and_gradient(points_torso_direct)
        np.testing.assert_allclose(
            direct_distance, back_to_world[:, 2], rtol=0.0, atol=1e-12
        )

    def test_offset_floor_shifts_the_half_space(self):
        torso_pose_world = Pose(
            np.array([0.0, 0.0, 1.20]), rotation_from_rpy((0.0, 0.0, 0.9))
        )
        plane = world_plane_in_torso(
            torso_pose_world, (0.0, 0.0, 1.0), 0.30
        )
        # A torso-frame point one metre below the torso origin sits at
        # world z=0.20, i.e. 0.10 m below a floor at world z=0.30.
        self.assertAlmostEqual(
            _distance(plane, np.array([0.0, 0.0, -1.0])), -0.10, places=12
        )


class SplineLinearityTest(unittest.TestCase):
    """The property the whole analytic Jacobian rests on."""

    def _assert_basis_reproduces_trajectory(self, times_s, seed):
        rng = np.random.default_rng(seed)
        knots = rng.uniform(-1.0, 1.0, size=(len(times_s), 3))
        sample_times = np.linspace(0.0, float(times_s[-1]), 37)

        trajectory = CartesianWaypointTrajectory(
            TargetFrame.WORLD,
            tuple(
                CartesianWaypoint(float(time_s), Pose(position, np.eye(3)))
                for time_s, position in zip(times_s, knots)
            ),
        )
        sampled = np.stack([
            trajectory.sample(float(time_s)).pose.position_m
            for time_s in sample_times
        ])

        basis = _spline_basis(np.asarray(times_s, dtype=float), sample_times)
        np.testing.assert_allclose(
            basis @ knots, sampled, rtol=0.0, atol=1e-12
        )
        # Partition of unity: a path whose knots are all the same point
        # must stay at that point, so every basis row sums to one.
        np.testing.assert_allclose(
            basis.sum(axis=1), np.ones(len(sample_times)),
            rtol=0.0,
            atol=1e-12,
        )

    def test_uniform_times(self):
        self._assert_basis_reproduces_trajectory(
            np.linspace(0.0, 1.0, 5), seed=11
        )

    def test_non_uniform_times(self):
        self._assert_basis_reproduces_trajectory(
            np.array([0.0, 0.7, 1.1, 2.6, 3.0, 4.4]), seed=12
        )

    def test_superposition_holds_for_the_real_trajectory_class(self):
        # Linearity restated without the basis: the sampled path of a sum
        # of two knot sets is the sum of their sampled paths.
        times_s = np.linspace(0.0, 1.0, 5)
        sample_times = np.linspace(0.0, 1.0, 21)
        rng = np.random.default_rng(13)
        left = rng.uniform(-1.0, 1.0, size=(5, 3))
        right = rng.uniform(-1.0, 1.0, size=(5, 3))

        def sampled(knots):
            trajectory = CartesianWaypointTrajectory(
                TargetFrame.WORLD,
                tuple(
                    CartesianWaypoint(
                        float(time_s), Pose(position, np.eye(3))
                    )
                    for time_s, position in zip(times_s, knots)
                ),
            )
            return np.stack([
                trajectory.sample(float(time_s)).pose.position_m
                for time_s in sample_times
            ])

        np.testing.assert_allclose(
            sampled(2.0 * left + 3.0 * right),
            2.0 * sampled(left) + 3.0 * sampled(right),
            rtol=0.0,
            atol=1e-12,
        )


class OptimizePathTest(unittest.TestCase):
    def test_path_through_the_wearer_is_pushed_clear(self):
        start, goal, torso_pose_world, obstacles = _crossing_scene()
        params = OptimizerParams()

        result = optimize_path(
            start, goal, torso_pose_world, obstacles, _limits(), params
        )

        self.assertLess(result.initial_min_clearance_m, 0.0)
        self.assertGreaterEqual(result.final_min_clearance_m, 0.0)
        # The optimiser aims for the hinge margin, not merely for
        # non-penetration; allow the module's own re-timing slack.
        self.assertGreaterEqual(
            result.final_min_clearance_m,
            params.clearance_margin_m - 1e-3,
        )
        self.assertTrue(result.success)

        # Endpoints are constraints, not variables.
        np.testing.assert_allclose(
            result.knots_torso_m[0], start.position_m, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            result.knots_torso_m[-1], goal.position_m, rtol=0.0, atol=1e-12
        )
        self.assertGreater(result.duration_s, 0.0)

    def test_already_clear_path_keeps_its_straight_line(self):
        torso_pose_world = Pose(np.zeros(3), np.eye(3))
        obstacles = ObstacleSet((HumanCylinder(_human_config()),))
        start = Pose(np.array([0.60, -0.40, 0.30]), np.eye(3))
        goal = Pose(np.array([0.60, 0.40, 0.30]), np.eye(3))
        params = OptimizerParams()

        result = optimize_path(
            start, goal, torso_pose_world, obstacles, _limits(), params
        )

        self.assertGreater(
            result.initial_min_clearance_m, params.clearance_margin_m
        )
        straight = np.linspace(
            start.position_m,
            goal.position_m,
            params.waypoint_count + 2,
        )
        np.testing.assert_allclose(
            result.knots_torso_m, straight, rtol=0.0, atol=1e-9
        )
        self.assertTrue(result.success)

    def test_plan_is_delivered_in_the_world_frame(self):
        torso_pose_world = Pose(
            np.array([0.10, -0.05, 1.00]),
            rotation_from_rpy((0.0, 0.0, 0.40)),
        )
        obstacles = ObstacleSet((HumanCylinder(_human_config()),))
        start_torso = np.array([0.60, -0.40, 0.10])
        goal_torso = np.array([0.60, 0.40, 0.10])

        def to_world(position_torso):
            return (
                torso_pose_world.rotation @ position_torso
                + torso_pose_world.position_m
            )

        result = optimize_path(
            Pose(to_world(start_torso), np.eye(3)),
            Pose(to_world(goal_torso), np.eye(3)),
            torso_pose_world,
            obstacles,
            _limits(),
        )

        np.testing.assert_allclose(
            result.knots_world_m,
            result.knots_torso_m @ torso_pose_world.rotation.T
            + torso_pose_world.position_m,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result.trajectory.sample(0.0).pose.position_m,
            result.knots_world_m[0],
            rtol=0.0,
            atol=1e-12,
        )


class OptimizerJacobianTest(unittest.TestCase):
    """The Gauss-Newton step is only as good as this matrix."""

    def test_analytic_jacobian_matches_finite_difference(self):
        start, goal, torso_pose_world, obstacles = _crossing_scene()
        _, captured = _solve_capturing_closures(
            start, goal, torso_pose_world, obstacles, _limits()
        )
        residuals = captured["residuals"]
        jacobian = captured["jacobian"]

        rng = np.random.default_rng(31)
        probes = [captured["initial"], captured["solution"]]
        for _ in range(4):
            probes.append(
                captured["initial"]
                + rng.normal(scale=0.05, size=captured["initial"].size)
            )

        for index, probe in enumerate(probes):
            analytic = jacobian(probe)
            numeric = _finite_difference_jacobian(residuals, probe)
            self.assertEqual(analytic.shape, numeric.shape)
            np.testing.assert_allclose(
                analytic,
                numeric,
                rtol=0.0,
                atol=_JACOBIAN_TOLERANCE,
                err_msg=f"Jacobian disagrees at probe {index}",
            )

    def test_inactive_obstacle_rows_are_exactly_zero(self):
        # A path that never comes near the margin has no obstacle
        # coupling at all, so the whole obstacle block must vanish.
        torso_pose_world = Pose(np.zeros(3), np.eye(3))
        obstacles = ObstacleSet((HumanCylinder(_human_config()),))
        _, captured = _solve_capturing_closures(
            Pose(np.array([0.60, -0.40, 0.30]), np.eye(3)),
            Pose(np.array([0.60, 0.40, 0.30]), np.eye(3)),
            torso_pose_world,
            obstacles,
            _limits(),
        )
        # Residual layout is [smoothness | deviation | obstacle]. The
        # deviation block sits between the other two and its Jacobian is
        # NOT zero, so the obstacle block starts after both.
        params = OptimizerParams()
        smooth_rows = (params.waypoint_count + 2 - 2) * 3
        deviation_rows = params.waypoint_count * 3
        obstacle_start = smooth_rows + deviation_rows

        block = captured["jacobian"](captured["initial"])
        np.testing.assert_allclose(
            block[obstacle_start:], 0.0, rtol=0.0, atol=0.0
        )
        residual = captured["residuals"](captured["initial"])
        np.testing.assert_allclose(
            residual[obstacle_start:], 0.0, rtol=0.0, atol=0.0
        )

        # The deviation block is exactly the weighted identity, and at the
        # straight-line start its residual is exactly zero.
        np.testing.assert_allclose(
            block[smooth_rows:obstacle_start],
            params.deviation_weight * np.eye(deviation_rows),
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            residual[smooth_rows:obstacle_start], 0.0, rtol=0.0, atol=0.0
        )


if __name__ == "__main__":
    unittest.main()
