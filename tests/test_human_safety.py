import unittest

import mujoco
import numpy as np

from controller import frames, human_safety
from controller.cylinder_router import CylinderKeepout
from controller.link_spheres import LINK_SPHERES
from controller.reactive_controller import constrain_velocity_for_human
from controller.runner import ReactivePositionRunner
from controller.state import (
    ArmHumanSafetyState,
    DualArmFramedTargets,
    FramedTarget,
    HumanDistanceConstraints,
    Pose,
    TargetFrame,
    Twist,
)
from controller.trajectory import StaticDualArmTargetSource
from runtime_config import CONFIG, HumanSafetyConfig
from sim.world import MujocoBackend


def _config(**changes):
    values = {
        "enabled": True,
        "center_xy_torso_m": (0.0, 0.0),
        "radius_m": 0.25,
        "z_min_torso_m": -1.1,
        "z_max_torso_m": 0.7,
        "clearance_m": 0.02,
        "control_margin_m": 0.02,
        "activation_distance_m": 0.10,
        "recovery_gain_s_inv": 4.0,
        "approach_velocity_damping": 0.0,
        "projection_iterations": 200,
        "constraint_tolerance_m_s": 1e-9,
    }
    values.update(changes)
    return HumanSafetyConfig(**values)


def _constraint(name, clearance, jacobian, active=True):
    return name, clearance, np.asarray(jacobian, dtype=float), active


def _safety_state(*items):
    return ArmHumanSafetyState(HumanDistanceConstraints(
        point_names=tuple(item[0] for item in items),
        signed_clearance_m=np.array([item[1] for item in items]),
        distance_jacobian_m_rad=np.stack([item[2] for item in items]),
        active=np.array([item[3] for item in items]),
        mount_exempt=np.zeros(len(items), dtype=bool),
    ))


class HumanCylinderGeometryTest(unittest.TestCase):
    def test_signed_distance_gradient_matches_finite_difference(self):
        config = _config()
        points = (
            np.array([0.40, 0.10, 0.0]),
            np.array([0.10, 0.10, 0.85]),
            np.array([0.40, 0.10, 0.85]),
            np.array([0.20, 0.05, 0.0]),
        )
        step = 1e-7
        for point in points:
            distance, gradient = (
                human_safety.finite_cylinder_distance_gradient(
                    point, config
                )
            )
            numerical = np.zeros(3)
            for axis in range(3):
                offset = np.zeros(3)
                offset[axis] = step
                plus, _ = human_safety.finite_cylinder_distance_gradient(
                    point + offset, config
                )
                minus, _ = human_safety.finite_cylinder_distance_gradient(
                    point - offset, config
                )
                numerical[axis] = (plus - minus) / (2.0 * step)
            self.assertTrue(np.isfinite(distance))
            np.testing.assert_allclose(
                gradient, numerical, rtol=0.0, atol=2e-8
            )

    def test_torso_rigid_motion_does_not_change_clearance(self):
        backend = MujocoBackend()
        plant_a = backend.read_state(Twist.zero())
        states_a = frames.controller_states(
            plant_a, backend.mount_calibration
        )
        clearances_a = human_safety.evaluate_dual_arm(
            plant_a, states_a, CONFIG.human_safety
        )

        mocap_id = int(
            backend.model.body_mocapid[backend.torso_body_id]
        )
        backend.data.mocap_pos[mocap_id] = np.array([0.4, -0.2, 1.5])
        quaternion = np.array([0.9, 0.2, -0.1, 0.3])
        quaternion /= np.linalg.norm(quaternion)
        backend.data.mocap_quat[mocap_id] = quaternion
        mujoco.mj_forward(backend.model, backend.data)
        plant_b = backend.read_state(Twist.zero())
        states_b = frames.controller_states(
            plant_b, backend.mount_calibration
        )
        clearances_b = human_safety.evaluate_dual_arm(
            plant_b, states_b, CONFIG.human_safety
        )

        for side in ("right", "left"):
            a = clearances_a.for_arm(side).constraints
            b = clearances_b.for_arm(side).constraints
            np.testing.assert_allclose(
                a.signed_clearance_m,
                b.signed_clearance_m,
                rtol=0.0,
                atol=2e-10,
            )
            np.testing.assert_allclose(
                a.distance_jacobian_m_rad,
                b.distance_jacobian_m_rad,
                rtol=0.0,
                atol=2e-10,
            )


class LinkSphereGroundTruthTest(unittest.TestCase):
    def test_sphere_chains_cover_every_collision_mesh_vertex(self):
        backend = MujocoBackend()
        model = backend.model
        by_frame = {}
        for sphere in LINK_SPHERES:
            by_frame.setdefault(sphere.frame_name, []).append(sphere)

        covered_frames = set()
        for geom_id in range(model.ngeom):
            if (
                int(model.geom_contype[geom_id]) == 0
                or int(model.geom_type[geom_id])
                != int(mujoco.mjtGeom.mjGEOM_MESH)
            ):
                continue
            body_id = int(model.geom_bodyid[geom_id])
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, body_id
            ) or ""
            if not body_name.startswith("right_"):
                continue
            frame_name = body_name.removeprefix("right_")
            self.assertIn(frame_name, by_frame)

            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(
                model.mesh_vert[start:start + count], dtype=float
            )
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(
                rotation, np.asarray(model.geom_quat[geom_id])
            )
            vertices_body = (
                vertices @ rotation.reshape(3, 3).T
                + np.asarray(model.geom_pos[geom_id])
            )
            distances = np.stack([
                np.linalg.norm(
                    vertices_body
                    - np.asarray(sphere.center_frame_m),
                    axis=1,
                )
                - sphere.radius_m
                for sphere in by_frame[frame_name]
            ])
            self.assertLessEqual(float(np.max(np.min(distances, axis=0))), 0.0)
            covered_frames.add(frame_name)

        self.assertEqual(covered_frames, set(by_frame))

    def test_point_positions_and_jacobians_match_mujoco(self):
        backend = MujocoBackend()
        rng = np.random.default_rng(20260727)
        sphere_by_name = {
            sphere.name: sphere for sphere in LINK_SPHERES
        }

        for _ in range(5):
            for side in ("right", "left"):
                backend.data.qpos[backend.qpos_adrs[side]] = rng.uniform(
                    -0.8, 0.8, 7
                )
            mocap_id = int(
                backend.model.body_mocapid[backend.torso_body_id]
            )
            backend.data.mocap_pos[mocap_id] = rng.uniform(-0.4, 0.4, 3)
            quaternion = rng.standard_normal(4)
            quaternion /= np.linalg.norm(quaternion)
            backend.data.mocap_quat[mocap_id] = quaternion
            mujoco.mj_forward(backend.model, backend.data)
            plant = backend.read_state(Twist.zero())

            for side in ("right", "left"):
                state = frames.arm_controller_state(
                    plant, side, backend.mount_calibration
                )
                points = state.link_safety_points
                for index, point_name in enumerate(points.names):
                    sphere = sphere_by_name[point_name]
                    body_id = mujoco.mj_name2id(
                        backend.model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        f"{side}_{points.frame_names[index]}",
                    )
                    expected_position = (
                        backend.data.xpos[body_id]
                        + backend.data.xmat[body_id].reshape(3, 3)
                        @ np.asarray(sphere.center_frame_m)
                    )
                    np.testing.assert_allclose(
                        points.position_world_m[index],
                        expected_position,
                        rtol=0.0,
                        atol=2e-9,
                    )

                    jacobian_position = np.zeros(
                        (3, backend.model.nv)
                    )
                    jacobian_rotation = np.zeros(
                        (3, backend.model.nv)
                    )
                    mujoco.mj_jac(
                        backend.model,
                        backend.data,
                        jacobian_position,
                        jacobian_rotation,
                        expected_position,
                        body_id,
                    )
                    expected_jacobian = jacobian_position[
                        :, backend.dof_adrs[side]
                    ]
                    np.testing.assert_allclose(
                        points.jacobian_world_m_rad[index],
                        expected_jacobian,
                        rtol=0.0,
                        atol=2e-8,
                    )


class SafetyProjectionTest(unittest.TestCase):
    def test_penetrated_point_is_commanded_outward(self):
        state = _safety_state(
            _constraint(
                "forearm",
                -0.02,
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        )
        result = constrain_velocity_for_human(
            np.array([-0.5, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.full(7, -1.0),
            np.full(7, 1.0),
            state,
            _config(),
        )
        self.assertFalse(result.stopped)
        self.assertTrue(result.human_adjusted)
        self.assertEqual(result.reason, "filtered")
        self.assertGreaterEqual(result.qdot_safe[0], 0.08 - 1e-9)

    def test_joint_velocity_bounds_and_distance_hold_together(self):
        state = _safety_state(
            _constraint(
                "wrist",
                0.01,
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        )
        result = constrain_velocity_for_human(
            np.array([-2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.full(7, -0.5),
            np.full(7, 0.5),
            state,
            _config(),
        )
        self.assertFalse(result.stopped)
        self.assertGreaterEqual(result.qdot_safe[0], -0.04 - 1e-9)
        self.assertLessEqual(float(np.max(result.qdot_safe)), 0.5)
        self.assertGreaterEqual(float(np.min(result.qdot_safe)), -0.5)

    def test_infeasible_penetration_returns_truthful_hold(self):
        state = _safety_state(
            _constraint(
                "opposed_a",
                -0.1,
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
            _constraint(
                "opposed_b",
                -0.1,
                [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ),
        )
        result = constrain_velocity_for_human(
            np.full(7, 0.2),
            np.full(7, -1.0),
            np.full(7, 1.0),
            state,
            _config(projection_iterations=20),
        )
        self.assertTrue(result.stopped)
        self.assertEqual(result.reason, "unsafe_initial_state_hold")
        np.testing.assert_array_equal(result.qdot_safe, np.zeros(7))

    def test_disabled_filter_preserves_requested_velocity(self):
        result = constrain_velocity_for_human(
            np.array([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            np.full(7, -0.5),
            np.full(7, 0.5),
            ArmHumanSafetyState(),
            _config(enabled=False),
        )
        np.testing.assert_array_equal(
            result.qdot_safe,
            np.array([2.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        )
        self.assertFalse(result.stopped)
        self.assertFalse(result.limit_adjusted)
        self.assertEqual(result.reason, "disabled")


class HeadlessHumanAvoidanceTest(unittest.TestCase):
    def test_inside_target_is_blocked_without_waypoint_router(self):
        backend = MujocoBackend()
        plant = backend.takeover()
        states = frames.controller_states(
            plant, backend.mount_calibration
        )
        backend.release()
        source = StaticDualArmTargetSource(DualArmFramedTargets(
            right=FramedTarget(
                TargetFrame.WORLD,
                Pose(
                    np.array([0.0, 0.0, 1.1]),
                    states.right.ee_pose_world.rotation,
                ),
                Twist.zero(),
            ),
            left=FramedTarget(
                TargetFrame.WORLD,
                states.left.ee_pose_world,
                Twist.zero(),
            ),
        ))
        runner = ReactivePositionRunner(
            backend,
            backend.mount_calibration,
            backend.pipeline_setup,
            source,
            arms=("right",),
            cylinder_keepout=CylinderKeepout(enabled=False),
            human_safety_config=CONFIG.human_safety,
        )
        minimum_clearance = float("inf")
        adjusted_cycles = 0
        stopped_cycles = 0
        torso_contacts = 0
        try:
            runner.start()
            for _ in range(1000):
                cycle = runner.cycle()
                status = cycle.human_safety_statuses["right"]
                minimum_clearance = min(
                    minimum_clearance, status.minimum_clearance_m
                )
                adjusted_cycles += int(status.human_adjusted)
                stopped_cycles += int(status.stopped)
                for contact_index in range(backend.data.ncon):
                    contact = backend.data.contact[contact_index]
                    body_names = {
                        mujoco.mj_id2name(
                            backend.model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            int(backend.model.geom_bodyid[geom_id]),
                        )
                        for geom_id in (contact.geom1, contact.geom2)
                    }
                    if (
                        "torso" in body_names
                        and any(
                            (name or "").startswith("right_")
                            for name in body_names
                        )
                    ):
                        torso_contacts += 1
        finally:
            runner.close()

        self.assertGreater(minimum_clearance, 0.0)
        self.assertGreater(adjusted_cycles, 0)
        self.assertEqual(stopped_cycles, 0)
        self.assertEqual(torso_contacts, 0)


if __name__ == "__main__":
    unittest.main()
