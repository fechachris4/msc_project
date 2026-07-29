"""Torso-frame obstacle set with signed distances and gradients.

Every obstacle answers the same question for a batch of points:

    distance_and_gradient(points_torso_m) -> (distance_m, gradient)

``distance_m`` is the signed distance from the point to the obstacle
SURFACE, positive outside.  ``gradient`` is the unit outward direction,
i.e. d(distance)/d(point).  An ``ObstacleSet`` returns the elementwise
minimum over its members, which is the clearance the planner must keep
positive.

The human envelope reuses ``controller.human_safety`` verbatim, so the
planner and the real-time safety filter cannot disagree about where the
wearer is.  The remaining obstacles (floor, torso box, the other arm)
exist because nothing else in the project models them at all.
"""

from dataclasses import dataclass

import numpy as np

from controller.human_safety import _finite_cylinder_distance_gradient_batch
from controller.state import Pose
from runtime_config import HumanSafetyConfig


def _points(points_torso_m):
    points = np.asarray(points_torso_m, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(
        np.isfinite(points)
    ):
        raise ValueError("points_torso_m must be a finite (N, 3) array")
    return points


@dataclass(frozen=True, slots=True)
class HumanCylinder:
    """The wearer envelope, identical to the real-time safety filter's."""

    config: HumanSafetyConfig

    def __post_init__(self):
        if not isinstance(self.config, HumanSafetyConfig):
            raise TypeError("config must be a HumanSafetyConfig")

    def distance_and_gradient(self, points_torso_m):
        points = _points(points_torso_m)
        distance, gradient = _finite_cylinder_distance_gradient_batch(
            points, self.config
        )
        # The safety filter subtracts clearance_m from the same surface, so
        # planner clearance and filter signed_clearance are the same number.
        # They are NOT the filter's activation boundary: the filter arms at
        # signed_clearance <= activation_distance_m (0.10 m by default), so
        # a plan sitting near zero clearance is already inside the band
        # where the filter will act.
        return distance - self.config.clearance_m, gradient


@dataclass(frozen=True, slots=True)
class Sphere:
    """A ball obstacle, e.g. a coarse stand-in for the other arm."""

    center_torso_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self):
        center = np.asarray(self.center_torso_m, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_torso_m must be a finite shape-(3,)")
        radius = float(self.radius_m)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius_m must be finite and non-negative")
        object.__setattr__(self, "center_torso_m", tuple(center.tolist()))
        object.__setattr__(self, "radius_m", radius)

    def distance_and_gradient(self, points_torso_m):
        points = _points(points_torso_m)
        offset = points - np.asarray(self.center_torso_m, dtype=float)
        norm = np.linalg.norm(offset, axis=1)
        gradient = np.zeros_like(offset)
        gradient[:, 0] = 1.0
        off_center = norm > 1e-12
        gradient[off_center] = offset[off_center] / norm[off_center, None]
        return norm - self.radius_m, gradient


@dataclass(frozen=True, slots=True)
class Box:
    """An axis-aligned torso-frame box, e.g. the torso collision geom.

    The distance is exact outside the box and a conservative negative
    (largest face penetration) inside it, which is all the hinge cost
    needs to push a point back out.
    """

    center_torso_m: tuple[float, float, float]
    half_extent_m: tuple[float, float, float]

    def __post_init__(self):
        center = np.asarray(self.center_torso_m, dtype=float)
        half = np.asarray(self.half_extent_m, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("center_torso_m must be a finite shape-(3,)")
        if half.shape != (3,) or not np.all(np.isfinite(half)) or np.any(
            half < 0.0
        ):
            raise ValueError("half_extent_m must be finite and non-negative")
        object.__setattr__(self, "center_torso_m", tuple(center.tolist()))
        object.__setattr__(self, "half_extent_m", tuple(half.tolist()))

    def distance_and_gradient(self, points_torso_m):
        points = _points(points_torso_m)
        offset = points - np.asarray(self.center_torso_m, dtype=float)
        half = np.asarray(self.half_extent_m, dtype=float)
        excess = np.abs(offset) - half
        outside = np.maximum(excess, 0.0)
        outside_norm = np.linalg.norm(outside, axis=1)

        gradient = np.zeros_like(offset)
        exterior = outside_norm > 1e-12
        gradient[exterior] = (
            np.sign(offset[exterior])
            * outside[exterior]
            / outside_norm[exterior, None]
        )

        # Interior points: push out along the least-penetrated axis.
        # Chained fancy indexing would write to a copy, so the sign fix is
        # computed first and assigned once.
        interior = ~exterior
        if np.any(interior):
            rows = np.flatnonzero(interior)
            axis = np.argmax(excess[rows], axis=1)
            direction = np.sign(offset[rows, axis])
            direction[direction == 0.0] = 1.0
            gradient[rows, axis] = direction

        distance = np.where(
            exterior, outside_norm, np.max(excess, axis=1)
        )
        return distance, gradient


@dataclass(frozen=True, slots=True)
class Plane:
    """A half-space obstacle: everything below ``normal . x = offset``."""

    normal_torso: tuple[float, float, float]
    offset_m: float

    def __post_init__(self):
        normal = np.asarray(self.normal_torso, dtype=float)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("normal_torso must be a finite shape-(3,)")
        norm = float(np.linalg.norm(normal))
        if norm < 1e-12:
            raise ValueError("normal_torso must be non-zero")
        object.__setattr__(
            self, "normal_torso", tuple((normal / norm).tolist())
        )
        object.__setattr__(self, "offset_m", float(self.offset_m))

    def distance_and_gradient(self, points_torso_m):
        points = _points(points_torso_m)
        normal = np.asarray(self.normal_torso, dtype=float)
        distance = points @ normal - self.offset_m
        gradient = np.broadcast_to(normal, points.shape).copy()
        return distance, gradient


@dataclass(frozen=True, slots=True)
class ObstacleSet:
    """Elementwise-minimum clearance over a fixed list of obstacles."""

    obstacles: tuple

    def __post_init__(self):
        obstacles = tuple(self.obstacles)
        for item in obstacles:
            if not hasattr(item, "distance_and_gradient"):
                raise TypeError(
                    "every obstacle must provide distance_and_gradient"
                )
        object.__setattr__(self, "obstacles", obstacles)

    def __len__(self):
        return len(self.obstacles)

    def distance_and_gradient(self, points_torso_m):
        points = _points(points_torso_m)
        if not self.obstacles:
            return (
                np.full(len(points), np.inf),
                np.zeros_like(points),
            )
        distances = np.empty((len(self.obstacles), len(points)))
        gradients = np.empty((len(self.obstacles), len(points), 3))
        for index, obstacle in enumerate(self.obstacles):
            distances[index], gradients[index] = (
                obstacle.distance_and_gradient(points)
            )
        nearest = np.argmin(distances, axis=0)
        rows = np.arange(len(points))
        return distances[nearest, rows], gradients[nearest, rows]


def world_plane_in_torso(torso_pose_world, normal_world, offset_world_m):
    """Express a world half-space in torso coordinates.

    The floor is world-fixed, so under torso motion its torso-frame
    description changes; the planner therefore rebuilds the obstacle set
    from the torso pose captured when the plan was made.
    """
    if not isinstance(torso_pose_world, Pose):
        raise TypeError("torso_pose_world must be a Pose")
    normal = np.asarray(normal_world, dtype=float)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("normal_world must be a finite shape-(3,) array")
    norm = float(np.linalg.norm(normal))
    if norm < 1e-12:
        raise ValueError("normal_world must be non-zero")
    # Normalise BEFORE computing the offset: Plane stores a unit normal, so
    # an un-normalised input would otherwise place the plane at
    # offset/|n| while the offset was computed for |n|.
    unit = normal / norm
    normal_torso = torso_pose_world.rotation.T @ unit
    offset_torso = float(offset_world_m) / norm - float(
        unit @ torso_pose_world.position_m
    )
    return Plane(tuple(normal_torso.tolist()), offset_torso)


def build_obstacle_set(
    human_safety_config,
    torso_pose_world,
    floor_height_world_m=None,
    torso_box_half_extent_m=None,
    extra_spheres=(),
):
    """Assemble the planner's torso-frame obstacle set.

    ``torso_pose_world`` is the torso pose AT PLAN TIME.  The human
    cylinder is torso-static and therefore exact for the life of the
    plan; the floor is world-static and is only valid while the torso
    stays near that captured pose, which is what the replan trigger
    exists to detect.
    """
    obstacles = [HumanCylinder(human_safety_config)]
    if torso_box_half_extent_m is not None:
        obstacles.append(Box((0.0, 0.0, 0.0), torso_box_half_extent_m))
    if floor_height_world_m is not None:
        obstacles.append(
            world_plane_in_torso(
                torso_pose_world, (0.0, 0.0, 1.0), floor_height_world_m
            )
        )
    for sphere in extra_spheres:
        if not isinstance(sphere, Sphere):
            raise TypeError("extra_spheres must contain Sphere values")
        obstacles.append(sphere)
    return ObstacleSet(tuple(obstacles))
