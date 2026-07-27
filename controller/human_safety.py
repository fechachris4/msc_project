"""Torso-attached human envelope and whole-arm distance constraints.

This module owns geometry only.  Controller policy remains in
``reactive_controller.py``.  The finite cylinder is expressed in the torso
frame, so both it and the arm mounts share torso translation and rotation.
Only joint-induced relative motion appears in the distance Jacobian.
"""

import numpy as np

from controller.state import (
    ArmHumanSafetyState,
    DualArmControllerStates,
    DualArmHumanSafetyStates,
    HumanDistanceConstraints,
    Pose,
)
from runtime_config import HumanSafetyConfig


_AXIS_EPSILON = 1e-12


def finite_cylinder_distance_gradient(point_torso_m, config):
    """Signed distance and outward gradient for a capped torso-z cylinder."""
    if not isinstance(config, HumanSafetyConfig):
        raise TypeError("config must be a HumanSafetyConfig")
    point = np.asarray(point_torso_m, dtype=float)
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("point_torso_m must be a finite shape-(3,) array")

    radial_vector = point[:2] - np.asarray(
        config.center_xy_torso_m, dtype=float
    )
    radial_distance = float(np.linalg.norm(radial_vector))
    radial_gradient = np.array([1.0, 0.0, 0.0])
    if radial_distance > _AXIS_EPSILON:
        radial_gradient[:2] = radial_vector / radial_distance

    radial_signed = radial_distance - config.radius_m
    below = config.z_min_torso_m - point[2]
    above = point[2] - config.z_max_torso_m
    vertical_signed = max(below, above)
    vertical_gradient = np.array([
        0.0,
        0.0,
        -1.0 if below >= above else 1.0,
    ])

    if radial_signed > 0.0 and vertical_signed > 0.0:
        outside = np.array([radial_signed, vertical_signed])
        distance = float(np.linalg.norm(outside))
        gradient = (
            radial_gradient * (radial_signed / distance)
            + vertical_gradient * (vertical_signed / distance)
        )
        return distance, gradient
    if radial_signed >= vertical_signed:
        return radial_signed, radial_gradient
    return vertical_signed, vertical_gradient


def _finite_cylinder_distance_gradient_batch(points_torso_m, config):
    points = np.asarray(points_torso_m, dtype=float)
    radial_vectors = (
        points[:, :2]
        - np.asarray(config.center_xy_torso_m, dtype=float)
    )
    radial_distances = np.linalg.norm(radial_vectors, axis=1)
    radial_gradients = np.zeros((len(points), 3))
    radial_gradients[:, 0] = 1.0
    off_axis = radial_distances > _AXIS_EPSILON
    radial_gradients[off_axis, :2] = (
        radial_vectors[off_axis]
        / radial_distances[off_axis, None]
    )

    radial_signed = radial_distances - config.radius_m
    below = config.z_min_torso_m - points[:, 2]
    above = points[:, 2] - config.z_max_torso_m
    vertical_signed = np.maximum(below, above)
    vertical_gradients = np.zeros((len(points), 3))
    vertical_gradients[:, 2] = np.where(below >= above, -1.0, 1.0)

    use_radial = radial_signed >= vertical_signed
    distances = np.where(use_radial, radial_signed, vertical_signed)
    gradients = np.where(
        use_radial[:, None], radial_gradients, vertical_gradients
    )

    corners = np.logical_and(
        radial_signed > 0.0, vertical_signed > 0.0
    )
    if np.any(corners):
        corner_distance = np.hypot(
            radial_signed[corners], vertical_signed[corners]
        )
        distances[corners] = corner_distance
        gradients[corners] = (
            radial_gradients[corners]
            * (radial_signed[corners] / corner_distance)[:, None]
            + vertical_gradients[corners]
            * (vertical_signed[corners] / corner_distance)[:, None]
        )
    return distances, gradients


def evaluate_arm(plant_torso_pose_world, arm_state, config):
    """Build distance-rate constraints from one explicit plant sample."""
    if not isinstance(plant_torso_pose_world, Pose):
        raise TypeError("plant_torso_pose_world must be a Pose")
    if not isinstance(config, HumanSafetyConfig):
        raise TypeError("config must be a HumanSafetyConfig")

    R_T_W = plant_torso_pose_world.rotation.T
    torso_position_world = plant_torso_pose_world.position_m
    points = arm_state.link_safety_points
    if points is None:
        raise ValueError("arm_state has no link safety geometry")
    point_torso = np.einsum(
        "ij,nj->ni",
        R_T_W,
        points.position_world_m - torso_position_world,
    )
    jacobian_torso = np.einsum(
        "ij,njk->nik", R_T_W, points.jacobian_world_m_rad
    )
    center_distance, gradient_torso = (
        _finite_cylinder_distance_gradient_batch(point_torso, config)
    )
    signed_clearance = (
        center_distance - points.radius_m - config.clearance_m
    )
    distance_jacobian = np.einsum(
        "ni,nij->nj", gradient_torso, jacobian_torso
    )
    active = np.logical_and.reduce((
        np.full(len(points), config.enabled, dtype=bool),
        ~points.mount_exempt,
        signed_clearance <= config.activation_distance_m,
    ))
    return ArmHumanSafetyState(HumanDistanceConstraints(
        point_names=points.names,
        signed_clearance_m=signed_clearance,
        distance_jacobian_m_rad=distance_jacobian,
        active=active,
        mount_exempt=points.mount_exempt,
    ))


def evaluate_dual_arm(plant, controller_states, config):
    if not isinstance(controller_states, DualArmControllerStates):
        raise TypeError("controller_states must be DualArmControllerStates")
    return DualArmHumanSafetyStates(
        right=evaluate_arm(
            plant.torso_pose_world, controller_states.right, config
        ),
        left=evaluate_arm(
            plant.torso_pose_world, controller_states.left, config
        ),
    )
