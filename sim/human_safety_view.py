"""Viewer-only rendering for the torso-attached whole-arm safety model."""

import mujoco
import numpy as np

from runtime_config import HumanSafetyConfig


HUMAN_RGBA = (0.95, 0.30, 0.15, 0.22)
CLEARANCE_RGBA = (0.20, 0.65, 1.00, 0.09)
SAFE_SPHERE_RGBA = (0.15, 0.95, 0.35, 0.13)
ACTIVE_SPHERE_RGBA = (1.00, 0.80, 0.10, 0.28)
UNSAFE_SPHERE_RGBA = (1.00, 0.05, 0.05, 0.55)
MOUNT_SPHERE_RGBA = (0.55, 0.55, 0.60, 0.10)


def describe(config):
    if not isinstance(config, HumanSafetyConfig):
        raise TypeError("config must be a HumanSafetyConfig")
    state = "ENABLED" if config.enabled else "DISABLED"
    return (
        f"whole-arm human safety: {state}  frame=TORSO  axis=torso +z  "
        f"radius={config.radius_m:.3f} m  "
        f"z=[{config.z_min_torso_m:.3f}, "
        f"{config.z_max_torso_m:.3f}] m  "
        f"clearance={config.clearance_m:.3f} m  "
        f"control_margin={config.control_margin_m:.3f} m\n"
        "  18 mesh-covering spheres per arm; 13 active, 5 intentional "
        "mount-interface exemptions"
    )


def _add_geom(scene):
    if scene.ngeom >= scene.maxgeom:
        return None
    geom = scene.geoms[scene.ngeom]
    scene.ngeom += 1
    return geom


def _add_cylinder(scene, torso_pose_world, config, expansion, rgba):
    geom = _add_geom(scene)
    if geom is None:
        return
    z_low = config.z_min_torso_m - expansion
    z_high = config.z_max_torso_m + expansion
    center_torso = np.array([
        config.center_xy_torso_m[0],
        config.center_xy_torso_m[1],
        0.5 * (z_low + z_high),
    ])
    center_world = (
        torso_pose_world.position_m
        + torso_pose_world.rotation @ center_torso
    )
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        np.array([
            config.radius_m + expansion,
            0.5 * (z_high - z_low),
            0.0,
        ]),
        center_world,
        torso_pose_world.rotation.flatten(),
        np.array(rgba, dtype=np.float32),
    )


def _add_sphere(scene, position_world_m, radius_m, rgba):
    geom = _add_geom(scene)
    if geom is None:
        return
    mujoco.mjv_initGeom(
        geom,
        int(mujoco.mjtGeom.mjGEOM_SPHERE),
        np.array([radius_m, 0.0, 0.0]),
        np.asarray(position_world_m, dtype=float),
        np.eye(3).flatten(),
        np.array(rgba, dtype=np.float32),
    )


def draw(scene, plant, controller_states, safety_states, config, sides):
    """Draw the exact moving envelope and every conservative arm sphere."""
    before = scene.ngeom
    if not config.enabled:
        return 0
    _add_cylinder(scene, plant.torso_pose_world, config, 0.0, HUMAN_RGBA)
    _add_cylinder(
        scene,
        plant.torso_pose_world,
        config,
        config.clearance_m + config.control_margin_m,
        CLEARANCE_RGBA,
    )
    for side in sides:
        arm_state = controller_states.for_arm(side)
        points = arm_state.link_safety_points
        constraints = safety_states.for_arm(side).constraints
        for index in range(len(points)):
            if points.mount_exempt[index]:
                color = MOUNT_SPHERE_RGBA
            elif constraints.signed_clearance_m[index] < 0.0:
                color = UNSAFE_SPHERE_RGBA
            elif constraints.active[index]:
                color = ACTIVE_SPHERE_RGBA
            else:
                color = SAFE_SPHERE_RGBA
            _add_sphere(
                scene,
                points.position_world_m[index],
                points.radius_m[index],
                color,
            )
    return scene.ngeom - before
