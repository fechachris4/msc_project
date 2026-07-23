"""Desired end-effector poses — the reference the controller will track.

Target placements come from config/control.toml: pos [x, y, z] in metres,
rpy [roll, pitch, yaw] in radians (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)).

apply() returns retained framed targets and initializes their world-frame mocap
markers. The Runner resolves retained targets from the latest PlantState every
cycle; controller math receives only world-frame quantities.
"""

import mujoco
import numpy as np

from controller import frames
from controller.state import (
    DualArmFramedTargets,
    DualArmWorldTargets,
    FramedTarget,
    Pose,
    TargetFrame,
    Twist,
)
from controller.transforms import rotation_from_rpy
from runtime_config import CONFIG
from sim import targets, world


def _quat_from_rotation(rot):
    """Rotation matrix -> MuJoCo quaternion [w, x, y, z]."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def configured_targets(config=CONFIG):
    values = {}
    for side in world.SIDES:
        target = config.target(side)
        values[side] = FramedTarget(
            TargetFrame(target.reference_frame),
            Pose(target.position_m, rotation_from_rpy(target.rpy_rad)),
            Twist.zero(),
        )
    return DualArmFramedTargets(values["right"], values["left"])


def show_targets(world_targets):
    if not isinstance(world_targets, DualArmWorldTargets):
        raise TypeError("world_targets must be DualArmWorldTargets")
    for side in world.SIDES:
        resolved = world_targets.for_arm(side)
        targets.set_target(side, resolved.pose_world.position_m)
        targets.set_target_quat(
            side, _quat_from_rotation(resolved.pose_world.rotation))


def apply(config=CONFIG):
    """Initialize target markers and return the retained framed targets."""
    source_targets = configured_targets(config)
    plant = world.read_state(Twist.zero())
    show_targets(frames.resolve_targets_world(
        plant, world.MOUNT_CALIBRATION, source_targets))
    return source_targets
