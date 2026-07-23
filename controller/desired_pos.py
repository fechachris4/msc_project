"""Desired end-effector poses — the reference the controller will track.

Target placements come from config/control.toml: pos [x, y, z] in metres,
rpy [roll, pitch, yaw] in radians (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)).

apply() resolves them against the torso pose once and writes world-frame
poses into the target mocap bodies. The targets stay fixed in the world
afterwards — that is the task: world-frame pose hold while the base moves.
"""

import mujoco
import numpy as np

from controller import frames
from controller.transforms import rotation_from_rpy
from runtime_config import CONFIG
from sim import targets, world


def resolve_world(pos, rpy, torso_pose):
    """World-frame (pos, rot) of a torso-frame reference."""
    torso_pos, torso_rot = torso_pose
    world_pos = torso_pos + torso_rot @ np.asarray(pos, dtype=float)
    world_rot = torso_rot @ rotation_from_rpy(rpy)
    return world_pos, world_rot


def _quat_from_rotation(rot):
    """Rotation matrix -> MuJoCo quaternion [w, x, y, z]."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def apply(config=CONFIG):
    torso_pose = frames.torso_pose()
    for side in world.SIDES:
        target = config.target(side)
        if target.reference_frame != "torso":
            raise ValueError(
                "the pre-Step-3 target boundary only supports torso targets"
            )
        pos, rot = resolve_world(target.position_m, target.rpy_rad, torso_pose)
        targets.set_target(side, pos)
        targets.set_target_quat(side, _quat_from_rotation(rot))
