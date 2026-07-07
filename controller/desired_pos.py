"""Desired end-effector poses — the reference the controller will track.

RIGHT and LEFT are torso-frame initial placements: pos [x, y, z] in meters,
rpy [roll, pitch, yaw] in radians (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)).

apply() resolves them against the torso pose once and writes world-frame
poses into the target mocap bodies. The targets stay fixed in the world
afterwards — that is the task: world-frame pose hold while the base moves.
"""

import mujoco
import numpy as np

from controller import frames
from controller.transforms import rotation_from_rpy
from sim import targets

RIGHT = {"pos": [0.45, -0.20, 0.10], "rpy": [0.0, 0.0, 0.0]}
LEFT = {"pos": [0.45, 0.20, 0.10], "rpy": [0.0, 0.0, 0.0]}


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


def apply():
    torso_pose = frames.torso_pose()

    pos, rot = resolve_world(RIGHT["pos"], RIGHT["rpy"], torso_pose)
    targets.set_right_target(pos)
    targets.set_right_target_quat(_quat_from_rotation(rot))

    pos, rot = resolve_world(LEFT["pos"], LEFT["rpy"], torso_pose)
    targets.set_left_target(pos)
    targets.set_left_target_quat(_quat_from_rotation(rot))
