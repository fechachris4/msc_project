"""Desired end-effector poses — the reference the controller will track.

The simple case: state a world-frame position, optionally an orientation,
and you get a full pose back. A record is just:

    {"pos": [x, y, z]}                          # world frame, no rotation
    {"pos": [...], "rpy": [roll, pitch, yaw]}   # world frame, oriented
    {"frame": "torso", "pos": [...]}            # opt-in: torso-relative

Positions in meters; rpy in radians, R = Rz(yaw) @ Ry(pitch) @ Rx(roll);
rpy omitted means identity orientation.

resolve_world() is pure (never reads MuJoCo): record in, world (pos, rot)
out. apply() resolves both records against the current torso pose and
writes them into the target mocap bodies once — a torso-frame target does
not follow the torso afterwards.
"""

import mujoco
import numpy as np

from controller import frames
from controller.transforms import (
    pose_from_transform,
    rotation_from_rpy,
    transform_from_pose,
)
from sim import targets

# World frame. Same points as before: 0.45 m in front of the torso
# (torso is at world z = 1.1), 0.20 m to each side, 0.10 m above center.
RIGHT = {"pos": [0.45, -0.20, 1.20]}
LEFT = {"pos": [0.45, 0.20, 1.20]}


def resolve_world(ref, torso_pose=None):
    """World-frame (pos, rot) for a reference record.

    torso_pose is a (position, rotation matrix) pair for the torso in the
    world frame; only required when ref["frame"] == "torso".
    """
    frame = ref.get("frame", "world")
    T = transform_from_pose(
        np.asarray(ref["pos"], dtype=float),
        rotation_from_rpy(ref.get("rpy", [0.0, 0.0, 0.0])),
    )
    if frame == "torso":
        if torso_pose is None:
            raise ValueError("torso-frame reference needs a torso_pose")
        T = transform_from_pose(*torso_pose) @ T
    elif frame != "world":
        raise ValueError(f"frame must be 'world' or 'torso', got {frame!r}")
    return pose_from_transform(T)


def _quat_from_rotation(rot):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def apply():
    torso_pose = frames.torso_pose()

    pos, rot = resolve_world(RIGHT, torso_pose)
    targets.set_right_target(pos)
    targets.set_right_target_quat(_quat_from_rotation(rot))

    pos, rot = resolve_world(LEFT, torso_pose)
    targets.set_left_target(pos)
    targets.set_left_target_quat(_quat_from_rotation(rot))
