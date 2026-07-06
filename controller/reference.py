"""Desired end-effector poses — the reference the controller will track.

Each reference is one record: a pose expressed in a named frame.

  frame: "world" — absolute pose, world frame (the default interpretation)
         "torso" — pose relative to the torso body
  pos:   [x, y, z] in meters
  rpy:   [roll, pitch, yaw] in radians, R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

resolve_world() is the pure primitive: record + torso pose in, world pose
out. It never reads MuJoCo state, so a controller can call it every tick
(e.g. to keep a torso-frame target attached to a moving torso).

apply() is the one-shot convenience: resolve both records against the torso
pose *right now* and write them into the target mocap bodies. A torso-frame
target does NOT follow the torso afterwards — call apply() again to update.
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

RIGHT = {"frame": "torso", "pos": [0.45, -0.20, 0.10], "rpy": [0.0, 0.0, 0.0]}
LEFT = {"frame": "torso", "pos": [0.45, 0.20, 0.10], "rpy": [0.0, 0.0, 0.0]}


def resolve_world(ref, torso_pose=None):
    """World-frame (pos, rot) for a reference record.

    torso_pose is a (position, rotation matrix) pair for the torso in the
    world frame; only required when ref["frame"] == "torso".
    """
    frame = ref.get("frame", "world")
    T = transform_from_pose(
        np.asarray(ref["pos"], dtype=float), rotation_from_rpy(ref["rpy"])
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
