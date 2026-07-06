"""Desired end-effector poses — the reference the controller will track.

Edit the values below, then call apply() to write them into the target
mocap bodies. This file only sets the targets; it does no control.

FRAME selects how the poses are interpreted:
  "world" — absolute pose, world frame
  "torso" — pose relative to the torso body; converted to world using the
            torso pose at the moment apply() is called (the target does NOT
            follow the torso afterwards — call apply() again to update)

Positions are [x, y, z] in meters. Orientations are [roll, pitch, yaw] in
radians, composed as R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

import mujoco
import numpy as np

from controller import kinematics
from sim import targets

FRAME = "torso"

RIGHT_POS = [0.45, -0.20, 0.10]
RIGHT_RPY = [0.0, 0.0, 0.0]

LEFT_POS = [0.45, 0.20, 0.10]
LEFT_RPY = [0.0, 0.0, 0.0]


def rotation_from_rpy(rpy):
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _quat_from_rotation(rot):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    return quat


def _world_pose(pos, rpy):
    T = kinematics.transform_from_pose(np.asarray(pos, dtype=float), rotation_from_rpy(rpy))
    if FRAME == "torso":
        T_W_T = kinematics.transform_from_pose(*kinematics.torso_mocap_position())
        T = T_W_T @ T
    elif FRAME != "world":
        raise ValueError(f"FRAME must be 'world' or 'torso', got {FRAME!r}")
    return kinematics.pose_from_transform(T)


def apply():
    pos, rot = _world_pose(RIGHT_POS, RIGHT_RPY)
    targets.set_right_target(pos)
    targets.set_right_target_quat(_quat_from_rotation(rot))

    pos, rot = _world_pose(LEFT_POS, LEFT_RPY)
    targets.set_left_target(pos)
    targets.set_left_target_quat(_quat_from_rotation(rot))
