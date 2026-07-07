"""Reactive controller. Currently: pose error between the world-frame
reference (the target mocap bodies, set once by desired_pos.apply()) and
the FK end-effector pose from frames — no EE pose is read from MuJoCo.

Error convention: e = reference - actual, expressed in the world frame.
e_pos in meters. e_rot is the axis-angle vector (radians) of
R_err = R_ref @ R_ee.T, the world-frame rotation taking the actual
orientation to the reference.
"""

import numpy as np
import pinocchio as pin

from controller import frames
from controller.transforms import rotation_from_quat
from sim import targets


def position_error(ref_pos, ee_pos):
    """World-frame position error, meters: reference - actual."""
    return np.asarray(ref_pos, dtype=float) - np.asarray(ee_pos, dtype=float)


def rotation_error(ref_rot, ee_rot):
    """World-frame axis-angle error, radians: log3(R_ref @ R_ee.T)."""
    return pin.log3(ref_rot @ ee_rot.T)


def right_pose_error():
    """(e_pos, e_rot) of the right arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.right_ee_pose()
    ref_pos = targets.right_target_position()
    ref_rot = rotation_from_quat(targets.right_target_quat())
    return position_error(ref_pos, ee_pos), rotation_error(ref_rot, ee_rot)


def left_pose_error():
    """(e_pos, e_rot) of the left arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.left_ee_pose()
    ref_pos = targets.left_target_position()
    ref_rot = rotation_from_quat(targets.left_target_quat())
    return position_error(ref_pos, ee_pos), rotation_error(ref_rot, ee_rot)
