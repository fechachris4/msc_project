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
from sim import targets, world


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


# --- Proportional controller (resolved-rate) ---------------------------
# v = Kp * e is a commanded world twist; qdot comes from a damped-least-
# squares solve; the servo setpoint data.ctrl is the integrator state
# (ctrl += qdot*dt), so at the fixed point qdot = 0 => e -> 0, and the
# servos' gravity droop is compensated automatically.

KP_POS = 2.0    # 1/s task-space bandwidth
KP_ROT = 2.0    # 1/s
DAMPING = 0.05  # DLS lambda


def _ctrl_bounds(ctrl_adrs):
    low = np.full(len(ctrl_adrs), -np.inf)
    high = np.full(len(ctrl_adrs), np.inf)
    for i, adr in enumerate(ctrl_adrs):
        if world.model.actuator_ctrllimited[adr]:
            low[i], high[i] = world.model.actuator_ctrlrange[adr]
    return low, high


_RIGHT_BOUNDS = _ctrl_bounds(world.right_ctrl_adrs)
_LEFT_BOUNDS = _ctrl_bounds(world.left_ctrl_adrs)


def qdot_from_error(J, e_pos, e_rot, damping=DAMPING):
    """Joint rates from the world-frame pose error: DLS solve of v = Kp*e."""
    v = np.concatenate([KP_POS * np.asarray(e_pos, dtype=float),
                        KP_ROT * np.asarray(e_rot, dtype=float)])
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), v)


def init_ctrl():
    """Sync servo setpoints to the current joint angles (once, pre-loop)."""
    world.data.ctrl[world.right_ctrl_adrs] = world.data.qpos[frames.right_qpos_adrs]
    world.data.ctrl[world.left_ctrl_adrs] = world.data.qpos[frames.left_qpos_adrs]


def _ctrl(dt, pose_error, jacobian_world, ctrl_adrs, bounds):
    e_pos, e_rot = pose_error()
    qdot = qdot_from_error(jacobian_world(), e_pos, e_rot)
    ctrl = world.data.ctrl[ctrl_adrs] + qdot * dt
    return np.clip(ctrl, bounds[0], bounds[1])


def right_ctrl(dt):
    """Updated right-arm servo targets (does not write data.ctrl)."""
    return _ctrl(dt, right_pose_error, frames.right_jacobian_world,
                 world.right_ctrl_adrs, _RIGHT_BOUNDS)


def left_ctrl(dt):
    """Updated left-arm servo targets (does not write data.ctrl)."""
    return _ctrl(dt, left_pose_error, frames.left_jacobian_world,
                 world.left_ctrl_adrs, _LEFT_BOUNDS)
