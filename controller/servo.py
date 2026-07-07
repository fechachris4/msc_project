"""The reactive controller: pure control math plus the MuJoCo plumbing
that gathers state from the sim and produces position-servo setpoints.

Error convention: e = reference - actual, expressed in the world frame.
e_pos in meters. e_rot is the axis-angle vector (radians) of
R_err = R_ref @ R_ee.T, the world-frame rotation taking the actual
orientation to the reference. Twist ordering is linear-first, matching
the Jacobian rows: v = [vx, vy, vz (m/s); wx, wy, wz (rad/s)], world frame.

The reference is the world-frame target mocap pose (set once by
desired_pos.apply()); the actual EE pose is frames FK — no EE pose is
read from MuJoCo. The servo setpoint data.ctrl (joint angles, rad) is
the integrator state (ctrl += qdot*dt), so at the fixed point qdot = 0
=> e -> 0, and the servos' gravity droop is compensated automatically.
init_ctrl() must sync the setpoints to the current joint angles once
before the loop.
"""

import numpy as np
import pinocchio as pin

from controller import frames
from controller.transforms import rotation_from_quat
from sim import targets, world

# --- control math (pure: numpy + pinocchio, no MuJoCo) ---------------------

KP_POS = 2.0    # 1/s task-space bandwidth
KP_ROT = 2.0    # 1/s
DAMPING = 0.05  # DLS lambda


def position_error(ref_pos, ee_pos):
    """World-frame position error, meters: reference - actual."""
    return ref_pos - ee_pos


def rotation_error(ref_rot, ee_rot):
    """World-frame axis-angle error, radians: log3(R_ref @ R_ee.T)."""
    return pin.log3(ref_rot @ ee_rot.T)


def qdot_from_error(J, e_pos, e_rot, damping=DAMPING):
    """World-frame pose error -> joint rates (rad/s), two equations:

    1. control law:   v = [KP_POS*e_pos; KP_ROT*e_rot]  (commanded twist)
    2. DLS inversion: qdot = J^T (J J^T + damping^2 I)^-1 v
       — bounded qdot through singularities at the cost of a small bias."""
    v = np.concatenate([KP_POS * e_pos, KP_ROT * e_rot])
    return J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), v)


# --- pose error from sim state (target mocap vs frames FK) ------------------


def right_pose_error():
    """(e_pos, e_rot) of the right arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.right_ee_pose()
    ref_pos = targets.right_target_position()
    ref_rot = rotation_from_quat(targets.right_target_quat())
    return (position_error(ref_pos, ee_pos),
            rotation_error(ref_rot, ee_rot))


def left_pose_error():
    """(e_pos, e_rot) of the left arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.left_ee_pose()
    ref_pos = targets.left_target_position()
    ref_rot = rotation_from_quat(targets.left_target_quat())
    return (position_error(ref_pos, ee_pos),
            rotation_error(ref_rot, ee_rot))


# --- servo actuation (qdot limits, setpoint integration, data.ctrl) ---------

# Kinova Gen3 spec sheet: max joint speed, large actuators (1-4) then
# small (5-7). The real arm saturates here, so the baseline must too.
QDOT_LIMIT = np.radians([79.6, 79.6, 79.6, 79.6, 69.9, 69.9, 69.9])


def _ctrl_bounds(ctrl_adrs):
    low = np.full(len(ctrl_adrs), -np.inf)
    high = np.full(len(ctrl_adrs), np.inf)
    for i, adr in enumerate(ctrl_adrs):
        if world.model.actuator_ctrllimited[adr]:
            low[i], high[i] = world.model.actuator_ctrlrange[adr]
    return low, high


_RIGHT_BOUNDS = _ctrl_bounds(world.right_ctrl_adrs)
_LEFT_BOUNDS = _ctrl_bounds(world.left_ctrl_adrs)


def init_ctrl():
    """Sync servo setpoints to the current joint angles (once, pre-loop)."""
    world.data.ctrl[world.right_ctrl_adrs] = world.data.qpos[frames.right_qpos_adrs]
    world.data.ctrl[world.left_ctrl_adrs] = world.data.qpos[frames.left_qpos_adrs]


def apply_ctrl(dt, arms=("right", "left")):
    """Write the selected arms' updated servo setpoints into data.ctrl:
    pose error -> qdot -> clip to the joint speed limits ->
    integrate the setpoints by qdot*dt -> clip to the actuator ctrl range.

    The one loop-body block every front-end (viewer, plots, tests) must
    share — call it once per step, before mj_step. An unselected arm
    keeps its init_ctrl() setpoints and simply holds posture."""
    if "right" in arms:
        e_pos, e_rot = right_pose_error()
        qdot = qdot_from_error(frames.right_jacobian_world(), e_pos, e_rot)
        qdot = np.clip(qdot, -QDOT_LIMIT, QDOT_LIMIT)
        ctrl = world.data.ctrl[world.right_ctrl_adrs] + qdot * dt
        world.data.ctrl[world.right_ctrl_adrs] = np.clip(ctrl, *_RIGHT_BOUNDS)
    if "left" in arms:
        e_pos, e_rot = left_pose_error()
        qdot = qdot_from_error(frames.left_jacobian_world(), e_pos, e_rot)
        qdot = np.clip(qdot, -QDOT_LIMIT, QDOT_LIMIT)
        ctrl = world.data.ctrl[world.left_ctrl_adrs] + qdot * dt
        world.data.ctrl[world.left_ctrl_adrs] = np.clip(ctrl, *_LEFT_BOUNDS)
