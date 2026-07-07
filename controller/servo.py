"""MuJoCo plumbing for the reactive controller: gather state from the sim,
call the pure math in pd, and produce position-servo setpoints.

The reference is the world-frame target mocap pose (set once by
desired_pos.apply()); the actual EE pose is frames FK — no EE pose is
read from MuJoCo. The servo setpoint data.ctrl is the integrator state
(ctrl += qdot*dt), so at the fixed point qdot = 0 => e -> 0, and the
servos' gravity droop is compensated automatically. init_ctrl() must
sync the setpoints to the current joint angles once before the loop.
"""

import numpy as np

from controller import frames, pd
from controller.transforms import rotation_from_quat
from sim import targets, world


def right_pose_error():
    """(e_pos, e_rot) of the right arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.right_ee_pose()
    ref_pos = targets.right_target_position()
    ref_rot = rotation_from_quat(targets.right_target_quat())
    return (pd.position_error(ref_pos, ee_pos),
            pd.rotation_error(ref_rot, ee_rot))


def left_pose_error():
    """(e_pos, e_rot) of the left arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.left_ee_pose()
    ref_pos = targets.left_target_position()
    ref_rot = rotation_from_quat(targets.left_target_quat())
    return (pd.position_error(ref_pos, ee_pos),
            pd.rotation_error(ref_rot, ee_rot))


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


def _ctrl(dt, pose_error, jacobian_world, ctrl_adrs, bounds):
    e_pos, e_rot = pose_error()
    qdot = pd.qdot_from_error(jacobian_world(), e_pos, e_rot)
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


_ARMS = {
    "right": (right_ctrl, world.right_ctrl_adrs),
    "left": (left_ctrl, world.left_ctrl_adrs),
}


def apply_ctrl(dt, arms=("right", "left")):
    """Write the selected arms' updated servo setpoints into data.ctrl.

    The one loop-body block every front-end (viewer, plots, tests) must
    share — call it once per step, before mj_step. An unselected arm
    keeps its init_ctrl() setpoints and simply holds posture."""
    for name in arms:
        ctrl_fn, ctrl_adrs = _ARMS[name]
        world.data.ctrl[ctrl_adrs] = ctrl_fn(dt)
