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

import mujoco
import numpy as np
import pinocchio as pin

from controller import frames
from controller.transforms import rotation_from_quat
from sim import targets, world

# --- control math (pure: numpy + pinocchio, no MuJoCo) ---------------------

KP_POS = 2.0    # 1/s task-space bandwidth
KP_ROT = 2.0    # 1/s
KD_POS = 0.3    # dimensionless: velocity error -> velocity command
KD_ROT = 0.3    # dimensionless
K_NULL = 1.0    # 1/s null-space joint-centering
DAMPING = 0.05  # DLS lambda


def position_error(ref_pos, ee_pos):
    """World-frame position error, meters: reference - actual."""
    return ref_pos - ee_pos


def rotation_error(ref_rot, ee_rot):
    """World-frame axis-angle error, radians: log3(R_ref @ R_ee.T)."""
    return pin.log3(ref_rot @ ee_rot.T)


def velocity_error(ref_vel, ee_vel):
    """World-frame velocity error: reference - actual. Linear (m/s) and
    angular (rad/s) alike — angular velocity lives in R^3, no log map."""
    return ref_vel - ee_vel


def qdot_from_error(J, e_pos, e_rot, e_v, e_w, q, q_mid, k_null,
                    damping=DAMPING):
    """World-frame pose + velocity errors -> joint rates (rad/s):

    1. PD law:        v = [KP_POS*e_pos + KD_POS*e_v;
                           KP_ROT*e_rot + KD_ROT*e_w]  (commanded twist)
       e_v/e_w are computed velocity errors (twist_error) — never a
       numerical derivative of the position error.
    2. DLS inversion: qdot_task = J^T (J J^T + damping^2 I)^-1 v
       — bounded qdot through singularities at the cost of a small bias.
    3. null space:    qdot = qdot_task + (I - J+ J) (-k_null (q - q_mid))
       — joint centering that cannot disturb the task; k_null may be a
       per-joint vector (0 = no centering). The projector uses the exact
       pseudoinverse J+, not the damped one: the damped projector leaks
       centering into the task and left a measured ~14 mm steady-state
       error at damping=0.05 (exact projector: 0.4 mm)."""
    v = np.concatenate([KP_POS * e_pos + KD_POS * e_v,
                        KP_ROT * e_rot + KD_ROT * e_w])
    qdot_task = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), v)
    J_pinv = np.linalg.pinv(J)
    qdot_null = -k_null * (q - q_mid)
    return qdot_task + (np.eye(J.shape[1]) - J_pinv @ J) @ qdot_null


# --- pose error from sim state (target mocap vs frames FK) ------------------


def pose_error(side):
    """(e_pos, e_rot) of one arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.ee_pose(side)
    ref_pos = targets.target_position(side)
    ref_rot = rotation_from_quat(targets.target_quat(side))
    return (position_error(ref_pos, ee_pos),
            rotation_error(ref_rot, ee_rot))


def twist_error(side, base_twist):
    """(e_v, e_w): desired minus actual EE world twist, world frame.

    The desired twist comes from the target trajectory
    (targets.target_velocity — exactly zero for the static world-frame
    hold); the actual from frames.ee_velocity. base_twist is required
    with no default, same loud-failure convention as ee_velocity."""
    v_des, w_des = targets.target_velocity(side)
    v_ee, w_ee = frames.ee_velocity(side, base_twist)
    return (velocity_error(v_des, v_ee),
            velocity_error(w_des, w_ee))


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


_BOUNDS = {s: _ctrl_bounds(world.ctrl_adrs[s]) for s in world.SIDES}


def _centering(side):
    """(q_mid, k_vec) for the null-space objective: mid of jnt_range and
    gain K_NULL on the limited joints; zero gain on the continuous ones
    (no range to center in)."""
    q_mid = np.zeros(7)
    k_vec = np.zeros(7)
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{i}"
        )
        if world.model.jnt_limited[jnt_id]:
            low, high = world.model.jnt_range[jnt_id]
            q_mid[i - 1] = 0.5 * (low + high)
            k_vec[i - 1] = K_NULL
    return q_mid, k_vec


_Q_MID = {}
_K_NULL_VEC = {}
for _s in world.SIDES:
    _Q_MID[_s], _K_NULL_VEC[_s] = _centering(_s)


def init_ctrl():
    """Sync servo setpoints to the current joint angles (once, pre-loop)."""
    for side in world.SIDES:
        world.data.ctrl[world.ctrl_adrs[side]] = \
            world.data.qpos[frames.qpos_adrs[side]]


def apply_ctrl(dt, base_twist, arms=world.SIDES):
    """Write the selected arms' updated servo setpoints into data.ctrl:
    pose + twist errors -> qdot (PD) -> clip to the joint speed limits ->
    integrate the setpoints by qdot*dt -> clip to the actuator ctrl range.

    base_twist = (v_T, w_T): the torso world twist, required with no
    default (motion.torso_twist_at in sim, Vicon on hardware, zeros for
    a genuinely stationary base) — same loud-failure convention as
    frames.ee_velocity.

    The one loop-body block every front-end (viewer, plots, tests) must
    share — call it once per step, before mj_step. An unselected arm
    keeps its init_ctrl() setpoints and simply holds posture."""
    for side in arms:
        e_pos, e_rot = pose_error(side)
        e_v, e_w = twist_error(side, base_twist)
        q = world.data.qpos[frames.qpos_adrs[side]]
        qdot = qdot_from_error(frames.jacobian_world(side), e_pos, e_rot,
                               e_v, e_w, q, _Q_MID[side], _K_NULL_VEC[side])
        qdot = np.clip(qdot, -QDOT_LIMIT, QDOT_LIMIT)
        ctrl = world.data.ctrl[world.ctrl_adrs[side]] + qdot * dt
        world.data.ctrl[world.ctrl_adrs[side]] = np.clip(ctrl, *_BOUNDS[side])
