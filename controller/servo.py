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

from dataclasses import dataclass

import mujoco
import numpy as np
import pinocchio as pin

from controller import frames
from controller.transforms import rotation_from_quat
from sim import targets, world

# --- control math (pure: numpy + pinocchio, no MuJoCo) ---------------------

# Two structural constraints on these gains (tests/test_servo.py,
# GainInvariantsTest):
# - KP_ROT > 0: the task is a world-frame POSE hold. With KP_ROT = 0 the
#   6-row DLS still drives the EE angular rate to ~0, but orientation
#   error has no feedback and drifts uncorrected (measured ~0.5 deg vs
#   0.02 deg controlled, 50 mm sway at 0.5 Hz).
# - KD < 1: e_v feeds back measured qdot one step delayed — a discrete
#   loop with gain ~KD that chatters at the step frequency as KD -> 1
#   (qddot 7.3 rad/s^2 at KD_POS = 1.0 vs 2.3 at 0.3). The D-term also
#   lowers the effective bandwidth to KP/(1+KD) while feeding forward
#   the fraction KD/(1+KD) of the base velocity: larger KD trades
#   settle speed for disturbance rejection.
KP_POS = 20.0    # 1/s task-space bandwidth
KP_ROT = 2.0    # 1/s
KD_POS = 0.9    # dimensionless: velocity error -> velocity command
KD_ROT = 0.3    # dimensionless
K_NULL = 1.0    # 1/s null-space joint-centering
DAMPING = 0.05  # DLS lambda


def _read_only_copy(value):
    copied = np.array(value, copy=True)
    return np.frombuffer(copied.tobytes(), dtype=copied.dtype).reshape(
        copied.shape)


@dataclass(frozen=True)
class ControlTrace:
    """Read-only snapshots of one arm's control stages for one step."""

    J: np.ndarray
    e_pos: np.ndarray
    e_rot: np.ndarray
    e_v: np.ndarray
    e_w: np.ndarray
    p_twist: np.ndarray
    d_twist: np.ndarray
    task_twist: np.ndarray
    q: np.ndarray
    qdot_measured: np.ndarray
    qdot_raw: np.ndarray
    qdot_speed_clipped: np.ndarray
    qdot_effective: np.ndarray
    ctrl_before: np.ndarray
    ctrl_after: np.ndarray
    speed_saturated: np.ndarray
    lead_clamped: np.ndarray
    range_clamped: np.ndarray

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            object.__setattr__(self, name, _read_only_copy(getattr(self, name)))


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


def task_twist_terms(e_pos, e_rot, e_v, e_w):
    """Separate P and D task-twist terms, linear first, in the world frame."""
    p_twist = np.concatenate([KP_POS * e_pos, KP_ROT * e_rot])
    d_twist = np.concatenate([KD_POS * e_v, KD_ROT * e_w])
    return p_twist, d_twist, p_twist + d_twist


def _qdot_from_task_twist(J, task_twist, q, q_mid, k_null, damping):
    qdot_task = J.T @ np.linalg.solve(
        J @ J.T + damping**2 * np.eye(6), task_twist)
    J_pinv = np.linalg.pinv(J)
    qdot_null = -k_null * (q - q_mid)
    return qdot_task + (np.eye(J.shape[1]) - J_pinv @ J) @ qdot_null


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
    _, _, task_twist = task_twist_terms(e_pos, e_rot, e_v, e_w)
    return _qdot_from_task_twist(
        J, task_twist, q, q_mid, k_null, damping)


# --- pose error from sim state (target mocap vs frames FK) ------------------


def pose_error(side):
    """(e_pos, e_rot) of one arm: target mocap vs FK EE pose."""
    ee_pos, ee_rot = frames.ee_pose(side)
    ref_pos = targets.target_position(side)
    ref_rot = rotation_from_quat(targets.target_quat(side))
    return (position_error(ref_pos, ee_pos),
            rotation_error(ref_rot, ee_rot))


def twist_error(side, base_twist, J=None):
    """(e_v, e_w): desired minus actual EE world twist, world frame.

    The desired twist comes from the target trajectory
    (targets.target_velocity — exactly zero for the static world-frame
    hold); the actual from frames.ee_velocity. base_twist is required
    with no default, same loud-failure convention as ee_velocity.
    J, if given, is forwarded to ee_velocity (reuse, see there)."""
    v_des, w_des = targets.target_velocity(side)
    v_ee, w_ee = frames.ee_velocity(side, base_twist, J)
    return (velocity_error(v_des, v_ee),
            velocity_error(w_des, w_ee))


# --- servo actuation (qdot limits, setpoint integration, data.ctrl) ---------

# Kinova Gen3 spec sheet: max joint speed, large actuators (1-4) then
# small (5-7). The real arm saturates here, so the baseline must too.
QDOT_LIMIT = np.radians([79.6, 79.6, 79.6, 79.6, 69.9, 69.9, 69.9])

# Anti-windup: max setpoint lead |ctrl - qpos| per joint (rad). While a
# joint is physically blocked (e.g. the diagnosed torso collision) the
# integrator otherwise winds away from qpos — without bound on the
# continuous joints, which have no ctrlrange — and the kp=2000 servo
# snaps violently on release. The margin must exceed the lead of normal
# operation, (kv*qdot_max + tau_gravity)/kp ~ 0.09 rad (large
# actuators) / 0.14 rad (small); 0.2 rad clears both without throttling
# legitimate tracking.
CTRL_LEAD = 0.2


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

# Fixed at import time from the original _centering() result: which
# entries are the limited joints (nonzero gain) vs. the continuous ones
# (always 0). set_k_null must consult this, not _K_NULL_VEC's *current*
# nonzero pattern -- rescaling to 0 would otherwise erase the mask and
# strand every later nonzero value at 0 too.
_K_NULL_MASK = {s: _K_NULL_VEC[s] != 0.0 for s in world.SIDES}


def set_k_null(value):
    """Update K_NULL and rescale the nonzero (limited-joint) entries of
    _K_NULL_VEC to match; continuous joints stay at 0. The gain panel
    calls this instead of touching _K_NULL_VEC directly."""
    global K_NULL
    K_NULL = value
    for side in world.SIDES:
        _K_NULL_VEC[side][_K_NULL_MASK[side]] = value


def init_ctrl():
    """Sync servo setpoints to the current joint angles (once, pre-loop)."""
    for side in world.SIDES:
        world.data.ctrl[world.ctrl_adrs[side]] = \
            world.data.qpos[frames.qpos_adrs[side]]


def apply_ctrl(dt, base_twist, arms=world.SIDES):
    """Write the selected arms' updated servo setpoints into data.ctrl:
    pose + twist errors -> qdot (PD) -> clip to the joint speed limits ->
    integrate the setpoints by qdot*dt -> clamp the lead over qpos to
    CTRL_LEAD (anti-windup) -> clip to the actuator ctrl range.

    base_twist = (v_T, w_T): the torso world twist, required with no
    default (motion.torso_twist_at in sim, Vicon on hardware, zeros for
    a genuinely stationary base) — same loud-failure convention as
    frames.ee_velocity.

    The one loop-body block every front-end (viewer, plots, tests) must
    share — call it once per step, before mj_step. An unselected arm
    keeps its init_ctrl() setpoints and simply holds posture.

    Returns one immutable ControlTrace per selected arm."""
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and greater than zero")

    traces = {}
    for side in arms:
        J = frames.jacobian_world(side)
        e_pos, e_rot = pose_error(side)
        e_v, e_w = twist_error(side, base_twist, J)
        q = world.data.qpos[frames.qpos_adrs[side]]
        p_twist, d_twist, task_twist = task_twist_terms(
            e_pos, e_rot, e_v, e_w)
        qdot_measured = world.data.qvel[frames.dof_adrs[side]]
        qdot_raw = _qdot_from_task_twist(
            J, task_twist, q, _Q_MID[side], _K_NULL_VEC[side], DAMPING)
        qdot_speed_clipped = np.clip(qdot_raw, -QDOT_LIMIT, QDOT_LIMIT)
        speed_saturated = qdot_raw != qdot_speed_clipped
        ctrl_before = world.data.ctrl[world.ctrl_adrs[side]].copy()
        ctrl_integrated = ctrl_before + qdot_speed_clipped * dt
        ctrl_lead_limited = np.clip(
            ctrl_integrated, q - CTRL_LEAD, q + CTRL_LEAD)
        lead_clamped = ctrl_integrated != ctrl_lead_limited
        ctrl_after = np.clip(ctrl_lead_limited, *_BOUNDS[side])
        range_clamped = ctrl_lead_limited != ctrl_after
        world.data.ctrl[world.ctrl_adrs[side]] = ctrl_after
        qdot_effective = (ctrl_after - ctrl_before) / dt
        traces[side] = ControlTrace(
            J=J,
            e_pos=e_pos,
            e_rot=e_rot,
            e_v=e_v,
            e_w=e_w,
            p_twist=p_twist,
            d_twist=d_twist,
            task_twist=task_twist,
            q=q,
            qdot_measured=qdot_measured,
            qdot_raw=qdot_raw,
            qdot_speed_clipped=qdot_speed_clipped,
            qdot_effective=qdot_effective,
            ctrl_before=ctrl_before,
            ctrl_after=ctrl_after,
            speed_saturated=speed_saturated,
            lead_clamped=lead_clamped,
            range_clamped=range_clamped,
        )
    return traces
