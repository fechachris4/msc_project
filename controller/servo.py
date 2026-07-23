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

The file reads in control-flow order: gains, limits, and component
switches, per-arm setup computed once at import, the error stage, the
control law, the logging record, and finally apply_ctrl — the one loop
body every front-end shares, where each stage appears in sequence.

Position, orientation, and velocity feedback are independent
components: POSITION_ENABLED / ORIENTATION_ENABLED / VELOCITY_ENABLED
each gate one term of the commanded task twist, composed in one place
(task_twist_terms). Levers: gains and limits here; the world-frame
reference in sim/targets (set by desired_pos.apply()); the control law
in task_twist_terms + qdot_from_error.
"""

from dataclasses import dataclass

import mujoco
import numpy as np
import pinocchio as pin

from controller import frames
from controller.transforms import rotation_from_quat
from runtime_config import CONFIG
from sim import targets, world

# --- immutable startup configuration -----------------------------------------

# Two structural constraints on these gains (tests/test_servo.py,
# GainInvariantsTest):
# - kp_rotation_s_inv > 0: the task is a world-frame POSE hold. With it zero the
#   6-row DLS still drives the EE angular rate to ~0, but orientation
#   error has no feedback and drifts uncorrected (measured ~0.5 deg vs
#   0.02 deg controlled, 50 mm sway at 0.5 Hz).
# - kd < 1: e_v feeds back measured qdot one step delayed — a discrete
#   loop with gain ~kd that chatters at the step frequency as kd -> 1
#   (qddot 7.3 rad/s^2 at kd_position = 1.0 vs 2.3 at 0.3). The D-term also
#   lowers the effective bandwidth to kp/(1+kd) while feeding forward
#   the fraction kd/(1+kd) of the base velocity: larger kd trades
#   settle speed for disturbance rejection.
CONTROL = CONFIG.reactive_pose
LIMITS = CONFIG.limits

# Independent control components. Each flag gates one term of the
# commanded task twist (composed in task_twist_terms, the single point
# apply_ctrl and qdot_from_error share):
# - position_enabled:    linear P,  kp_position * e_pos (twist rows 0-2)
# - orientation_enabled: angular P, kp_rotation * e_rot (twist rows 3-5)
# - velocity_enabled:    D term,    kd * twist error    (all 6 rows)
# A disabled component contributes exact zeros to the commanded twist.
# NOTE the physics: the 6-row DLS still runs, so a P-disabled axis is
# commanded ZERO RATE (rate-damped, error drifts uncorrected — see the
# rotation-gain note above), not left free. Freeing an axis outright would
# mean dropping Jacobian rows: a structurally different controller.
# LIMITS contains the Gen3 joint-speed ceiling (rad/s) and anti-windup
# position lead (rad), loaded from config/control.toml. While a
# joint is physically blocked (e.g. the diagnosed torso collision) the
# integrator otherwise winds away from qpos — without bound on the
# continuous joints, which have no ctrlrange — and the kp=2000 servo
# snaps violently on release. The margin must exceed the lead of normal
# operation, (kv*qdot_max + tau_gravity)/kp ~ 0.09 rad (large
# actuators) / 0.14 rad (small); 0.2 rad clears both without throttling
# legitimate tracking.

# --- per-arm setup, computed once at import ----------------------------------


def _ctrl_bounds(ctrl_adrs):
    low = np.full(len(ctrl_adrs), -np.inf)
    high = np.full(len(ctrl_adrs), np.inf)
    for i, adr in enumerate(ctrl_adrs):
        if world.model.actuator_ctrllimited[adr]:
            low[i], high[i] = world.model.actuator_ctrlrange[adr]
    return low, high


_BOUNDS = {s: _ctrl_bounds(world.ctrl_adrs[s]) for s in world.SIDES}


def _centering(side):
    """(q_mid, limited_mask) for the null-space centering objective."""
    q_mid = np.zeros(7)
    limited_mask = np.zeros(7, dtype=bool)
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(
            world.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{i}"
        )
        if world.model.jnt_limited[jnt_id]:
            low, high = world.model.jnt_range[jnt_id]
            q_mid[i - 1] = 0.5 * (low + high)
            limited_mask[i - 1] = True
    return q_mid, limited_mask


_Q_MID = {}
_K_NULL_MASK = {}
for _s in world.SIDES:
    _Q_MID[_s], _K_NULL_MASK[_s] = _centering(_s)


def _null_gain_vector(side, control=CONTROL):
    return _K_NULL_MASK[side] * control.null_gain_s_inv


def init_ctrl():
    """Sync servo setpoints to the current joint angles (once, pre-loop)."""
    for side in world.SIDES:
        world.data.ctrl[world.ctrl_adrs[side]] = \
            world.data.qpos[frames.qpos_adrs[side]]


# --- error stage: world-frame errors vs the target ---------------------------


def pose_error(side):
    """(e_pos, e_rot) of one arm: target mocap vs FK EE pose.

    e_pos = ref - actual (m); e_rot = log3(R_ref @ R_ee.T), the
    world-frame axis-angle (rad) taking actual to reference."""
    ee_pos, ee_rot = frames.ee_pose(side)
    ref_pos = targets.target_position(side)
    ref_rot = rotation_from_quat(targets.target_quat(side))
    return ref_pos - ee_pos, pin.log3(ref_rot @ ee_rot.T)


def twist_error(side, base_twist, J=None):
    """(e_v, e_w): desired minus actual EE world twist, world frame.
    Linear (m/s) and angular (rad/s) alike — angular velocity lives in
    R^3, no log map.

    The desired twist comes from the target trajectory
    (targets.target_velocity — exactly zero for the static world-frame
    hold); the actual from frames.ee_velocity. base_twist is required
    with no default, same loud-failure convention as ee_velocity.
    J, if given, is forwarded to ee_velocity (reuse, see there)."""
    v_des, w_des = targets.target_velocity(side)
    v_ee, w_ee = frames.ee_velocity(side, base_twist, J)
    return v_des - v_ee, w_des - w_ee


# --- control law: errors -> joint rates ---------------------------------------


def task_twist_terms(e_pos, e_rot, e_v, e_w, control=CONTROL):
    """(p_twist, d_twist): the components' contributions to the
    commanded EE task twist, linear first, world frame. The single
    composition point for the position / orientation / velocity
    components — a disabled component contributes exact zeros (see the
    component flags for what that means physically)."""
    p_twist = np.concatenate([
        control.kp_position_s_inv * e_pos
        if control.position_enabled else np.zeros(3),
        control.kp_rotation_s_inv * e_rot
        if control.orientation_enabled else np.zeros(3),
    ])
    d_twist = (
        np.concatenate([
            control.kd_position * e_v,
            control.kd_rotation * e_w,
        ])
        if control.velocity_enabled else np.zeros(6)
    )
    return p_twist, d_twist


def qdot_from_error(J, e_pos, e_rot, e_v, e_w, q, q_mid, k_null,
                    damping=None, control=CONTROL):
    """World-frame pose + velocity errors -> joint rates (rad/s):

    1. PD law:        v = [KP_POS*e_pos + KD_POS*e_v;
                           KP_ROT*e_rot + KD_ROT*e_w]  (commanded twist)
       e_v/e_w are computed velocity errors (twist_error) — never a
       numerical derivative of the position error. Each term is gated
       by its component flag (task_twist_terms).
    2. DLS inversion: qdot_task = J^T (J J^T + damping^2 I)^-1 v
       — bounded qdot through singularities at the cost of a small bias.
    3. null space:    qdot = qdot_task + (I - J+ J) (-k_null (q - q_mid))
       — joint centering that cannot disturb the task; k_null may be a
       per-joint vector (0 = no centering). The projector uses the exact
       pseudoinverse J+, not the damped one: the damped projector leaks
       centering into the task and left a measured ~14 mm steady-state
       error at damping=0.05 (exact projector: 0.4 mm).

    apply_ctrl inlines this same law so the loop reads top-to-bottom;
    test_control_trace pins the two equal. This standalone form is for
    the analysis scripts (dashboard, diagnose), which re-derive qdot at
    other damping values."""
    damping = control.dls_damping if damping is None else damping
    p_twist, d_twist = task_twist_terms(
        e_pos, e_rot, e_v, e_w, control)
    task_twist = p_twist + d_twist
    qdot_task = J.T @ np.linalg.solve(
        J @ J.T + damping**2 * np.eye(6), task_twist)
    J_pinv = np.linalg.pinv(J)
    qdot_null = -k_null * (q - q_mid)
    return qdot_task + (np.eye(J.shape[1]) - J_pinv @ J) @ qdot_null


# --- per-step logging record --------------------------------------------------


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


# --- main loop body ------------------------------------------------------------


def apply_ctrl(dt, base_twist, arms=world.SIDES, control=CONTROL, limits=LIMITS):
    """Write the selected arms' updated servo setpoints into data.ctrl:
    read state -> pose + twist errors -> qdot (PD law + DLS) -> clip to
    the joint speed limits -> integrate the setpoints by qdot*dt ->
    clamp the lead over qpos to CTRL_LEAD (anti-windup) -> clip to the
    actuator ctrl range -> write data.ctrl -> record a trace.

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
        # Read the current state: joint angles and measured joint rates
        # straight from the sim, plus the world-frame EE Jacobian.
        q = world.data.qpos[frames.qpos_adrs[side]]
        qdot_measured = world.data.qvel[frames.dof_adrs[side]]
        J = frames.jacobian_world(side)

        # World-frame errors vs the target: pose (drives the P term)
        # and twist (drives the D term).
        e_pos, e_rot = pose_error(side)
        e_v, e_w = twist_error(side, base_twist, J)

        # PD law: errors -> commanded EE task twist, linear-first. Each
        # component (position, orientation, velocity) contributes only
        # if its flag is enabled — task_twist_terms is the one
        # composition point shared with qdot_from_error.
        p_twist, d_twist = task_twist_terms(
            e_pos, e_rot, e_v, e_w, control)
        task_twist = p_twist + d_twist

        # Task twist -> joint rates: DLS inversion (bounded through
        # singularities) plus null-space joint centering behind the
        # exact-pseudoinverse projector (cannot disturb the task).
        # Same law as qdot_from_error, inlined; a test pins them equal.
        qdot_task = J.T @ np.linalg.solve(
            J @ J.T + control.dls_damping**2 * np.eye(6), task_twist)
        qdot_null = -_null_gain_vector(side, control) * (q - _Q_MID[side])
        qdot_raw = qdot_task + (np.eye(7) - np.linalg.pinv(J) @ J) @ qdot_null

        # Safety limits: clip to the Gen3 joint speed limits, integrate
        # the setpoints, clamp the lead over qpos (anti-windup), clip to
        # the actuator ctrl range.
        qdot_limit = np.asarray(limits.joint_velocity_rad_s)
        qdot_speed_clipped = np.clip(qdot_raw, -qdot_limit, qdot_limit)
        speed_saturated = qdot_raw != qdot_speed_clipped
        ctrl_before = world.data.ctrl[world.ctrl_adrs[side]].copy()
        ctrl_integrated = ctrl_before + qdot_speed_clipped * dt
        ctrl_lead_limited = np.clip(
            ctrl_integrated,
            q - limits.position_lead_rad,
            q + limits.position_lead_rad,
        )
        lead_clamped = ctrl_integrated != ctrl_lead_limited
        ctrl_after = np.clip(ctrl_lead_limited, *_BOUNDS[side])
        range_clamped = ctrl_lead_limited != ctrl_after

        # Send the command: the new servo setpoints, applied at the
        # next mj_step.
        world.data.ctrl[world.ctrl_adrs[side]] = ctrl_after

        # Log every stage as an immutable snapshot.
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
