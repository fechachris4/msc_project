"""Scripted EE-target motion: sinusoidal per-side target trajectory,
world frame, base held static.

controller/desired_pos.py resolves each side's torso-frame anchor pose
through the torso pose once, at t=0 (apply()), giving a static world-frame
EE-target hold. This module adds an independent research lever on top of
that hold: a sinusoidal offset added directly in world frame to the
resolved home pose, so the reactive controller's own tracking bandwidth
(error / phase lag vs. commanded frequency) can be characterized —
a companion experiment to tests/test_motion.py's base-motion-rejection
test, which characterizes disturbance rejection instead.

Per CLAUDE.md, EE references stay world-frame: the sinusoidal offset is
added on top of the pose captured once by init_home() (which itself runs
after desired_pos.apply() has resolved the static torso-frame anchor to
world) and is never re-resolved through the torso pose. That keeps this
signal an independent research lever from base motion (sim/motion.py) —
if both are scripted in the same run, one must not leak into the other.

Levers are per-side (right/left independent) and default to zero, unlike
sim/motion.py's always-on base levers: this motion must be off by default
so main.py's current static-hold behavior is unchanged unless a caller
explicitly passes nonzero amplitude. main.py never imports this module.
"""

import mujoco
import numpy as np

from controller.transforms import (
    angular_velocity_from_rpy_rates,
    rotation_from_quat,
    rotation_from_rpy,
    sine_offset,
    sine_rate,
)
from sim import targets, world

# research levers: EE-target sway amplitude and frequency, per side,
# independent. Zero by default — no motion unless a caller opts in.
EE_LINEAR_AMPLITUDE = {"right": np.zeros(3), "left": np.zeros(3)}       # m, world xyz
EE_LINEAR_FREQUENCY = {"right": 0.0, "left": 0.0}                       # Hz
EE_ROTATIONAL_AMPLITUDE = {"right": np.zeros(3), "left": np.zeros(3)}   # rad, rpy
EE_ROTATIONAL_FREQUENCY = {"right": 0.0, "left": 0.0}                   # Hz

# Populated by init_home() — not module-level constants like motion.py's
# HOME_POS, because the EE-target's world-frame home only exists after
# desired_pos.apply() has resolved the torso-frame anchor to world; it is
# not a scene default available at import time.
HOME_POS = {}
HOME_ROT = {}


def init_home():
    """Capture each side's current world-frame target pose as home.

    Must be called after desired_pos.apply() has resolved the static
    torso-frame anchors to world — that resolution is what makes this
    home meaningful. Sinusoidal motion is added on top of this captured
    pose directly in world frame afterwards; it must never be re-derived
    by re-resolving through the (possibly moving) torso pose."""
    for side in world.SIDES:
        HOME_POS[side] = targets.target_position(side)
        HOME_ROT[side] = rotation_from_quat(targets.target_quat(side))


def _resolve_levers(side, linear_amplitude, linear_frequency,
                    rotational_amplitude, rotational_frequency):
    """None kwargs -> the per-side lever dicts; explicit values pass through."""
    if linear_amplitude is None:
        linear_amplitude = EE_LINEAR_AMPLITUDE[side]
    if linear_frequency is None:
        linear_frequency = EE_LINEAR_FREQUENCY[side]
    if rotational_amplitude is None:
        rotational_amplitude = EE_ROTATIONAL_AMPLITUDE[side]
    if rotational_frequency is None:
        rotational_frequency = EE_ROTATIONAL_FREQUENCY[side]
    return linear_amplitude, linear_frequency, rotational_amplitude, \
        rotational_frequency


def target_pose_at(t, side, linear_amplitude=None, linear_frequency=None,
                   rotational_amplitude=None, rotational_frequency=None):
    """(pos, rot) of the scripted EE target at time t — pure math."""
    linear_amplitude, linear_frequency, rotational_amplitude, \
        rotational_frequency = _resolve_levers(
            side, linear_amplitude, linear_frequency,
            rotational_amplitude, rotational_frequency)

    # World frame directly on top of the captured home — NOT routed
    # through desired_pos.resolve_world() or the live torso pose. Doing
    # so would leak base motion into what must be an independent signal;
    # this is the single most important correctness point in this module.
    pos = HOME_POS[side] + sine_offset(t, linear_amplitude, linear_frequency)

    # Matrix composition, NOT motion.py's elementwise-rpy-sum shortcut:
    # that shortcut is only exact there because HOME_RPY is the zero
    # vector. The EE-target home rotation is not guaranteed identity in
    # general, so the sinusoidal rpy perturbation must be composed as a
    # body-frame rotation on top of HOME_ROT[side] via matrix multiply.
    rot = HOME_ROT[side] @ rotation_from_rpy(
        sine_offset(t, rotational_amplitude, rotational_frequency))
    return pos, rot


def target_twist_at(t, side, linear_amplitude=None, linear_frequency=None,
                    rotational_amplitude=None, rotational_frequency=None):
    """(v (3,) m/s, w (3,) rad/s) of the scripted EE target at time t,
    world frame — the analytic time derivative of target_pose_at. Not
    consumed by the control law yet (no feedforward — matches the
    current reactive-baseline phase scope); this is the analytic
    ground-truth companion, the same role motion.torso_twist_at plays
    for the base."""
    linear_amplitude, linear_frequency, rotational_amplitude, \
        rotational_frequency = _resolve_levers(
            side, linear_amplitude, linear_frequency,
            rotational_amplitude, rotational_frequency)

    v = sine_rate(t, linear_amplitude, linear_frequency)  # home is constant

    rpy_local = sine_offset(t, rotational_amplitude, rotational_frequency)
    rpy_dot_local = sine_rate(t, rotational_amplitude, rotational_frequency)
    # R(t) = R_home @ R_local(t), R_home constant => w_world = R_home @ w_local.
    w = HOME_ROT[side] @ angular_velocity_from_rpy_rates(rpy_local,
                                                          rpy_dot_local)
    return v, w


def set_target_pose(t, side, linear_amplitude=None, linear_frequency=None,
                    rotational_amplitude=None, rotational_frequency=None):
    """Write the scripted EE-target pose into the target mocap body.

    All-zero amplitudes (the default): no write at all — preserves the
    ability to hand-drag a target sphere in the viewer, and keeps default
    behavior identical to today's static hold."""
    linear_amplitude, linear_frequency, rotational_amplitude, \
        rotational_frequency = _resolve_levers(
            side, linear_amplitude, linear_frequency,
            rotational_amplitude, rotational_frequency)
    if not (np.any(linear_amplitude) or np.any(rotational_amplitude)):
        return
    pos, rot = target_pose_at(t, side, linear_amplitude, linear_frequency,
                              rotational_amplitude, rotational_frequency)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    targets.set_target(side, pos)
    targets.set_target_quat(side, quat)
