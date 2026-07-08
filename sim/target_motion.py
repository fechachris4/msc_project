"""Scripted EE-target motion: sinusoidal target trajectory per side,
world frame — for characterizing the reactive loop's own tracking
bandwidth (base held static), the companion lever to sim/motion.py's
base disturbance.

Off by default (zero amplitudes = no write). Usage (full example:
analysis/bandwidth_sweep.py):

    desired_pos.apply()        # resolve static anchors to world, once
    target_motion.init_home()  # capture them as the sinusoid anchors
    # then, once per sim step for each side being driven:
    target_motion.set_target_pose(t, side, linear_amplitude=..., ...)

The sinusoid is added on top of the home pose captured by init_home()
and never re-resolved through the (possibly moving) torso — keeping
this signal independent of base motion.

side is "right" or "left" (world.SIDES). t is seconds since THIS motion
started — not necessarily world.data.time. The sine is zero at t=0, so
starting the caller's clock at motion start guarantees the target
begins exactly at home (no teleport when the motion switches on).

Layout: research levers -> home anchors (sim read, once) -> trajectory
(pure math) -> sim interaction (mocap write).
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

# --- research levers ---------------------------------------------------
# Deliberately NO module-level lever constants (contrast sim/motion.py,
# whose always-on base levers live at the top of the file): this motion
# is off by default, and the levers ARE the per-call amplitude/frequency
# kwargs on the three trajectory functions below. Each experiment owns
# its lever values at its own call site (see analysis/bandwidth_sweep.py
# for the pattern). To add a new lever: add a kwarg to target_pose_at,
# target_twist_at, AND set_target_pose, and thread it through — no
# module state.

_ZERO3 = np.zeros(3)  # shared kwarg default, never mutated


# --- home anchors (read from sim, once) ---------------------------------

# Per-side world-frame anchor the sinusoid swings about. Populated by
# init_home() — unlike motion.py's import-time HOME_POS, the EE-target's
# world-frame home only exists after desired_pos.apply(). A
# KeyError("right"/"left") from these dicts means init_home() was never
# called.
HOME_POS = {}
HOME_ROT = {}


def init_home():
    """Capture each side's current world-frame target pose as home.
    Call once, after desired_pos.apply()."""
    for side in world.SIDES:
        HOME_POS[side] = targets.target_position(side)
        HOME_ROT[side] = rotation_from_quat(targets.target_quat(side))


# --- trajectory (pure math: pose and its analytic twist) ----------------
# EXTENSION POINT — trajectory shape. To add a new shape (circle,
# figure-eight, square wave): write its offset(t)/rate(t) pair in
# controller/transforms.py as analytic derivatives of each other (like
# sine_offset/sine_rate), then swap the pair into BOTH target_pose_at
# and target_twist_at together. target_twist_at is the ground-truth
# companion used for velocity validation — an offset/rate pair that is
# not an exact derivative breaks it silently (no test fails at the
# pose level).


def target_pose_at(t, side, linear_amplitude=_ZERO3, linear_frequency=0.0,
                   rotational_amplitude=_ZERO3, rotational_frequency=0.0):
    """(pos (3,) m, rot 3x3) of the scripted EE target at time t, world
    frame — pure math, no sim reads or writes."""
    # World frame directly on top of the captured home — never routed
    # through the live torso pose (that would leak base motion in).
    pos = HOME_POS[side] + sine_offset(t, linear_amplitude, linear_frequency)
    # The rotation swings by a sinusoidal rpy perturbation about the home
    # orientation. Matrix composition, not motion.py's rpy-sum shortcut:
    # that is only exact for an identity home rotation (Euler angles do
    # not add on top of a nonzero home), which this home need not be.
    rot = HOME_ROT[side] @ rotation_from_rpy(
        sine_offset(t, rotational_amplitude, rotational_frequency))
    return pos, rot


def target_twist_at(t, side, linear_amplitude=_ZERO3, linear_frequency=0.0,
                    rotational_amplitude=_ZERO3, rotational_frequency=0.0):
    """(v (3,) m/s, w (3,) rad/s) of the scripted EE target at time t,
    world frame — the analytic time derivative of target_pose_at. Not
    consumed by the control law (no feedforward yet); it is the
    ground-truth companion, like motion.torso_twist_at for the base."""
    v = sine_rate(t, linear_amplitude, linear_frequency)  # home is constant
    rpy = sine_offset(t, rotational_amplitude, rotational_frequency)
    rpy_dot = sine_rate(t, rotational_amplitude, rotational_frequency)
    # R(t) = R_home @ R_local(t), R_home constant => w = R_home @ w_local.
    w = HOME_ROT[side] @ angular_velocity_from_rpy_rates(rpy, rpy_dot)
    return v, w


# --- sim interaction (mocap write) ---------------------------------------


def set_target_pose(t, side, linear_amplitude=_ZERO3, linear_frequency=0.0,
                    rotational_amplitude=_ZERO3, rotational_frequency=0.0):
    """Write the scripted EE-target pose into the target mocap body —
    the apply-to-sim step. The arm then follows because the closed loop
    (servo.apply_ctrl) chases the target every step; there is no
    separate "move the arm" command in this codebase.

    All-zero amplitudes (the default): no write at all — the target stays
    hand-draggable in the viewer, and default behavior is the static hold."""
    if not (np.any(linear_amplitude) or np.any(rotational_amplitude)):
        return
    pos, rot = target_pose_at(t, side, linear_amplitude, linear_frequency,
                              rotational_amplitude, rotational_frequency)
    quat = np.zeros(4)  # out-parameter: MuJoCo quaternion [w, x, y, z]
    mujoco.mju_mat2Quat(quat, rot.flatten())
    targets.set_target(side, pos)
    targets.set_target_quat(side, quat)
