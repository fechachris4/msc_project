"""Scripted EE-target motion: sinusoidal target trajectory per side,
world frame — for characterizing the reactive loop's own tracking
bandwidth (base held static), the companion lever to sim/motion.py's
base disturbance.

Off by default (zero amplitudes = no write). Call init_home() once after
desired_pos.apply(): the sinusoid is added on top of that captured
world-frame pose and never re-resolved through the (possibly moving)
torso — keeping this signal independent of base motion.
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

_ZERO3 = np.zeros(3)  # shared default, never mutated

# Populated by init_home() — unlike motion.py's import-time HOME_POS, the
# EE-target's world-frame home only exists after desired_pos.apply().
HOME_POS = {}
HOME_ROT = {}


def init_home():
    """Capture each side's current world-frame target pose as home.
    Call once, after desired_pos.apply()."""
    for side in world.SIDES:
        HOME_POS[side] = targets.target_position(side)
        HOME_ROT[side] = rotation_from_quat(targets.target_quat(side))


def target_pose_at(t, side, linear_amplitude=_ZERO3, linear_frequency=0.0,
                   rotational_amplitude=_ZERO3, rotational_frequency=0.0):
    """(pos, rot) of the scripted EE target at time t — pure math."""
    # World frame directly on top of the captured home — never routed
    # through the live torso pose (that would leak base motion in).
    pos = HOME_POS[side] + sine_offset(t, linear_amplitude, linear_frequency)
    # Matrix composition, not motion.py's rpy-sum shortcut: that is only
    # exact for an identity home rotation, which this home need not be.
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


def set_target_pose(t, side, linear_amplitude=_ZERO3, linear_frequency=0.0,
                    rotational_amplitude=_ZERO3, rotational_frequency=0.0):
    """Write the scripted EE-target pose into the target mocap body.

    All-zero amplitudes (the default): no write at all — the target stays
    hand-draggable in the viewer, and default behavior is the static hold."""
    if not (np.any(linear_amplitude) or np.any(rotational_amplitude)):
        return
    pos, rot = target_pose_at(t, side, linear_amplitude, linear_frequency,
                              rotational_amplitude, rotational_frequency)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot.flatten())
    targets.set_target(side, pos)
    targets.set_target_quat(side, quat)
