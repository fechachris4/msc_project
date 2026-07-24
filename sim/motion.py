"""Scripted base motion: sinusoidal torso disturbance, world frame.

The torso mocap body is driven as home + A*sin(2*pi*f*t) in position
(world xyz) and orientation (rpy). The EE targets were resolved to the
world frame once at startup, so this motion IS the disturbance — the
task stays a world-frame pose hold.

Mocap writes teleport the torso: mocap bodies carry no velocity state,
so the arms feel base motion as position steps. Accepted phase-1 gap.
"""

import mujoco
import numpy as np

from controller.transforms import (
    angular_velocity_from_rpy_rates,
    rotation_from_rpy,
    sine_offset,
    sine_rate,
)
from sim import world

# --- research levers ---------------------------------------------------
# Edit these SI values to change the default base-motion scenario used by
# the main viewer and general-purpose analysis views.
LINEAR_AMPLITUDE = np.array([0.0, 0.00, 0.0])
ROTATIONAL_AMPLITUDE = np.array([0.0, 0.0, 0.0])
LINEAR_FREQUENCY = 0.5
ROTATIONAL_FREQUENCY = 0.5

_TORSO_MOCAP_IDX = world.model.body_mocapid[world.torso_body_id]
HOME_POS = world.data.mocap_pos[_TORSO_MOCAP_IDX].copy()
# torso_pose_at's elementwise rpy sum is only exact because the home
# rotation is identity (cf. target_motion.py's matrix composition); a
# tilted torso home in scene.xml must fail here, not silently bend the
# scripted motion.
assert np.allclose(world.data.mocap_quat[_TORSO_MOCAP_IDX],
                   [1.0, 0.0, 0.0, 0.0]), \
    "torso home rotation is not identity; HOME_RPY = zeros is invalid"
HOME_RPY = np.zeros(3)


def torso_pose_at(t, linear_amplitude=LINEAR_AMPLITUDE,
                  linear_frequency=LINEAR_FREQUENCY,
                  rotational_amplitude=ROTATIONAL_AMPLITUDE,
                  rotational_frequency=ROTATIONAL_FREQUENCY):
    """(pos, rpy) of the scripted torso at time t — pure math."""
    pos = HOME_POS + sine_offset(t, linear_amplitude, linear_frequency)
    rpy = HOME_RPY + sine_offset(t, rotational_amplitude,
                                 rotational_frequency)
    return pos, rpy


def torso_twist_at(t, linear_amplitude=LINEAR_AMPLITUDE,
                   linear_frequency=LINEAR_FREQUENCY,
                   rotational_amplitude=ROTATIONAL_AMPLITUDE,
                   rotational_frequency=ROTATIONAL_FREQUENCY):
    """(v (3,) m/s, w (3,) rad/s) of the scripted torso at time t, world
    frame — the analytic time derivative of torso_pose_at. This is the
    base twist the mocap teleports never give the simulator; on hardware
    the same quantity comes from Vicon."""
    v = sine_rate(t, linear_amplitude, linear_frequency)
    rpy = HOME_RPY + sine_offset(t, rotational_amplitude,
                                 rotational_frequency)
    rpy_dot = sine_rate(t, rotational_amplitude, rotational_frequency)
    return v, angular_velocity_from_rpy_rates(rpy, rpy_dot)


def set_torso_pose(t, linear_amplitude=LINEAR_AMPLITUDE,
                   linear_frequency=LINEAR_FREQUENCY,
                   rotational_amplitude=ROTATIONAL_AMPLITUDE,
                   rotational_frequency=ROTATIONAL_FREQUENCY):
    """Write the scripted torso pose into the mocap body.

    All-zero amplitudes: no write at all — the static condition, and the
    torso stays free (hand-draggable in the viewer as a perturbation)."""
    if not (np.any(linear_amplitude) or np.any(rotational_amplitude)):
        return
    pos, rpy = torso_pose_at(t, linear_amplitude, linear_frequency,
                             rotational_amplitude, rotational_frequency)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rotation_from_rpy(rpy).flatten())
    world.data.mocap_pos[_TORSO_MOCAP_IDX] = pos
    world.data.mocap_quat[_TORSO_MOCAP_IDX] = quat
