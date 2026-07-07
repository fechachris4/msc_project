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

from controller.transforms import rotation_from_rpy
from sim import world

# research levers: base sway amplitude and frequency
LINEAR_AMPLITUDE = np.array([0.2, 0.0, 0.0])     # m, world xyz
ROTATIONAL_AMPLITUDE = np.array([0.0, 0.0, 0.0])  # rad, rpy
LINEAR_FREQUENCY = 0.5      # Hz
ROTATIONAL_FREQUENCY = 0.0  # Hz

_TORSO_MOCAP_IDX = world.model.body_mocapid[world.torso_body_id]
HOME_POS = world.data.mocap_pos[_TORSO_MOCAP_IDX].copy()
HOME_RPY = np.zeros(3)  # scene starts the torso at identity rotation


def sine_offset(t, amplitude, frequency):
    return amplitude * np.sin(2.0 * np.pi * frequency * t)


def torso_pose_at(t, linear_amplitude=LINEAR_AMPLITUDE,
                  linear_frequency=LINEAR_FREQUENCY,
                  rotational_amplitude=ROTATIONAL_AMPLITUDE,
                  rotational_frequency=ROTATIONAL_FREQUENCY):
    """(pos, rpy) of the scripted torso at time t — pure math."""
    pos = HOME_POS + sine_offset(t, linear_amplitude, linear_frequency)
    rpy = HOME_RPY + sine_offset(t, rotational_amplitude,
                                 rotational_frequency)
    return pos, rpy


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
