"""Set and read the end-effector target poses.

The targets are mocap bodies in scene.xml (right_target: green sphere,
left_target: red sphere). Mocap bodies are driven directly through
data.mocap_pos (xyz, meters) and data.mocap_quat (w x y z), in world frame.
side is "right" or "left" throughout.
"""

import numpy as np

from sim import world


def _mocap_index(side):
    return world.model.body_mocapid[world.target_body_id[side]]


def set_target(side, pos):
    world.data.mocap_pos[_mocap_index(side)] = pos


def set_target_quat(side, quat):
    world.data.mocap_quat[_mocap_index(side)] = quat


def target_position(side):
    return world.data.mocap_pos[_mocap_index(side)].copy()


def target_quat(side):
    return world.data.mocap_quat[_mocap_index(side)].copy()


def target_velocity(side):
    """(v_des (3,) m/s, w_des (3,) rad/s): world-frame target twist.

    Pinned zero on purpose, even under sim/target_motion.py's scripted
    target motion: the reactive baseline gets no reference-velocity
    feedforward (phase scope), so the D-term damps against the EE's own
    tracking velocity and the loop tracks with effective bandwidth
    KP/(1+KD) — the modeling basis of analysis/bandwidth_sweep.py and
    tests/test_target_motion.py's TargetTrackingBandwidthTest. For the
    static world-frame pose hold this is also simply the true target
    twist. If feedforward enters scope later, wire this to
    target_motion.target_twist_at and recalibrate both."""
    return np.zeros(3), np.zeros(3)
