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

    The targets are resolved once by desired_pos.apply() and then never
    move (world-frame pose hold), so the reference twist is exactly
    zero — the derivative of the actual trajectory, not a placeholder.
    A future moving-target generator must update this together with
    set_target/set_target_quat, the way motion.torso_twist_at pairs
    with motion.set_torso_pose."""
    return np.zeros(3), np.zeros(3)
