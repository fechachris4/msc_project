"""Set and read the end-effector target poses.

The targets are mocap bodies in scene.xml (right_target: green sphere,
left_target: red sphere). Mocap bodies are driven directly through
data.mocap_pos (xyz, meters) and data.mocap_quat (w x y z), in world frame.
"""

from sim import world


def _mocap_index(body_id):
    return world.model.body_mocapid[body_id]


def set_right_target(pos):
    world.data.mocap_pos[_mocap_index(world.right_target_id)] = pos


def set_left_target(pos):
    world.data.mocap_pos[_mocap_index(world.left_target_id)] = pos


def set_right_target_quat(quat):
    world.data.mocap_quat[_mocap_index(world.right_target_id)] = quat


def set_left_target_quat(quat):
    world.data.mocap_quat[_mocap_index(world.left_target_id)] = quat


def right_target_position():
    return world.data.mocap_pos[_mocap_index(world.right_target_id)].copy()


def left_target_position():
    return world.data.mocap_pos[_mocap_index(world.left_target_id)].copy()


def right_target_quat():
    return world.data.mocap_quat[_mocap_index(world.right_target_id)].copy()


def left_target_quat():
    return world.data.mocap_quat[_mocap_index(world.left_target_id)].copy()
