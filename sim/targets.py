"""Set and read the end-effector target positions.

The targets are mocap bodies in scene.xml (right_target: green sphere,
left_target: red sphere). Mocap bodies are driven directly through
data.mocap_pos, in world frame, meters.
"""

from sim import world


def _mocap_index(body_id):
    return world.model.body_mocapid[body_id]


def set_right_target(pos):
    world.data.mocap_pos[_mocap_index(world.right_target_id)] = pos


def set_left_target(pos):
    world.data.mocap_pos[_mocap_index(world.left_target_id)] = pos


def reset_to_ee():
    """Place each target at its arm's current end-effector position.

    Call after mj_forward/mj_step so site positions are up to date.
    """
    set_right_target(world.data.site_xpos[world.right_ee_id])
    set_left_target(world.data.site_xpos[world.left_ee_id])


def right_target_position():
    return world.data.mocap_pos[_mocap_index(world.right_target_id)].copy()


def left_target_position():
    return world.data.mocap_pos[_mocap_index(world.left_target_id)].copy()
