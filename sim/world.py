"""Load the scene and cache the model ids the rest of the code addresses.

model/data are module-level singletons: one simulation per process, shared
by controller, plotting, and tests alike.
"""

import mujoco

model = mujoco.MjModel.from_xml_path("sim/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)


def _named_id(objtype, name):
    """Checked mj_name2id: a typo'd name must fail here, not index -1."""
    obj_id = mujoco.mj_name2id(model, objtype, name)
    assert obj_id >= 0, f"{name!r} not in sim/scene.xml"
    return obj_id


right_ee_id = _named_id(mujoco.mjtObj.mjOBJ_SITE, "right_pinch_site")
left_ee_id = _named_id(mujoco.mjtObj.mjOBJ_SITE, "left_pinch_site")

# Body ids (index data.xpos/xmat), NOT mocap indices (data.mocap_pos);
# convert via model.body_mocapid[body_id] where a mocap index is needed.
torso_body_id = _named_id(mujoco.mjtObj.mjOBJ_BODY, "torso")
kinova_right_base_id = _named_id(mujoco.mjtObj.mjOBJ_BODY, "right_base_link")
kinova_left_base_id = _named_id(mujoco.mjtObj.mjOBJ_BODY, "left_base_link")
right_target_id = _named_id(mujoco.mjtObj.mjOBJ_BODY, "right_target")
left_target_id = _named_id(mujoco.mjtObj.mjOBJ_BODY, "left_target")

# ctrl indices of one arm's 7 position servos (ctrl index = actuator id)
right_ctrl_adrs = [
    _named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_joint_{i}")
    for i in range(1, 8)
]
left_ctrl_adrs = [
    _named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_joint_{i}")
    for i in range(1, 8)
]
