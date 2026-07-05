import mujoco

model = mujoco.MjModel.from_xml_path("sim/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

right_ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_pinch_site",
)

left_ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "left_pinch_site",
)

torso_mocap_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "torso",
)

kinova_right_base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "right_base_link",
)

kinova_left_base_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "left_base_link",
)

