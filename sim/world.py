"""Load the scene and cache the model ids the rest of the code addresses.

model/data are module-level singletons: one simulation per process, shared
by controller, plotting, and tests alike. Per-arm ids are dicts keyed by
side ("right" | "left") — SIDES is the canonical tuple.
"""

import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path("sim/scene.xml")
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

SIDES = ("right", "left")


def _named_id(objtype, name):
    """Checked mj_name2id: a typo'd name must fail here, not index -1."""
    obj_id = mujoco.mj_name2id(model, objtype, name)
    assert obj_id >= 0, f"{name!r} not in sim/scene.xml"
    return obj_id


ee_site_id = {s: _named_id(mujoco.mjtObj.mjOBJ_SITE, f"{s}_pinch_site")
              for s in SIDES}

# Body ids (index data.xpos/xmat), NOT mocap indices (data.mocap_pos);
# convert via model.body_mocapid[body_id] where a mocap index is needed.
torso_body_id = _named_id(mujoco.mjtObj.mjOBJ_BODY, "torso")
arm_base_id = {s: _named_id(mujoco.mjtObj.mjOBJ_BODY, f"{s}_base_link")
               for s in SIDES}
target_body_id = {s: _named_id(mujoco.mjtObj.mjOBJ_BODY, f"{s}_target")
                  for s in SIDES}

# ctrl indices of one arm's 7 position servos (ctrl index = actuator id)
ctrl_adrs = {s: [_named_id(mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_joint_{i}")
                 for i in range(1, 8)]
             for s in SIDES}


def jnt_range(side):
    """(low, high, limited): per-joint bounds (rad) and a boolean mask of
    which of the 7 joints are jnt_limited — continuous joints get ±inf
    and False, same convention as servo._centering."""
    low = np.full(7, -np.inf)
    high = np.full(7, np.inf)
    limited = np.zeros(7, dtype=bool)
    for i in range(1, 8):
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                   f"{side}_joint_{i}")
        if model.jnt_limited[jnt_id]:
            low[i - 1], high[i - 1] = model.jnt_range[jnt_id]
            limited[i - 1] = True
    return low, high, limited
