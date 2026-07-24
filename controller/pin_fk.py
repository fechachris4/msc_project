"""Pinocchio-backed forward kinematics for one Gen3 arm.

Loads the arm-only MJCF (sim/assets/kinova_gen3/gen3.xml), where base_link
is fixed at the origin — so oMf of the EE site is directly T_K_E. The torso
mocap body and world-frame composition are MuJoCo-side concerns
(controller/frames.py); Pinocchio never sees them.

Pinocchio's MJCF parser keeps MuJoCo's joint order, axes, and angle signs,
so q is the same 7-vector used for the MuJoCo qpos chain.

Usage:
    model, data, frame_id = build_pin_model(
        "sim/assets/kinova_gen3/gen3.xml", "base_link", "pinch_site")
    T_K_E = pin_T_K_E(model, data, frame_id, q)
"""

import numpy as np
import pinocchio as pin


def build_pin_model(mjcf_path, base_body_name, ee_site_name):
    """Load an arm MJCF into Pinocchio; return (model, data, ee_frame_id)."""
    model = pin.buildModelFromMJCF(str(mjcf_path))
    # base_body_name must be the fixed root of this MJCF (pose = identity),
    # otherwise oMf below is not expressed in the base frame.
    if not model.existFrame(base_body_name):
        raise ValueError(f"base frame {base_body_name!r} not found in {mjcf_path}")
    if not model.frames[model.getFrameId(
            base_body_name)].placement.isIdentity():
        raise ValueError(
            f"base frame {base_body_name!r} is not fixed at identity")
    if not model.existFrame(ee_site_name):
        raise ValueError(
            f"MJCF site {ee_site_name!r} did not map to a Pinocchio frame "
            f"in {mjcf_path!r}; frames: {[f.name for f in model.frames]}"
        )
    return model, pin.Data(model), model.getFrameId(ee_site_name)


def pin_T_K_E(model, data, frame_id, q):
    """EE site pose in the arm base_link frame, as a 4x4 transform."""
    pin.framesForwardKinematics(model, data, np.asarray(q, dtype=float))
    return data.oMf[frame_id].homogeneous.copy()
