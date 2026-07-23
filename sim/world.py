"""Load the scene and cache the model ids the rest of the code addresses.

model/data are module-level singletons: one simulation per process, shared
by controller, plotting, and tests alike. Per-arm ids are dicts keyed by
side ("right" | "left") — SIDES is the canonical tuple.
"""

from pathlib import Path

import mujoco
import numpy as np

from controller.state import (
    ArmJointState,
    JointPositionCommand,
    MountCalibration,
    PlantState,
    Pose,
    Twist,
)
from controller.position_actuation import PositionActuationLimits
from controller.reactive_pose import JointCentering
from controller.servo import ArmPipelineSetup, DualArmPipelineSetup
from controller.transforms import rotation_from_quat
from runtime_config import CONFIG

SCENE_PATH = Path(__file__).resolve().with_name("scene.xml")
model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
model.opt.timestep = CONFIG.run.nominal_dt_s
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

SIDES = ("right", "left")


def _named_id(objtype, name):
    """Checked mj_name2id: a typo'd name must fail here, not index -1."""
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ValueError(f"{name!r} not in {SCENE_PATH}")
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


def _joint_addresses(field):
    addresses = {}
    for side in SIDES:
        values = []
        for index in range(1, 8):
            joint_id = _named_id(
                mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint_{index}"
            )
            values.append(int(field[joint_id]))
        addresses[side] = values
    return addresses


qpos_adrs = _joint_addresses(model.jnt_qposadr)
dof_adrs = _joint_addresses(model.jnt_dofadr)


def _mount_pose(side):
    body_id = arm_base_id[side]
    return Pose(
        model.body_pos[body_id],
        rotation_from_quat(model.body_quat[body_id]),
    )


MOUNT_CALIBRATION = MountCalibration(
    right_torso_to_base=_mount_pose("right"),
    left_torso_to_base=_mount_pose("left"),
)


def read_state(torso_twist):
    """Read one fixed-shape plant sample at the current simulation time."""
    if not isinstance(torso_twist, Twist):
        try:
            torso_twist = Twist(*torso_twist)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "torso_twist must contain finite linear and angular vectors"
            ) from error
    torso_pose = Pose(
        data.xpos[torso_body_id],
        data.xmat[torso_body_id].reshape(3, 3),
    )
    arms = {
        side: ArmJointState(
            data.qpos[list(qpos_adrs[side])],
            data.qvel[list(dof_adrs[side])],
        )
        for side in SIDES
    }
    return PlantState(
        sample_time_s=data.time,
        nominal_dt_s=model.opt.timestep,
        torso_pose_world=torso_pose,
        torso_twist_world=torso_twist,
        right=arms["right"],
        left=arms["left"],
    )


def measured_ee_pose(side):
    """MuJoCo ground-truth EE pose; never used by controller math."""
    return Pose(
        data.site_xpos[ee_site_id[side]],
        data.site_xmat[ee_site_id[side]].reshape(3, 3),
    )


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


def _actuator_ctrl_bounds(side):
    low = np.full(7, -np.inf)
    high = np.full(7, np.inf)
    for index, address in enumerate(ctrl_adrs[side]):
        if model.actuator_ctrllimited[address]:
            low[index], high[index] = model.actuator_ctrlrange[address]
    return low, high


def _arm_pipeline_setup(side):
    joint_low, joint_high, limited = jnt_range(side)
    midpoint = np.zeros(7)
    midpoint[limited] = 0.5 * (
        joint_low[limited] + joint_high[limited]
    )
    command_low, command_high = _actuator_ctrl_bounds(side)
    return ArmPipelineSetup(
        centering=JointCentering(midpoint, limited),
        actuation_limits=PositionActuationLimits(
            velocity_rad_s=np.asarray(CONFIG.limits.joint_velocity_rad_s),
            lead_rad=CONFIG.limits.position_lead_rad,
            lower_position_rad=command_low,
            upper_position_rad=command_high,
        ),
    )


PIPELINE_SETUP = DualArmPipelineSetup(
    right=_arm_pipeline_setup("right"),
    left=_arm_pipeline_setup("left"),
)


def apply_command(command):
    """Apply a complete dual-arm position command to MuJoCo ctrl."""
    if not isinstance(command, JointPositionCommand):
        raise TypeError("command must be a JointPositionCommand")
    for side in SIDES:
        data.ctrl[ctrl_adrs[side]] = command.for_arm(side)
