"""MuJoCo plant backend plus the repository's single default instance.

The controller and Runner use only takeover/exchange/release. Public model
handles remain available here for viewer setup, ground-truth comparisons, and
diagnostic experiments; they are not part of the cross-backend contract.
"""

from pathlib import Path

import mujoco
import numpy as np

from controller.position_actuation import PositionActuationLimits
from controller.reactive_controller import JointLimitAvoidance
from controller.servo import ArmPipelineSetup, DualArmPipelineSetup
from controller.state import (
    ArmJointState,
    JointPositionCommand,
    MountCalibration,
    PlantState,
    Pose,
    Twist,
)
from controller.transforms import rotation_from_quat, rotation_from_rpy
from runtime_config import CONFIG


SCENE_PATH = Path(__file__).resolve().with_name("scene.xml")
SIDES = ("right", "left")


class MujocoBackend:
    """Own MuJoCo state, simulation stepping, and lifecycle."""

    def __init__(self, scene_path=SCENE_PATH, config=CONFIG):
        self.scene_path = Path(scene_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_path))
        self.model.opt.timestep = config.run.nominal_dt_s
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.ee_site_id = {
            side: self._named_id(
                mujoco.mjtObj.mjOBJ_SITE, f"{side}_pinch_site")
            for side in SIDES
        }
        self.torso_body_id = self._named_id(
            mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.arm_base_id = {
            side: self._named_id(
                mujoco.mjtObj.mjOBJ_BODY, f"{side}_base_link")
            for side in SIDES
        }
        self.target_body_id = {
            side: self._named_id(
                mujoco.mjtObj.mjOBJ_BODY, f"{side}_target")
            for side in SIDES
        }
        self.ctrl_adrs = {
            side: [
                self._named_id(
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    f"{side}_joint_{index}",
                )
                for index in range(1, 8)
            ]
            for side in SIDES
        }
        self.qpos_adrs = self._joint_addresses(self.model.jnt_qposadr)
        self.dof_adrs = self._joint_addresses(self.model.jnt_dofadr)
        self.mount_calibration = MountCalibration(
            right_torso_to_base=self._mount_pose("right"),
            left_torso_to_base=self._mount_pose("left"),
        )
        self.pipeline_setup = DualArmPipelineSetup(
            right=self._arm_pipeline_setup("right", config),
            left=self._arm_pipeline_setup("left", config),
        )

        self._active = False
        self._torso_pose_at = None
        self._torso_twist_at = None
        self._torso_mocap_index = int(
            self.model.body_mocapid[self.torso_body_id])

    def _named_id(self, object_type, name):
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"{name!r} not in {self.scene_path}")
        return object_id

    def _joint_addresses(self, field):
        addresses = {}
        for side in SIDES:
            values = []
            for index in range(1, 8):
                joint_id = self._named_id(
                    mujoco.mjtObj.mjOBJ_JOINT,
                    f"{side}_joint_{index}",
                )
                values.append(int(field[joint_id]))
            addresses[side] = values
        return addresses

    def _mount_pose(self, side):
        body_id = self.arm_base_id[side]
        return Pose(
            self.model.body_pos[body_id],
            rotation_from_quat(self.model.body_quat[body_id]),
        )

    def _actuator_ctrl_bounds(self, side):
        low = np.full(7, -np.inf)
        high = np.full(7, np.inf)
        for index, address in enumerate(self.ctrl_adrs[side]):
            if self.model.actuator_ctrllimited[address]:
                low[index], high[index] = (
                    self.model.actuator_ctrlrange[address]
                )
        return low, high

    def _arm_pipeline_setup(self, side, config):
        joint_low, joint_high, limited = self.jnt_range(side)
        limit = np.zeros(7)
        limit[limited] = np.maximum(
            np.abs(joint_low[limited]), np.abs(joint_high[limited])
        )
        command_low, command_high = self._actuator_ctrl_bounds(side)
        return ArmPipelineSetup(
            avoidance=JointLimitAvoidance(
                limit, config.reactive_pose.limit_avoid_zone_rad
            ),
            actuation_limits=PositionActuationLimits(
                velocity_rad_s=np.asarray(
                    config.limits.joint_velocity_rad_s),
                lead_rad=config.limits.position_lead_rad,
                lower_position_rad=command_low,
                upper_position_rad=command_high,
            ),
        )

    def configure_torso_driver(self, pose_at, twist_at):
        """Install pure scripted pose/twist functions before takeover."""
        if self._active:
            raise RuntimeError(
                "cannot change torso driver while backend is active")
        if (pose_at is None) != (twist_at is None):
            raise ValueError("pose_at and twist_at must both be set or None")
        if pose_at is not None and (
            not callable(pose_at) or not callable(twist_at)
        ):
            raise TypeError("pose_at and twist_at must be callable")
        self._torso_pose_at = pose_at
        self._torso_twist_at = twist_at

    def reset(self):
        """Reset MuJoCo storage; control state is rebuilt by a new Runner."""
        if self._active:
            raise RuntimeError("release the backend before reset")
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def _refresh_torso(self):
        if self._torso_pose_at is None:
            mujoco.mj_kinematics(self.model, self.data)
            return Twist.zero()
        position, rpy = self._torso_pose_at(float(self.data.time))
        rotation = rotation_from_rpy(rpy)
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, rotation.flatten())
        self.data.mocap_pos[self._torso_mocap_index] = position
        self.data.mocap_quat[self._torso_mocap_index] = quaternion
        mujoco.mj_kinematics(self.model, self.data)
        return Twist(*self._torso_twist_at(float(self.data.time)))

    def takeover(self):
        if self._active:
            raise RuntimeError("MuJoCo backend is already active")
        self._active = True
        try:
            torso_twist = self._refresh_torso()
            return self.read_state(torso_twist)
        except Exception:
            self._active = False
            raise

    def exchange(self, command):
        if not self._active:
            raise RuntimeError("takeover must precede exchange")
        self.apply_command(command)
        mujoco.mj_step(self.model, self.data)
        torso_twist = self._refresh_torso()
        return self.read_state(torso_twist)

    def release(self):
        self._active = False

    def read_state(self, torso_twist):
        """Read one fixed-shape plant sample at the current sim time."""
        if not isinstance(torso_twist, Twist):
            try:
                torso_twist = Twist(*torso_twist)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "torso_twist must contain finite linear and angular "
                    "vectors"
                ) from error
        torso_pose = Pose(
            self.data.xpos[self.torso_body_id],
            self.data.xmat[self.torso_body_id].reshape(3, 3),
        )
        arms = {
            side: ArmJointState(
                self.data.qpos[list(self.qpos_adrs[side])],
                self.data.qvel[list(self.dof_adrs[side])],
            )
            for side in SIDES
        }
        return PlantState(
            sample_time_s=self.data.time,
            nominal_dt_s=self.model.opt.timestep,
            torso_pose_world=torso_pose,
            torso_twist_world=torso_twist,
            right=arms["right"],
            left=arms["left"],
        )

    def apply_command(self, command):
        if not isinstance(command, JointPositionCommand):
            raise TypeError("command must be a JointPositionCommand")
        for side in SIDES:
            self.data.ctrl[self.ctrl_adrs[side]] = command.for_arm(side)

    def measured_ee_pose(self, side):
        """MuJoCo ground truth; never consumed by controller math."""
        return Pose(
            self.data.site_xpos[self.ee_site_id[side]],
            self.data.site_xmat[self.ee_site_id[side]].reshape(3, 3),
        )

    def jnt_range(self, side):
        low = np.full(7, -np.inf)
        high = np.full(7, np.inf)
        limited = np.zeros(7, dtype=bool)
        for index in range(1, 8):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                f"{side}_joint_{index}",
            )
            if self.model.jnt_limited[joint_id]:
                low[index - 1], high[index - 1] = (
                    self.model.jnt_range[joint_id]
                )
                limited[index - 1] = True
        return low, high, limited


# One explicit default simulation instance. Existing diagnostics inspect its
# MuJoCo handles directly; production control uses ``backend`` through Runner.
backend = MujocoBackend()

model = backend.model
data = backend.data
ee_site_id = backend.ee_site_id
torso_body_id = backend.torso_body_id
arm_base_id = backend.arm_base_id
target_body_id = backend.target_body_id
ctrl_adrs = backend.ctrl_adrs
qpos_adrs = backend.qpos_adrs
dof_adrs = backend.dof_adrs
MOUNT_CALIBRATION = backend.mount_calibration
PIPELINE_SETUP = backend.pipeline_setup

read_state = backend.read_state
apply_command = backend.apply_command
measured_ee_pose = backend.measured_ee_pose
jnt_range = backend.jnt_range
