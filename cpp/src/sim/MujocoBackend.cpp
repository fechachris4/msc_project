#include "sim/MujocoBackend.h"

#include <array>
#include <limits>
#include <stdexcept>

#include <mujoco/mujoco.h>

#include "math/Transforms.h"

namespace srl::sim {
namespace {

// MuJoCo stores 3x3 matrices row-major; Eigen defaults to column-major.
using RowMajor3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;

Eigen::Matrix3d RotationFromMujocoMat(const mjtNum* values) {
  return Eigen::Matrix3d(Eigen::Map<const RowMajor3>(values));
}

}  // namespace

MujocoBackend::MujocoBackend(const std::string& scene_path,
                             const config::ProjectConfig& config) {
  std::array<char, 1024> error{};
  model_ = mj_loadXML(scene_path.c_str(), nullptr, error.data(),
                      static_cast<int>(error.size()));
  if (model_ == nullptr) {
    throw std::runtime_error("cannot load " + scene_path + ": " +
                             std::string(error.data()));
  }
  // The TOML timestep overrides whatever the MJCF declares (clause I5).
  model_->opt.timestep = config.run.nominal_dt_s;
  data_ = mj_makeData(model_);
  mj_forward(model_, data_);

  for (Side side : kSides) {
    const std::string name(SideName(side));
    ee_site_id_.for_arm(side) = NamedId(mjOBJ_SITE, name + "_pinch_site");
    arm_base_id_.for_arm(side) = NamedId(mjOBJ_BODY, name + "_base_link");
    target_body_id_.for_arm(side) = NamedId(mjOBJ_BODY, name + "_target");
    for (int index = 0; index < kJoints; ++index) {
      const std::string joint = name + "_joint_" + std::to_string(index + 1);
      ctrl_adrs_.for_arm(side)[index] = NamedId(mjOBJ_ACTUATOR, joint);
      const int joint_id = NamedId(mjOBJ_JOINT, joint);
      joint_id_.for_arm(side)[index] = joint_id;
      qpos_adrs_.for_arm(side)[index] = model_->jnt_qposadr[joint_id];
      dof_adrs_.for_arm(side)[index] = model_->jnt_dofadr[joint_id];
    }
  }
  torso_body_id_ = NamedId(mjOBJ_BODY, "torso");
  torso_mocap_index_ = model_->body_mocapid[torso_body_id_];

  for (Side side : kSides) {
    const int body_id = arm_base_id_.for_arm(side);
    Pose mount;
    mount.position_m = Eigen::Vector3d(model_->body_pos[3 * body_id + 0],
                                       model_->body_pos[3 * body_id + 1],
                                       model_->body_pos[3 * body_id + 2]);
    mount.rotation = transforms::RotationFromQuat(Eigen::Vector4d(
        model_->body_quat[4 * body_id + 0], model_->body_quat[4 * body_id + 1],
        model_->body_quat[4 * body_id + 2], model_->body_quat[4 * body_id + 3]));
    mount_calibration_.for_arm(side) = mount;
    pipeline_setup_.for_arm(side) = BuildArmPipelineSetup(side, config);
  }
}

MujocoBackend::~MujocoBackend() {
  if (data_ != nullptr) mj_deleteData(data_);
  if (model_ != nullptr) mj_deleteModel(model_);
}

int MujocoBackend::NamedId(int object_type, const std::string& name) const {
  const int object_id =
      mj_name2id(model_, static_cast<mjtObj>(object_type), name.c_str());
  if (object_id < 0) throw std::runtime_error("'" + name + "' not in scene");
  return object_id;
}

JointRange MujocoBackend::JointRangeOf(Side side) const {
  JointRange range;
  for (int index = 0; index < kJoints; ++index) {
    const int joint_id = joint_id_.for_arm(side)[index];
    if (model_->jnt_limited[joint_id]) {
      range.lower(index) = model_->jnt_range[2 * joint_id + 0];
      range.upper(index) = model_->jnt_range[2 * joint_id + 1];
      range.limited[index] = true;
    } else {
      range.lower(index) = -std::numeric_limits<double>::infinity();
      range.upper(index) = std::numeric_limits<double>::infinity();
      range.limited[index] = false;
    }
  }
  return range;
}

control::ArmPipelineSetup MujocoBackend::BuildArmPipelineSetup(
    Side side, const config::ProjectConfig& config) const {
  const JointRange range = JointRangeOf(side);

  control::ArmPipelineSetup setup;
  setup.centering.midpoint_rad.setZero();
  for (int index = 0; index < kJoints; ++index) {
    setup.centering.enabled[index] = range.limited[index];
    if (range.limited[index]) {
      setup.centering.midpoint_rad(index) =
          0.5 * (range.lower(index) + range.upper(index));
    }
  }

  setup.actuation_limits.velocity_rad_s = config.limits.joint_velocity_rad_s;
  setup.actuation_limits.lead_rad = config.limits.position_lead_rad;
  for (int index = 0; index < kJoints; ++index) {
    const int actuator = ctrl_adrs_.for_arm(side)[index];
    if (model_->actuator_ctrllimited[actuator]) {
      setup.actuation_limits.lower_position_rad(index) =
          model_->actuator_ctrlrange[2 * actuator + 0];
      setup.actuation_limits.upper_position_rad(index) =
          model_->actuator_ctrlrange[2 * actuator + 1];
    } else {
      setup.actuation_limits.lower_position_rad(index) =
          -std::numeric_limits<double>::infinity();
      setup.actuation_limits.upper_position_rad(index) =
          std::numeric_limits<double>::infinity();
    }
  }
  return setup;
}

void MujocoBackend::ConfigureTorsoDriver(TorsoPoseFn pose_at,
                                         TorsoTwistFn twist_at) {
  if (active_) {
    throw std::runtime_error("cannot change torso driver while backend is active");
  }
  if (static_cast<bool>(pose_at) != static_cast<bool>(twist_at)) {
    throw std::invalid_argument("pose_at and twist_at must both be set or None");
  }
  torso_pose_at_ = std::move(pose_at);
  torso_twist_at_ = std::move(twist_at);
}

Twist MujocoBackend::RefreshTorso() {
  if (!torso_pose_at_) {
    mj_kinematics(model_, data_);
    return Twist::Zero();
  }
  const auto [position, rpy] = torso_pose_at_(data_->time);
  const Eigen::Matrix3d rotation = transforms::RotationFromRpy(rpy);

  // mju_mat2Quat wants a row-major 3x3.
  std::array<mjtNum, 9> matrix{};
  Eigen::Map<RowMajor3>(matrix.data()) = rotation;
  std::array<mjtNum, 4> quaternion{};
  mju_mat2Quat(quaternion.data(), matrix.data());

  for (int axis = 0; axis < 3; ++axis) {
    data_->mocap_pos[3 * torso_mocap_index_ + axis] = position(axis);
  }
  for (int index = 0; index < 4; ++index) {
    data_->mocap_quat[4 * torso_mocap_index_ + index] = quaternion[index];
  }
  mj_kinematics(model_, data_);

  const auto [linear, angular] = torso_twist_at_(data_->time);
  Twist twist;
  twist.linear_m_s = linear;
  twist.angular_rad_s = angular;
  return twist;
}

PlantState MujocoBackend::Takeover() {
  if (active_) throw std::runtime_error("MuJoCo backend is already active");
  active_ = true;
  try {
    const Twist torso_twist = RefreshTorso();
    return ReadState(torso_twist);
  } catch (...) {
    active_ = false;
    throw;
  }
}

PlantState MujocoBackend::Exchange(const JointPositionCommand& command) {
  if (!active_) throw std::runtime_error("takeover must precede exchange");
  ApplyCommand(command);
  mj_step(model_, data_);
  const Twist torso_twist = RefreshTorso();
  return ReadState(torso_twist);
}

void MujocoBackend::Release() { active_ = false; }

void MujocoBackend::Reset() {
  if (active_) throw std::runtime_error("release the backend before reset");
  mj_resetData(model_, data_);
  mj_forward(model_, data_);
}

PlantState MujocoBackend::ReadState(const Twist& torso_twist) const {
  PlantState state;
  state.sample_time_s = data_->time;
  state.nominal_dt_s = model_->opt.timestep;
  state.torso_pose_world.position_m =
      Eigen::Vector3d(data_->xpos[3 * torso_body_id_ + 0],
                      data_->xpos[3 * torso_body_id_ + 1],
                      data_->xpos[3 * torso_body_id_ + 2]);
  state.torso_pose_world.rotation =
      RotationFromMujocoMat(&data_->xmat[9 * torso_body_id_]);
  state.torso_twist_world = torso_twist;
  for (Side side : kSides) {
    ArmJointState& arm = state.arms.for_arm(side);
    for (int index = 0; index < kJoints; ++index) {
      arm.position_rad(index) = data_->qpos[qpos_adrs_.for_arm(side)[index]];
      arm.velocity_rad_s(index) = data_->qvel[dof_adrs_.for_arm(side)[index]];
    }
  }
  return state;
}

void MujocoBackend::ApplyCommand(const JointPositionCommand& command) {
  for (Side side : kSides) {
    const Vector7& position = command.for_arm(side);
    for (int index = 0; index < kJoints; ++index) {
      data_->ctrl[ctrl_adrs_.for_arm(side)[index]] = position(index);
    }
  }
}

Pose MujocoBackend::MeasuredEePose(Side side) const {
  const int site_id = ee_site_id_.for_arm(side);
  Pose pose;
  pose.position_m = Eigen::Vector3d(data_->site_xpos[3 * site_id + 0],
                                    data_->site_xpos[3 * site_id + 1],
                                    data_->site_xpos[3 * site_id + 2]);
  pose.rotation = RotationFromMujocoMat(&data_->site_xmat[9 * site_id]);
  return pose;
}

void MujocoBackend::SetTargetPose(Side side, const Eigen::Vector3d& position,
                                  const Eigen::Vector4d& quaternion_wxyz) {
  const int mocap = model_->body_mocapid[target_body_id_.for_arm(side)];
  for (int axis = 0; axis < 3; ++axis) {
    data_->mocap_pos[3 * mocap + axis] = position(axis);
  }
  for (int index = 0; index < 4; ++index) {
    data_->mocap_quat[4 * mocap + index] = quaternion_wxyz(index);
  }
}

Eigen::Vector3d MujocoBackend::TargetPosition(Side side) const {
  const int mocap = model_->body_mocapid[target_body_id_.for_arm(side)];
  return Eigen::Vector3d(data_->mocap_pos[3 * mocap + 0],
                         data_->mocap_pos[3 * mocap + 1],
                         data_->mocap_pos[3 * mocap + 2]);
}

Eigen::Vector4d MujocoBackend::TargetQuaternion(Side side) const {
  const int mocap = model_->body_mocapid[target_body_id_.for_arm(side)];
  return Eigen::Vector4d(
      data_->mocap_quat[4 * mocap + 0], data_->mocap_quat[4 * mocap + 1],
      data_->mocap_quat[4 * mocap + 2], data_->mocap_quat[4 * mocap + 3]);
}

void MujocoBackend::SetJointPosition(Side side, const Vector7& position_rad) {
  for (int index = 0; index < kJoints; ++index) {
    data_->qpos[qpos_adrs_.for_arm(side)[index]] = position_rad(index);
  }
}

void MujocoBackend::ZeroVelocities() {
  for (int index = 0; index < model_->nv; ++index) data_->qvel[index] = 0.0;
}

void MujocoBackend::Forward() { mj_forward(model_, data_); }

}  // namespace srl::sim
