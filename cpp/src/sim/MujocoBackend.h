// MuJoCo plant backend: model/data ownership, checked ids, state read, command
// apply, stepping, lifecycle.
//
// Unlike sim/world.py there is no module-level default instance -- construction
// is explicit, which the Python project's own risk list asks a C++ port to do.
// One-to-one with sim/world.MujocoBackend otherwise.

#pragma once

#include <array>
#include <functional>
#include <memory>
#include <string>

#include "config/RuntimeConfig.h"
#include "control/Servo.h"
#include "core/Types.h"
#include "sim/Backend.h"

struct mjModel_;
struct mjData_;
using mjModel = mjModel_;
using mjData = mjData_;

namespace srl::sim {

// Scripted torso pose: returns (position, rpy) at simulation time t.
using TorsoPoseFn =
    std::function<std::pair<Eigen::Vector3d, Eigen::Vector3d>(double)>;
// Matching analytic twist: returns (linear, angular) at the same t.
using TorsoTwistFn =
    std::function<std::pair<Eigen::Vector3d, Eigen::Vector3d>(double)>;

struct JointRange {
  Vector7 lower{Vector7::Zero()};
  Vector7 upper{Vector7::Zero()};
  std::array<bool, kJoints> limited{};
};

class MujocoBackend final : public PlantBackend {
 public:
  MujocoBackend(const std::string& scene_path,
                const config::ProjectConfig& config);
  ~MujocoBackend() override;

  MujocoBackend(const MujocoBackend&) = delete;
  MujocoBackend& operator=(const MujocoBackend&) = delete;

  // Install pure scripted pose/twist functions before takeover. Both must be
  // set or both cleared.
  void ConfigureTorsoDriver(TorsoPoseFn pose_at, TorsoTwistFn twist_at);

  PlantState Takeover() override;
  PlantState Exchange(const JointPositionCommand& command) override;
  void Release() override;

  // Reset MuJoCo storage; control state is rebuilt by a new Runner.
  void Reset();

  PlantState ReadState(const Twist& torso_twist) const;
  void ApplyCommand(const JointPositionCommand& command);

  // MuJoCo ground truth; never consumed by controller math.
  Pose MeasuredEePose(Side side) const;
  JointRange JointRangeOf(Side side) const;

  const MountCalibration& mount_calibration() const { return mount_calibration_; }
  const control::DualArmPipelineSetup& pipeline_setup() const {
    return pipeline_setup_;
  }

  mjModel* model() const { return model_; }
  mjData* data() const { return data_; }

  // Display-only target markers (mocap bodies in scene.xml).
  void SetTargetPose(Side side, const Eigen::Vector3d& position,
                     const Eigen::Vector4d& quaternion_wxyz);
  Eigen::Vector3d TargetPosition(Side side) const;
  Eigen::Vector4d TargetQuaternion(Side side) const;

  // Direct qpos access for trajectory initialisation.
  void SetJointPosition(Side side, const Vector7& position_rad);
  void ZeroVelocities();
  void Forward();

 private:
  int NamedId(int object_type, const std::string& name) const;
  Twist RefreshTorso();
  control::ArmPipelineSetup BuildArmPipelineSetup(
      Side side, const config::ProjectConfig& config) const;

  mjModel* model_{nullptr};
  mjData* data_{nullptr};

  DualArm<int> ee_site_id_{};
  DualArm<int> arm_base_id_{};
  DualArm<int> target_body_id_{};
  DualArm<std::array<int, kJoints>> ctrl_adrs_{};
  DualArm<std::array<int, kJoints>> qpos_adrs_{};
  DualArm<std::array<int, kJoints>> dof_adrs_{};
  DualArm<std::array<int, kJoints>> joint_id_{};
  int torso_body_id_{0};
  int torso_mocap_index_{0};

  MountCalibration mount_calibration_{};
  control::DualArmPipelineSetup pipeline_setup_{};

  bool active_{false};
  TorsoPoseFn torso_pose_at_;
  TorsoTwistFn torso_twist_at_;
};

}  // namespace srl::sim
