// Explicit reactive-pose to joint-position control pipeline.
//
// Controller and actuation composition only: no MuJoCo reads, no frame
// transforms, no clock, no actuator writes. One-to-one with controller/servo.py.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "config/RuntimeConfig.h"
#include "control/PositionActuation.h"
#include "control/ReactiveController.h"
#include "control/SafetyFilter.h"
#include "core/Types.h"

namespace srl::control {

// Read-only snapshot of one arm's control stages for one cycle. Field order
// matches servo.ControlTrace so the golden-trace columns line up.
struct ControlTrace {
  Matrix6x7 J{Matrix6x7::Zero()};
  Eigen::Vector3d e_pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d e_rot{Eigen::Vector3d::Zero()};
  Eigen::Vector3d e_v{Eigen::Vector3d::Zero()};
  Eigen::Vector3d e_w{Eigen::Vector3d::Zero()};
  Vector6 p_twist{Vector6::Zero()};
  Vector6 d_twist{Vector6::Zero()};
  Vector6 task_twist{Vector6::Zero()};
  Vector7 q{Vector7::Zero()};
  Vector7 qdot_measured{Vector7::Zero()};
  Vector7 qdot_raw{Vector7::Zero()};
  Vector7 qdot_speed_clipped{Vector7::Zero()};
  Vector7 qdot_safety_filtered{Vector7::Zero()};
  Vector7 qdot_effective{Vector7::Zero()};
  Vector7 ctrl_before{Vector7::Zero()};
  Vector7 ctrl_after{Vector7::Zero()};
  JointFlags speed_saturated{};
  JointFlags lead_clamped{};
  JointFlags range_clamped{};
};

// Fixed dual-arm trace record; an unselected arm has no value.
using DualArmControlTraces = DualArm<std::optional<ControlTrace>>;

// Fixed kinematic/actuator facts needed by one arm pipeline.
struct ArmPipelineSetup {
  JointCentering centering;
  PositionActuationLimits actuation_limits;
};

using DualArmPipelineSetup = DualArm<ArmPipelineSetup>;

// Stateful pose-control pipeline; reset means reconstruct this object.
class ReactivePositionPipeline {
 public:
  ReactivePositionPipeline(const PlantState& plant_state,
                           const DualArmPipelineSetup& setup,
                           config::ReactivePoseConfig controller_config,
                           config::HumanSafetyConfig human_safety_config);

  // Current persistent position command for both arms.
  JointPositionCommand Command() const;

  const DualArm<std::optional<SafetyVelocitySolve>>& human_safety_statuses()
      const {
    return last_human_safety_statuses_;
  }

  // Compute one command without reading or writing a plant backend. Pass
  // `human_safety_states = nullptr` to retain the exact pre-safety path.
  std::pair<JointPositionCommand, DualArmControlTraces> Step(
      const DualArmControllerStates& states, const DualArmWorldTargets& targets,
      double dt_s, const std::vector<Side>& arms,
      const DualArmHumanSafetyStates* human_safety_states);

 private:
  struct ArmPipeline {
    std::optional<ReactiveController> controller;
    std::optional<PositionIntegrator> integrator;
    PositionActuationLimits actuation_limits;
    std::unique_ptr<SafetyVelocityProjector> projector;
  };

  DualArm<ArmPipeline> arms_;
  config::HumanSafetyConfig human_safety_config_;
  DualArm<std::optional<SafetyVelocitySolve>> last_human_safety_statuses_;
};

}  // namespace srl::control
