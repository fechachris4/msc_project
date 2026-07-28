// Execute one precomputed joint-space trajectory through the existing plant,
// whole-arm human-safety filter and persistent position-command integrator.

#pragma once

#include <memory>
#include <optional>

#include "config/RuntimeConfig.h"
#include "control/HumanSafety.h"
#include "control/JointTrajectoryController.h"
#include "control/PositionActuation.h"
#include "control/SafetyFilter.h"
#include "control/Servo.h"
#include "core/Types.h"
#include "kinematics/Frames.h"
#include "planning/JointTrajectory.h"
#include "sim/Backend.h"

namespace srl::control {

struct PlannedRunnerCycle {
  PlantState input_state;
  double elapsed_time_s{0.0};
  JointTrackingOutput tracking;
  ArmControllerState controller_state;
  ArmHumanSafetyState human_safety_state;
  SafetyVelocitySolve human_safety_status;
  PositionActuation actuation;
  JointPositionCommand command;
  PlantState next_state;
  bool execution_stopped{false};
  std::string stop_reason;
};

class PlannedJointRunner {
 public:
  PlannedJointRunner(sim::PlantBackend& backend, kinematics::PinModel& pin,
                     MountCalibration calibration,
                     DualArmPipelineSetup pipeline_setup,
                     std::shared_ptr<const planning::JointTrajectory> trajectory,
                     Side side, JointTrackingConfig tracking_config,
                     config::HumanSafetyConfig human_safety_config);

  const PlantState& current_state() const;
  bool stopped() const { return stopped_; }

  PlantState Start();
  PlannedRunnerCycle Cycle();
  void Close();

 private:
  JointPositionCommand HoldMeasured(const PlantState& state) const;

  sim::PlantBackend& backend_;
  kinematics::PinModel& pin_;
  MountCalibration calibration_;
  DualArmPipelineSetup pipeline_setup_;
  std::shared_ptr<const planning::JointTrajectory> trajectory_;
  Side side_;
  JointTrajectoryController controller_;
  config::HumanSafetyConfig human_safety_config_;

  std::optional<PlantState> plant_state_;
  std::optional<PositionIntegrator> integrator_;
  std::unique_ptr<SafetyVelocityProjector> projector_;
  JointPositionCommand command_;
  double time_origin_s_{0.0};
  bool stopped_{false};
};

}  // namespace srl::control
