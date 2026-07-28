#include "control/PlannedJointRunner.h"

#include <stdexcept>
#include <utility>

#include "kinematics/LinkSpheres.h"

namespace srl::control {

PlannedJointRunner::PlannedJointRunner(
    sim::PlantBackend& backend, kinematics::PinModel& pin,
    MountCalibration calibration, DualArmPipelineSetup pipeline_setup,
    std::shared_ptr<const planning::JointTrajectory> trajectory, Side side,
    JointTrackingConfig tracking_config,
    config::HumanSafetyConfig human_safety_config)
    : backend_(backend),
      pin_(pin),
      calibration_(std::move(calibration)),
      pipeline_setup_(std::move(pipeline_setup)),
      trajectory_(std::move(trajectory)),
      side_(side),
      controller_(std::move(tracking_config)),
      human_safety_config_(human_safety_config) {
  if (trajectory_ == nullptr) {
    throw std::invalid_argument("joint trajectory must not be null");
  }
}

const PlantState& PlannedJointRunner::current_state() const {
  if (!plant_state_) {
    throw std::runtime_error("planned runner has not taken over the backend");
  }
  return *plant_state_;
}

PlantState PlannedJointRunner::Start() {
  if (plant_state_) {
    throw std::runtime_error("planned runner is already started");
  }
  PlantState initial = backend_.Takeover();
  try {
    const auto first = trajectory_->Sample(trajectory_->start_time_s());
    const double start_error =
        (first.position_rad - initial.arm(side_).position_rad)
            .cwiseAbs()
            .maxCoeff();
    if (start_error > controller_.config().start_tolerance_rad) {
      throw std::runtime_error(
          "measured joints do not match trajectory start: max error=" +
          std::to_string(start_error) + " rad");
    }

    const ArmPipelineSetup& setup = pipeline_setup_.for_arm(side_);
    integrator_.emplace(initial.arm(side_).position_rad,
                        setup.actuation_limits);
    projector_ = std::make_unique<SafetyVelocityProjector>(
        kinematics::kMaxHumanConstraints, human_safety_config_);
  } catch (...) {
    backend_.Release();
    throw;
  }

  plant_state_ = initial;
  command_ = HoldMeasured(initial);
  time_origin_s_ = initial.sample_time_s;
  stopped_ = false;
  return initial;
}

JointPositionCommand PlannedJointRunner::HoldMeasured(
    const PlantState& state) const {
  JointPositionCommand hold;
  hold.position_rad.right = state.arms.right.position_rad;
  hold.position_rad.left = state.arms.left.position_rad;
  return hold;
}

PlannedRunnerCycle PlannedJointRunner::Cycle() {
  if (!plant_state_ || !integrator_ || !projector_) {
    throw std::runtime_error("planned runner must be started before cycling");
  }
  if (stopped_) {
    throw std::runtime_error(
        "planned runner is stopped; close and replan before continuing");
  }

  PlannedRunnerCycle cycle;
  cycle.input_state = *plant_state_;
  cycle.elapsed_time_s =
      cycle.input_state.sample_time_s - time_origin_s_;
  const double trajectory_time =
      trajectory_->start_time_s() + cycle.elapsed_time_s;
  const planning::JointTrajectorySample reference =
      trajectory_->Sample(trajectory_time);
  cycle.tracking =
      controller_.Compute(reference, cycle.input_state.arm(side_));

  if (cycle.tracking.stopped) {
    cycle.command = HoldMeasured(cycle.input_state);
    cycle.execution_stopped = true;
    cycle.stop_reason = cycle.tracking.reason;
    cycle.next_state = backend_.Exchange(cycle.command);
    plant_state_ = cycle.next_state;
    command_ = cycle.command;
    stopped_ = true;
    return cycle;
  }

  cycle.controller_state = kinematics::ArmControllerStateOf(
      pin_, cycle.input_state, side_, calibration_);
  cycle.human_safety_state =
      EvaluateArm(cycle.input_state.torso_pose_world, cycle.controller_state,
                  human_safety_config_);

  Vector7 lower_velocity;
  Vector7 upper_velocity;
  integrator_->VelocityBounds(cycle.controller_state.joints.position_rad,
                              cycle.input_state.nominal_dt_s, lower_velocity,
                              upper_velocity);
  cycle.human_safety_status = ConstrainVelocityForHuman(
      cycle.tracking.requested_velocity_rad_s, lower_velocity, upper_velocity,
      cycle.human_safety_state, human_safety_config_,
      cycle.controller_state.joints.velocity_rad_s, projector_.get());
  cycle.actuation = integrator_->Step(
      cycle.controller_state.joints.position_rad,
      cycle.human_safety_status.qdot_safe, cycle.input_state.nominal_dt_s);

  command_.position_rad.for_arm(side_) = cycle.actuation.command_after_rad;
  cycle.command = command_;
  cycle.execution_stopped = cycle.human_safety_status.stopped;
  cycle.stop_reason =
      cycle.execution_stopped ? cycle.human_safety_status.reason : "";
  cycle.next_state = backend_.Exchange(cycle.command);
  plant_state_ = cycle.next_state;
  stopped_ = cycle.execution_stopped;
  return cycle;
}

void PlannedJointRunner::Close() {
  if (!plant_state_) return;
  backend_.Release();
  plant_state_.reset();
  integrator_.reset();
  projector_.reset();
  time_origin_s_ = 0.0;
  stopped_ = false;
}

}  // namespace srl::control
