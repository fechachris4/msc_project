#include "control/Runner.h"

#include <algorithm>
#include <set>
#include <stdexcept>

namespace srl::control {

StaticDualArmTargetSource::StaticDualArmTargetSource(
    DualArmFramedTargets targets)
    : targets_(std::move(targets)) {
  for (Side side : kSides) {
    const FramedTarget& target = targets_.for_arm(side);
    if (!target.twist.linear_m_s.isZero(0.0) ||
        !target.twist.angular_rad_s.isZero(0.0)) {
      throw std::invalid_argument("static dual-arm targets must have zero twist");
    }
  }
}

StaticTargetSource::StaticTargetSource(FramedTarget target)
    : target_(std::move(target)) {
  if (!target_.twist.linear_m_s.isZero(0.0) ||
      !target_.twist.angular_rad_s.isZero(0.0)) {
    throw std::invalid_argument("a static target must have zero twist");
  }
}

ReactivePositionRunner::ReactivePositionRunner(
    sim::PlantBackend& backend, kinematics::PinModel& pin,
    MountCalibration calibration, DualArmPipelineSetup pipeline_setup,
    std::shared_ptr<DualArmTargetSource> source_targets, std::vector<Side> arms,
    config::ReactivePoseConfig controller_config,
    CylinderKeepout cylinder_keepout,
    config::HumanSafetyConfig human_safety_config)
    : backend_(backend),
      pin_(pin),
      calibration_(std::move(calibration)),
      pipeline_setup_(std::move(pipeline_setup)),
      target_source_(std::move(source_targets)),
      arms_(std::move(arms)),
      controller_config_(controller_config),
      human_safety_config_(human_safety_config),
      keepout_(std::move(cylinder_keepout)) {
  if (arms_.empty()) {
    throw std::invalid_argument("arms must be a non-empty subset of right/left");
  }
  const std::set<Side> unique(arms_.begin(), arms_.end());
  if (unique.size() != arms_.size()) {
    throw std::invalid_argument("arms must not contain duplicates");
  }
  if (target_source_ == nullptr) {
    throw std::invalid_argument("target source must not be null");
  }
  for (Side side : arms_) {
    followers_.for_arm(side).emplace(keepout_);
  }
}

const PlantState& ReactivePositionRunner::current_state() const {
  if (!plant_state_) {
    throw std::runtime_error("runner has not taken over the backend");
  }
  return *plant_state_;
}

PlantState ReactivePositionRunner::Start() {
  if (plant_state_) throw std::runtime_error("runner is already started");
  PlantState plant_state = backend_.Takeover();
  try {
    pipeline_.emplace(plant_state, pipeline_setup_, controller_config_,
                      human_safety_config_);
  } catch (...) {
    backend_.Release();
    throw;
  }
  plant_state_ = plant_state;
  target_time_origin_s_ = plant_state.sample_time_s;

  if (keepout_.enabled) {
    // Seed each follower at the measured pose so the very first Update has a
    // valid single-waypoint route (clause K3).
    const DualArmControllerStates states =
        kinematics::ControllerStates(pin_, plant_state, calibration_);
    for (Side side : arms_) {
      followers_.for_arm(side)->Reset(
          states.for_arm(side).ee_pose_world.position_m);
      accepted_target_.for_arm(side) = AcceptedTarget{};
    }
  }
  return plant_state;
}

void ReactivePositionRunner::RouteTargets(
    const DualArmFramedTargets& sampled, const DualArmWorldTargets& resolved,
    const DualArmControllerStates& controller_states,
    DualArmWorldTargets& routed,
    DualArm<std::optional<CylinderRouteStatus>>& statuses) {
  routed = resolved;
  if (!keepout_.enabled) return;

  for (Side side : arms_) {
    CylinderRouteFollower& follower = *followers_.for_arm(side);
    const Eigen::Vector3d ee_world =
        controller_states.for_arm(side).ee_pose_world.position_m;
    const WorldTarget& target = resolved.for_arm(side);
    const Eigen::Vector3d target_world = target.pose_world.position_m;

    // A world-fixed obstacle needs a fresh route whenever the resolved WORLD
    // target moves -- including a torso-carried target whose sampled local
    // coordinates are unchanged (clause G7).
    const TargetFrame frame = sampled.for_arm(side).reference_frame;
    const bool route_changed =
        !accepted_target_.for_arm(side).Matches(frame, target_world);
    if (route_changed) {
      accepted_target_.for_arm(side) = AcceptedTarget{true, frame, target_world};
      follower.SetTarget(ee_world, target_world);
    }

    const Eigen::Vector3d waypoint_world = follower.Update(ee_world);
    WorldTarget& routed_target = routed.for_arm(side);
    routed_target.pose_world.position_m = waypoint_world;
    routed_target.pose_world.rotation = target.pose_world.rotation;
    routed_target.twist_world = target.twist_world;

    const CylinderRoute& route = follower.route();
    CylinderRouteStatus status;
    status.kind = std::string(RouteKindName(route.kind));
    status.waypoint_count = route.size();
    status.waypoint_index = follower.index();
    status.at_final_waypoint = follower.AtFinalWaypoint();
    status.target_adjusted = route.target_adjusted;
    status.route_changed = route_changed;
    status.requested_target_world_m = route.requested_target;
    status.effective_target_world_m = route.effective_target;
    status.active_waypoint_world_m = waypoint_world;
    status.waypoints_world_m = route.waypoints;
    statuses.for_arm(side) = std::move(status);
  }
}

RunnerCycle ReactivePositionRunner::Cycle(DualArmTargetSource* source_override) {
  if (!plant_state_ || !pipeline_) {
    throw std::runtime_error("runner must be started before cycling");
  }
  RunnerCycle cycle;
  cycle.input_state = *plant_state_;
  cycle.target_elapsed_time_s =
      cycle.input_state.sample_time_s - target_time_origin_s_;

  DualArmTargetSource& source =
      source_override != nullptr ? *source_override : *target_source_;
  cycle.sampled_targets = source.Sample(cycle.target_elapsed_time_s);

  cycle.resolved_targets = kinematics::ResolveTargetsWorld(
      cycle.input_state, calibration_, cycle.sampled_targets);
  cycle.controller_states =
      kinematics::ControllerStates(pin_, cycle.input_state, calibration_);
  cycle.human_safety_states = EvaluateDualArm(
      cycle.input_state, cycle.controller_states, human_safety_config_);

  RouteTargets(cycle.sampled_targets, cycle.resolved_targets,
               cycle.controller_states, cycle.routed_targets,
               cycle.cylinder_routes);

  auto [command, traces] = pipeline_->Step(
      cycle.controller_states, cycle.routed_targets,
      cycle.input_state.nominal_dt_s, arms_, &cycle.human_safety_states);
  cycle.command = command;
  cycle.traces = traces;
  cycle.human_safety_statuses = pipeline_->human_safety_statuses();

  cycle.next_state = backend_.Exchange(cycle.command);
  plant_state_ = cycle.next_state;
  return cycle;
}

void ReactivePositionRunner::Close() {
  if (!plant_state_) return;
  backend_.Release();
  plant_state_.reset();
  pipeline_.reset();
  target_time_origin_s_ = 0.0;
}

}  // namespace srl::control
