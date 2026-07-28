// Explicit runner for the reactive pose-to-position pipeline.
//
// Owns cycle ordering; backend and controller never inspect each other.
// One-to-one with controller/runner.py. Contract clauses J1-J4, K1-K3.

#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "config/RuntimeConfig.h"
#include "control/CylinderRouter.h"
#include "control/HumanSafety.h"
#include "control/Servo.h"
#include "control/TargetSource.h"
#include "core/Types.h"
#include "kinematics/Frames.h"
#include "sim/Backend.h"

namespace srl::control {

// Per-arm routing diagnostics; the C++ ControllerStatus route_* fields.
struct CylinderRouteStatus {
  std::string kind;
  std::size_t waypoint_count{0};
  std::size_t waypoint_index{0};
  bool at_final_waypoint{false};
  bool target_adjusted{false};
  bool route_changed{false};
  Eigen::Vector3d requested_target_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d effective_target_world_m{Eigen::Vector3d::Zero()};
  Eigen::Vector3d active_waypoint_world_m{Eigen::Vector3d::Zero()};
  std::vector<Eigen::Vector3d> waypoints_world_m;
};

// One completed exchange, retaining both sides of the boundary.
struct RunnerCycle {
  PlantState input_state;
  double target_elapsed_time_s{0.0};
  DualArmFramedTargets sampled_targets;
  DualArmWorldTargets resolved_targets;
  DualArmWorldTargets routed_targets;
  DualArmControllerStates controller_states;
  DualArmHumanSafetyStates human_safety_states;
  DualArm<std::optional<SafetyVelocitySolve>> human_safety_statuses;
  JointPositionCommand command;
  DualArmControlTraces traces;
  PlantState next_state;
  DualArm<std::optional<CylinderRouteStatus>> cylinder_routes;
};

class ReactivePositionRunner {
 public:
  ReactivePositionRunner(sim::PlantBackend& backend, kinematics::PinModel& pin,
                         MountCalibration calibration,
                         DualArmPipelineSetup pipeline_setup,
                         std::shared_ptr<DualArmTargetSource> source_targets,
                         std::vector<Side> arms,
                         config::ReactivePoseConfig controller_config,
                         CylinderKeepout cylinder_keepout,
                         config::HumanSafetyConfig human_safety_config);

  const CylinderKeepout& cylinder_keepout() const { return keepout_; }
  const PlantState& current_state() const;

  PlantState Start();
  // `source_override` replaces the retained source for this cycle only, which
  // is how the golden trace feeds marker-derived targets.
  RunnerCycle Cycle(DualArmTargetSource* source_override = nullptr);
  void Close();

 private:
  struct AcceptedTarget {
    bool valid{false};
    TargetFrame frame{TargetFrame::World};
    Eigen::Vector3d position{Eigen::Vector3d::Zero()};

    bool Matches(TargetFrame other_frame,
                 const Eigen::Vector3d& other_position) const {
      return valid && frame == other_frame &&
             (position.array() == other_position.array()).all();
    }
  };

  void RouteTargets(const DualArmFramedTargets& sampled,
                    const DualArmWorldTargets& resolved,
                    const DualArmControllerStates& controller_states,
                    DualArmWorldTargets& routed,
                    DualArm<std::optional<CylinderRouteStatus>>& statuses);

  sim::PlantBackend& backend_;
  kinematics::PinModel& pin_;
  MountCalibration calibration_;
  DualArmPipelineSetup pipeline_setup_;
  std::shared_ptr<DualArmTargetSource> target_source_;
  std::vector<Side> arms_;
  config::ReactivePoseConfig controller_config_;
  config::HumanSafetyConfig human_safety_config_;
  CylinderKeepout keepout_;

  DualArm<std::optional<CylinderRouteFollower>> followers_;
  DualArm<AcceptedTarget> accepted_target_;

  std::optional<PlantState> plant_state_;
  std::optional<ReactivePositionPipeline> pipeline_;
  double target_time_origin_s_{0.0};
};

}  // namespace srl::control
