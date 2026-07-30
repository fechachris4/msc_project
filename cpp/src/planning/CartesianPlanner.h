// Native Cartesian path planning for the reactive srl_sim viewer.
//
// This mirrors planning/planner.py + planning/path_optimizer.py: optimise
// minimum-jerk spline knots in the captured TORSO frame, build the delivered
// trajectory in the WORLD frame, derive timing from the shared Cartesian
// limits, and optionally lead-condition the reference before it reaches the
// unchanged reactive controller.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "config/RuntimeConfig.h"
#include "control/TargetSource.h"
#include "core/Types.h"
#include "kinematics/PinModel.h"
#include "trajectory/Trajectory.h"

namespace srl::planning {

struct DistanceGradient {
  double distance_m{0.0};
  Eigen::Vector3d gradient{Eigen::Vector3d::UnitX()};
};

// The same obstacle set the Python Cartesian planner assembles: the human
// cylinder plus optional torso box and world-fixed floor, all evaluated in the
// torso frame captured at plan time.
class CartesianObstacleSet {
 public:
  CartesianObstacleSet(config::HumanSafetyConfig human_safety,
                       config::PlanningConfig planning,
                       Pose torso_pose_world);

  DistanceGradient Evaluate(const Eigen::Vector3d& point_torso_m) const;

 private:
  config::HumanSafetyConfig human_safety_;
  config::PlanningConfig planning_;
  Pose torso_pose_world_;
};

struct CartesianPlanResult {
  std::shared_ptr<trajectory::CartesianWaypointTrajectory> trajectory;
  std::vector<Eigen::Vector3d> knots_world_m;
  std::vector<Eigen::Vector3d> knots_torso_m;
  double initial_min_clearance_m{0.0};
  double final_min_clearance_m{0.0};
  int evaluations{0};
  bool solver_converged{false};
  bool collision_free{false};
  bool margin_met{false};
  bool success{false};
  std::string message;

  double duration_s() const { return trajectory->duration_s(); }
};

struct CartesianArmPlan {
  Side side{Side::Right};
  CartesianPlanResult result;
  Pose goal_pose_world;
  Pose start_pose_world;
  Pose torso_pose_world;
  std::shared_ptr<control::TargetSource> source;
  bool lead_compensation_enabled{false};
};

bool PlanningIncludesSide(const config::PlanningConfig& config, Side side);

trajectory::TrajectoryLimits PlanningTrajectoryLimits(
    const config::PlanningConfig& config);

// Tested construction of the linear map from knot positions to samples of the
// real CartesianWaypointTrajectory implementation.
Eigen::MatrixXd CartesianSplineBasis(const std::vector<double>& knot_times_s,
                                     const std::vector<double>& sample_times_s);

CartesianPlanResult OptimizeCartesianPath(
    const Pose& start_pose_world, const Pose& goal_pose_world,
    const Pose& torso_pose_world, const CartesianObstacleSet& obstacles,
    const trajectory::TrajectoryLimits& limits,
    const config::PlanningConfig& planning_config);

CartesianArmPlan PlanCartesianArm(
    kinematics::PinModel& pin, const PlantState& plant,
    const MountCalibration& calibration, Side side,
    const config::ProjectConfig& config);

void PrintCartesianPlans(const config::PlanningConfig& config,
                         const std::vector<CartesianArmPlan>& plans);

}  // namespace srl::planning
