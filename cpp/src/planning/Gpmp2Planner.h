// Optional GPMP2 adapter. The public boundary contains only project types so
// the rest of the controller builds without GPMP2/GTSAM installed.

#pragma once

#include <cstddef>
#include <memory>

#include "config/RuntimeConfig.h"
#include "control/PositionActuation.h"
#include "core/Types.h"
#include "planning/JointTrajectory.h"

namespace srl::planning {

struct Gpmp2Settings {
  std::size_t support_intervals{20};
  std::size_t collision_checks_per_interval{4};
  double duration_s{6.0};
  double output_sample_period_s{0.002};
  double velocity_limit_utilization{0.90};
  double endpoint_tolerance_rad{0.01};
  double sdf_cell_size_m{0.04};
  double required_clearance_m{0.02};
  double planning_margin_m{0.03};
  double obstacle_cost_sigma_m{0.005};
  double joint_limit_sigma_rad{0.001};
  double velocity_limit_sigma_rad_s{0.001};
  double endpoint_sigma_rad{0.0005};
  double endpoint_velocity_sigma_rad_s{1e-3};
  std::size_t max_optimizer_iterations{300};

  void Validate() const;
};

struct Gpmp2Request {
  Side side{Side::Left};
  Vector7 start_position_rad{Vector7::Zero()};
  Vector7 start_velocity_rad_s{Vector7::Zero()};
  Vector7 goal_position_rad{Vector7::Zero()};
  Vector7 goal_velocity_rad_s{Vector7::Zero()};
  MountCalibration mount_calibration;
  control::PositionActuationLimits limits;
  config::HumanSafetyConfig human_safety;
  Gpmp2Settings settings;
};

struct Gpmp2Result {
  std::shared_ptr<const JointTrajectory> trajectory;
  double initial_graph_error{0.0};
  double final_graph_error{0.0};
  double minimum_planner_sphere_clearance_m{0.0};
  std::size_t planning_sphere_count{0};
  std::size_t external_obstacle_sphere_count{0};
  std::size_t support_point_count{0};
  double output_sample_period_s{0.0};
  double retiming_scale{1.0};
  double maximum_start_error_rad{0.0};
  double maximum_goal_error_rad{0.0};
};

// Runs synchronously. Call outside the real-time control loop.
Gpmp2Result PlanWithGpmp2(const Gpmp2Request& request);

}  // namespace srl::planning
