// Native Cartesian planner geometry, optimisation and simulator composition.

#include "TestSupport.h"

#include <algorithm>
#include <vector>

#include "config/RuntimeConfig.h"
#include "control/HumanSafety.h"
#include "math/Transforms.h"
#include "planning/CartesianPlanner.h"
#include "sim/MujocoBackend.h"

using namespace srl;
using namespace srl::planning;

namespace {

config::HumanSafetyConfig HumanConfig() {
  config::HumanSafetyConfig config;
  config.enabled = true;
  config.center_xy_torso_m = {0.0, 0.0};
  config.radius_m = 0.25;
  config.z_min_torso_m = -1.10;
  config.z_max_torso_m = 0.70;
  config.clearance_m = 0.02;
  config.control_margin_m = 0.02;
  config.activation_distance_m = 0.10;
  config.recovery_gain_s_inv = 4.0;
  config.approach_velocity_damping = 0.0;
  config.projection_iterations = 50;
  config.constraint_tolerance_m_s = 1e-6;
  return config;
}

config::PlanningConfig PlanningConfig() {
  config::PlanningConfig config;
  config.enabled = true;
  config.arm = "right";
  config.waypoint_count = 3;
  config.dense_samples = 60;
  config.clearance_margin_m = 0.05;
  config.smoothness_weight = 1.0;
  config.obstacle_weight = 40.0;
  config.max_iterations = 200;
  config.tool_radius_m = 0.0;
  config.deviation_weight = 0.5;
  config.reach_allowance_m = 0.45;
  config.lead_compensation_enabled = true;
  config.replan_clearance_trigger_m = 0.01;
  config.include_floor = false;
  config.floor_height_world_m = 0.0;
  config.include_torso_box = false;
  config.torso_box_half_extent_m = Eigen::Vector3d(0.12, 0.18, 0.28);
  config.max_linear_speed_m_s = 0.2;
  config.max_linear_acceleration_m_s2 = 0.5;
  config.max_angular_speed_rad_s = 0.5;
  config.max_angular_acceleration_rad_s2 = 1.0;
  return config;
}

Pose IdentityPose(const Eigen::Vector3d& position) {
  Pose pose;
  pose.position_m = position;
  pose.rotation = Eigen::Matrix3d::Identity();
  return pose;
}

double Distance(const CartesianObstacleSet& obstacles,
                const Eigen::Vector3d& point) {
  return obstacles.Evaluate(point).distance_m;
}

Eigen::Vector3d FiniteDifferenceGradient(
    const CartesianObstacleSet& obstacles, const Eigen::Vector3d& point) {
  constexpr double step = 1e-6;
  Eigen::Vector3d result;
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d offset = Eigen::Vector3d::Zero();
    offset(axis) = step;
    result(axis) =
        (Distance(obstacles, point + offset) -
         Distance(obstacles, point - offset)) /
        (2.0 * step);
  }
  return result;
}

void HumanCylinderMatchesRealtimeGeometry() {
  const config::HumanSafetyConfig human = HumanConfig();
  const config::PlanningConfig planning = PlanningConfig();
  const Pose torso = IdentityPose(Eigen::Vector3d::Zero());
  const CartesianObstacleSet obstacles(human, planning, torso);
  const std::vector<Eigen::Vector3d> points = {
      {0.60, 0.10, 0.00}, {-0.45, 0.20, -0.30},
      {0.05, 0.00, 1.20}, {-0.40, -0.30, -1.45}};
  for (const Eigen::Vector3d& point : points) {
    double expected_distance = 0.0;
    Eigen::Vector3d expected_gradient;
    control::FiniteCylinderDistanceGradient(
        point, human, expected_distance, expected_gradient);
    const DistanceGradient actual = obstacles.Evaluate(point);
    CHECK_CLOSE(actual.distance_m, expected_distance - human.clearance_m,
                1e-15,
                "planner human clearance matches the realtime surface");
    CHECK_MATRIX(actual.gradient, expected_gradient, 1e-15,
                 "planner human gradient matches the realtime surface");
  }
}

void BoxAndFloorGradientsMatchFiniteDifferences() {
  config::HumanSafetyConfig human = HumanConfig();
  human.center_xy_torso_m = {100.0, 100.0};
  human.radius_m = 0.01;
  human.clearance_m = 0.0;

  config::PlanningConfig box_config = PlanningConfig();
  box_config.include_torso_box = true;
  const Pose torso = IdentityPose(Eigen::Vector3d(0.0, 0.0, 1.0));
  const CartesianObstacleSet box(human, box_config, torso);
  for (const Eigen::Vector3d& point :
       {Eigen::Vector3d(0.40, 0.30, 0.50),
        Eigen::Vector3d(0.05, 0.02, 0.10)}) {
    const DistanceGradient actual = box.Evaluate(point);
    CHECK_MATRIX(actual.gradient, FiniteDifferenceGradient(box, point), 1e-6,
                 "box gradient matches a central finite difference");
  }

  config::PlanningConfig floor_config = PlanningConfig();
  floor_config.include_floor = true;
  floor_config.floor_height_world_m = 0.30;
  const Pose rotated_torso{
      Eigen::Vector3d(0.35, -0.20, 1.05),
      transforms::RotationFromRpy(Eigen::Vector3d(0.20, -0.30, 0.50))};
  const CartesianObstacleSet floor(human, floor_config, rotated_torso);
  const Eigen::Vector3d point_torso(0.30, -0.25, -0.40);
  const Eigen::Vector3d point_world =
      rotated_torso.rotation * point_torso +
      rotated_torso.position_m;
  const DistanceGradient actual = floor.Evaluate(point_torso);
  CHECK_CLOSE(actual.distance_m,
              point_world(2) - floor_config.floor_height_world_m, 1e-12,
              "torso-frame floor distance equals world height");
  CHECK_MATRIX(rotated_torso.rotation * actual.gradient,
               Eigen::Vector3d::UnitZ(), 1e-12,
               "floor gradient rotates back to world +z");
}

void SplineBasisReproducesTheDeliveredTrajectory() {
  const std::vector<double> times = {0.0, 0.7, 1.1, 2.6, 4.4};
  std::vector<double> samples;
  for (int index = 0; index < 37; ++index) {
    samples.push_back(4.4 * static_cast<double>(index) / 36.0);
  }
  Eigen::MatrixXd knots(times.size(), 3);
  knots << -0.8, 0.2, 0.7, -0.2, -0.5, 0.9, 0.3, 0.4, 1.1,
      0.7, -0.1, 0.8, 1.0, 0.3, 1.2;

  std::vector<trajectory::CartesianWaypoint> waypoints;
  for (std::size_t index = 0; index < times.size(); ++index) {
    waypoints.push_back(
        {times[index], IdentityPose(knots.row(index).transpose())});
  }
  const trajectory::CartesianWaypointTrajectory trajectory(
      TargetFrame::World, std::move(waypoints));
  Eigen::MatrixXd sampled(samples.size(), 3);
  for (std::size_t index = 0; index < samples.size(); ++index) {
    sampled.row(index) =
        trajectory.Sample(samples[index]).pose.position_m.transpose();
  }

  const Eigen::MatrixXd basis = CartesianSplineBasis(times, samples);
  CHECK_MATRIX(basis * knots, sampled, 1e-12,
               "spline basis reproduces the real trajectory");
  CHECK_MATRIX(basis.rowwise().sum(), Eigen::VectorXd::Ones(samples.size()),
               1e-12, "spline basis is a partition of unity");
}

void CrossingPathIsPushedClear() {
  const Pose torso = IdentityPose(Eigen::Vector3d::Zero());
  const Pose start = IdentityPose(Eigen::Vector3d(0.0, -0.55, 0.20));
  const Pose goal = IdentityPose(Eigen::Vector3d(0.0, 0.55, 0.20));
  const config::PlanningConfig planning = PlanningConfig();
  const CartesianObstacleSet obstacles(HumanConfig(), planning, torso);
  const CartesianPlanResult result = OptimizeCartesianPath(
      start, goal, torso, obstacles, PlanningTrajectoryLimits(planning),
      planning);

  CHECK_TRUE(result.initial_min_clearance_m < 0.0,
             "straight crossing starts inside the wearer");
  CHECK_TRUE(result.final_min_clearance_m >=
                 planning.clearance_margin_m - 1e-3,
             "optimised delivered trajectory reaches the clearance margin");
  CHECK_TRUE(result.success, "crossing plan converges successfully");
  CHECK_MATRIX(result.knots_torso_m.front(), start.position_m, 1e-12,
               "planner keeps the start endpoint fixed");
  CHECK_MATRIX(result.knots_torso_m.back(), goal.position_m, 1e-12,
               "planner keeps the goal endpoint fixed");
  CHECK_TRUE(result.duration_s() > 0.0,
             "planned trajectory receives positive timing");

  // Python planning/path_optimizer.py on this same scene. The solvers need
  // not take the same number of iterations, but the delivered geometry,
  // clearance and timing must agree.
  const std::vector<Eigen::Vector3d> python_knots = {
      {0.0, -0.55, 0.20},
      {0.211403265, -0.280518966, 0.20},
      {0.319996613, 0.0, 0.20},
      {0.211403265, 0.280518966, 0.20},
      {0.0, 0.55, 0.20}};
  for (std::size_t index = 0; index < python_knots.size(); ++index) {
    CHECK_MATRIX(result.knots_torso_m[index], python_knots[index], 1e-6,
                 "native knot matches the Python planner");
  }
  CHECK_CLOSE(result.final_min_clearance_m, 0.049956163, 1e-6,
              "delivered clearance matches the Python planner");
  CHECK_CLOSE(result.duration_s(), 12.062101290, 1e-6,
              "derived duration matches the Python planner");
}

void ClearPathStaysStraight() {
  const Pose torso = IdentityPose(Eigen::Vector3d::Zero());
  const Pose start = IdentityPose(Eigen::Vector3d(0.60, -0.40, 0.30));
  const Pose goal = IdentityPose(Eigen::Vector3d(0.60, 0.40, 0.30));
  const config::PlanningConfig planning = PlanningConfig();
  const CartesianObstacleSet obstacles(HumanConfig(), planning, torso);
  const CartesianPlanResult result = OptimizeCartesianPath(
      start, goal, torso, obstacles, PlanningTrajectoryLimits(planning),
      planning);
  for (std::size_t index = 0; index < result.knots_torso_m.size(); ++index) {
    const double fraction =
        static_cast<double>(index) /
        static_cast<double>(result.knots_torso_m.size() - 1);
    const Eigen::Vector3d expected =
        (1.0 - fraction) * start.position_m + fraction * goal.position_m;
    CHECK_MATRIX(result.knots_torso_m[index], expected, 1e-9,
                 "already-clear plan keeps its straight knots");
  }
  CHECK_TRUE(result.success, "already-clear plan succeeds");
}

void RealSimulatorPlanUsesThePlannedSource() {
  config::ProjectConfig config =
      config::LoadConfig(config::DefaultConfigPath());
  config.planning.enabled = true;
  config.planning.arm = "right";
  config.planning.lead_compensation_enabled = true;
  config.right_target.trajectory.reset();
  sim::MujocoBackend backend(
      std::string(SRL_PYTHON_ROOT) + "/sim/scene.xml", config);
  kinematics::PinModel pin(
      std::string(SRL_PYTHON_ROOT) +
          "/sim/assets/kinova_gen3/gen3.xml",
      "base_link", "pinch_site");
  const PlantState plant = backend.ReadState(Twist::Zero());
  const CartesianArmPlan plan = PlanCartesianArm(
      pin, plant, backend.mount_calibration(), Side::Right, config);

  CHECK_MATRIX(plan.source->Sample(0.0).pose.position_m,
               plan.start_pose_world.position_m, 1e-9,
               "planned source starts at measured world pose");
  CHECK_MATRIX(
      plan.source->Sample(plan.result.duration_s() + 1.0).pose.position_m,
      plan.goal_pose_world.position_m, 1e-9,
      "planned source ends at configured world goal");
  const double middle_time = 0.5 * plan.result.duration_s();
  const control::KinematicTargetSample raw =
      plan.result.trajectory->SampleKinematics(middle_time);
  const FramedTarget delivered = plan.source->Sample(middle_time);
  CHECK_MATRIX(
      delivered.pose.position_m,
      raw.target.pose.position_m +
          raw.target.twist.linear_m_s /
              config.reactive_pose.kp_position_s_inv,
      1e-12, "planned source applies the configured position lead");
  CHECK_TRUE(plan.result.collision_free,
             "the current simulator plan is collision-free at the EE");
}

}  // namespace

int main() {
  HumanCylinderMatchesRealtimeGeometry();
  BoxAndFloorGradientsMatchFiniteDifferences();
  SplineBasisReproducesTheDeliveredTrajectory();
  CrossingPathIsPushedClear();
  ClearPathStaysStraight();
  RealSimulatorPlanUsesThePlannedSource();
  return srl::test::Finish("test_cartesian_planner");
}
