#include "planning/Gpmp2Planner.h"

#include <gpmp2/gp/GaussianProcessPriorLinear.h>
#include <gpmp2/kinematics/Arm.h>
#include <gpmp2/kinematics/ArmModel.h>
#include <gpmp2/kinematics/JointLimitFactorVector.h>
#include <gpmp2/kinematics/VelocityLimitFactorVector.h>
#include <gpmp2/obstacle/ObstacleSDFFactorArm.h>
#include <gpmp2/obstacle/ObstacleSDFFactorGPArm.h>
#include <gpmp2/obstacle/SignedDistanceField.h>
#include <gpmp2/planner/TrajUtils.h>

#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/Values.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace srl::planning {
namespace {

constexpr double kPi = 3.14159265358979323846;

gtsam::Vector ToGtsam(const Vector7& source) {
  gtsam::Vector result(kJoints);
  result = source;
  return result;
}

Vector7 FromGtsam(const gtsam::Vector& source, const char* name) {
  if (source.size() != kJoints || !source.allFinite()) {
    throw std::runtime_error(std::string(name) +
                             " is not a finite seven-joint vector");
  }
  return source;
}

double CappedCylinderSignedDistance(
    const gtsam::Point3& point,
    const config::HumanSafetyConfig& human) {
  const double dx = point.x() - human.center_xy_torso_m[0];
  const double dy = point.y() - human.center_xy_torso_m[1];
  const double radial_signed = std::hypot(dx, dy) - human.radius_m;
  const double vertical_signed =
      std::max(human.z_min_torso_m - point.z(),
               point.z() - human.z_max_torso_m);
  if (radial_signed > 0.0 && vertical_signed > 0.0) {
    return std::hypot(radial_signed, vertical_signed);
  }
  return std::max(radial_signed, vertical_signed);
}

gpmp2::SignedDistanceField MakeHumanSdf(
    const config::HumanSafetyConfig& human, double cell_size_m) {
  const gtsam::Point3 origin(-1.50, -1.50, -1.30);
  const gtsam::Point3 upper(1.50, 1.50, 1.70);
  const std::size_t x_count = static_cast<std::size_t>(
      std::floor((upper.x() - origin.x()) / cell_size_m)) + 1;
  const std::size_t y_count = static_cast<std::size_t>(
      std::floor((upper.y() - origin.y()) / cell_size_m)) + 1;
  const std::size_t z_count = static_cast<std::size_t>(
      std::floor((upper.z() - origin.z()) / cell_size_m)) + 1;

  std::vector<gtsam::Matrix> layers(
      z_count, gtsam::Matrix::Zero(y_count, x_count));
  for (std::size_t z = 0; z < z_count; ++z) {
    for (std::size_t y = 0; y < y_count; ++y) {
      for (std::size_t x = 0; x < x_count; ++x) {
        const gtsam::Point3 point(
            origin.x() + static_cast<double>(x) * cell_size_m,
            origin.y() + static_cast<double>(y) * cell_size_m,
            origin.z() + static_cast<double>(z) * cell_size_m);
        layers[z](static_cast<Eigen::Index>(y),
                  static_cast<Eigen::Index>(x)) =
            CappedCylinderSignedDistance(point, human);
      }
    }
  }
  return gpmp2::SignedDistanceField(origin, cell_size_m, layers);
}

gpmp2::BodySphereVector PlanningSpheres() {
  // Deliberately small, non-mount planning proxy. The exact 18-sphere
  // Pinocchio model is checked by ValidateForExecution after optimization and
  // by the safety filter during execution.
  return {
      gpmp2::BodySphere(2, 0.02, gtsam::Point3(0.0, 0.0, 0.0)),
      gpmp2::BodySphere(4, 0.02, gtsam::Point3(0.0, 0.0, 0.0)),
      gpmp2::BodySphere(6, 0.02, gtsam::Point3(0.0, 0.0, 0.0)),
  };
}

gpmp2::ArmModel MakeKinovaModel(const Pose& torso_to_base) {
  const gtsam::Vector a = gtsam::Vector::Zero(kJoints);
  gtsam::Vector alpha(kJoints);
  alpha << kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0,
      kPi / 2.0, kPi / 2.0, kPi;
  gtsam::Vector d(kJoints);
  d << -0.2848, -0.0118, -0.4208, -0.0128, -0.3143, 0.0, -0.3074;
  gtsam::Vector theta_bias(kJoints);
  theta_bias << 0.0, kPi, kPi, kPi, kPi, kPi, kPi;

  const gtsam::Pose3 torso_T_base(
      gtsam::Rot3(torso_to_base.rotation),
      gtsam::Point3(torso_to_base.position_m));
  return gpmp2::ArmModel(
      gpmp2::Arm(kJoints, a, alpha, d, torso_T_base, theta_bias),
      PlanningSpheres());
}

std::pair<gtsam::Vector, gtsam::Vector> FiniteJointLimits(
    const control::PositionActuationLimits& limits) {
  gtsam::Vector lower = ToGtsam(limits.lower_position_rad);
  gtsam::Vector upper = ToGtsam(limits.upper_position_rad);
  for (int joint = 0; joint < kJoints; ++joint) {
    if (!std::isfinite(lower(joint))) lower(joint) = -2.0 * kPi;
    if (!std::isfinite(upper(joint))) upper(joint) = 2.0 * kPi;
  }
  return {lower, upper};
}

std::map<std::size_t, gtsam::Vector> Sequence(
    const gtsam::Values& values, unsigned char requested_symbol) {
  std::map<std::size_t, gtsam::Vector> sequence;
  for (const gtsam::Key key : values.keys()) {
    const gtsam::Symbol symbol(key);
    if (symbol.chr() == requested_symbol) {
      sequence.emplace(static_cast<std::size_t>(symbol.index()),
                       values.at<gtsam::Vector>(key));
    }
  }
  return sequence;
}

double MinimumPlannerClearance(
    const gpmp2::ArmModel& arm, const gpmp2::SignedDistanceField& sdf,
    const std::map<std::size_t, gtsam::Vector>& positions) {
  double minimum = std::numeric_limits<double>::infinity();
  for (const auto& [unused_index, position] : positions) {
    (void)unused_index;
    std::vector<gtsam::Point3> centers;
    arm.sphereCenters(position, centers);
    for (std::size_t sphere = 0; sphere < centers.size(); ++sphere) {
      minimum = std::min(
          minimum,
          sdf.getSignedDistance(centers[sphere]) - arm.sphere_radius(sphere));
    }
  }
  return minimum;
}

}  // namespace

void Gpmp2Settings::Validate() const {
  if (support_intervals == 0) {
    throw std::invalid_argument("support_intervals must be positive");
  }
  if (collision_checks_per_interval == 0) {
    throw std::invalid_argument(
        "collision_checks_per_interval must be positive");
  }
  const double values[] = {
      duration_s,
      sdf_cell_size_m,
      required_clearance_m,
      planning_margin_m,
      obstacle_cost_sigma_m,
      joint_limit_sigma_rad,
      velocity_limit_sigma_rad_s,
      endpoint_sigma_rad,
      endpoint_velocity_sigma_rad_s,
  };
  for (double value : values) {
    if (!std::isfinite(value) || value <= 0.0) {
      throw std::invalid_argument(
          "GPMP2 scalar settings must be finite and positive");
    }
  }
  if (max_optimizer_iterations == 0) {
    throw std::invalid_argument("max_optimizer_iterations must be positive");
  }
}

Gpmp2Result PlanWithGpmp2(const Gpmp2Request& request) {
  request.settings.Validate();
  request.limits.Validate();
  if (!request.start_position_rad.allFinite() ||
      !request.start_velocity_rad_s.allFinite() ||
      !request.goal_position_rad.allFinite() ||
      !request.goal_velocity_rad_s.allFinite()) {
    throw std::invalid_argument("GPMP2 request joint vectors must be finite");
  }

  const auto [lower, upper] = FiniteJointLimits(request.limits);
  const gtsam::Vector start = ToGtsam(request.start_position_rad);
  const gtsam::Vector start_velocity =
      ToGtsam(request.start_velocity_rad_s);
  const gtsam::Vector goal = ToGtsam(request.goal_position_rad);
  const gtsam::Vector goal_velocity = ToGtsam(request.goal_velocity_rad_s);
  if ((start.array() < lower.array()).any() ||
      (start.array() > upper.array()).any() ||
      (goal.array() < lower.array()).any() ||
      (goal.array() > upper.array()).any()) {
    throw std::invalid_argument(
        "GPMP2 start or goal violates a joint-position limit");
  }

  const Gpmp2Settings& settings = request.settings;
  const double delta_t =
      settings.duration_s / static_cast<double>(settings.support_intervals);
  const gpmp2::ArmModel arm =
      MakeKinovaModel(request.mount_calibration.for_arm(request.side));
  const gpmp2::SignedDistanceField sdf =
      MakeHumanSdf(request.human_safety, settings.sdf_cell_size_m);
  const auto endpoint_model = gtsam::noiseModel::Isotropic::Sigma(
      kJoints, settings.endpoint_sigma_rad);
  const auto endpoint_velocity_model =
      gtsam::noiseModel::Isotropic::Sigma(
          kJoints, settings.endpoint_velocity_sigma_rad_s);
  const auto joint_limit_model = gtsam::noiseModel::Isotropic::Sigma(
      kJoints, settings.joint_limit_sigma_rad);
  const auto velocity_limit_model = gtsam::noiseModel::Isotropic::Sigma(
      kJoints, settings.velocity_limit_sigma_rad_s);
  const auto qc_model = gtsam::noiseModel::Gaussian::Covariance(
      gtsam::Matrix::Identity(kJoints, kJoints));
  const gtsam::Vector joint_threshold =
      gtsam::Vector::Constant(kJoints, 0.10);
  const gtsam::Vector velocity_threshold =
      gtsam::Vector::Constant(kJoints, 0.05);
  const gtsam::Vector velocity_limits =
      ToGtsam(request.limits.velocity_rad_s);
  const double obstacle_epsilon =
      settings.required_clearance_m + settings.planning_margin_m;

  gtsam::NonlinearFactorGraph graph;
  for (std::size_t index = 0; index <= settings.support_intervals; ++index) {
    const gtsam::Symbol q_key('x', index);
    const gtsam::Symbol v_key('v', index);
    if (index == 0) {
      graph.add(gtsam::PriorFactor<gtsam::Vector>(
          q_key, start, endpoint_model));
      graph.add(gtsam::PriorFactor<gtsam::Vector>(
          v_key, start_velocity, endpoint_velocity_model));
    }
    if (index == settings.support_intervals) {
      graph.add(
          gtsam::PriorFactor<gtsam::Vector>(q_key, goal, endpoint_model));
      graph.add(gtsam::PriorFactor<gtsam::Vector>(
          v_key, goal_velocity, endpoint_velocity_model));
    }
    graph.add(gpmp2::JointLimitFactorVector(
        q_key, joint_limit_model, lower, upper, joint_threshold));
    graph.add(gpmp2::VelocityLimitFactorVector(
        v_key, velocity_limit_model, velocity_limits, velocity_threshold));
    graph.add(gpmp2::ObstacleSDFFactorArm(
        q_key, arm, sdf, settings.obstacle_cost_sigma_m, obstacle_epsilon));

    if (index == 0) continue;
    const gtsam::Symbol previous_q('x', index - 1);
    const gtsam::Symbol previous_v('v', index - 1);
    graph.add(gpmp2::GaussianProcessPriorLinear(
        previous_q, previous_v, q_key, v_key, delta_t, qc_model));
    for (std::size_t check = 1;
         check <= settings.collision_checks_per_interval; ++check) {
      const double tau =
          delta_t * static_cast<double>(check) /
          static_cast<double>(settings.collision_checks_per_interval + 1);
      graph.add(gpmp2::ObstacleSDFFactorGPArm(
          previous_q, previous_v, q_key, v_key, arm, sdf,
          settings.obstacle_cost_sigma_m, obstacle_epsilon, qc_model, delta_t,
          tau));
    }
  }

  const gtsam::Values initial = gpmp2::initArmTrajStraightLine(
      start, goal, settings.support_intervals);
  gtsam::LevenbergMarquardtParams parameters;
  parameters.setVerbosity("ERROR");
  parameters.setRelativeErrorTol(1e-6);
  parameters.setAbsoluteErrorTol(1e-6);
  parameters.setMaxIterations(settings.max_optimizer_iterations);
  parameters.setlambdaInitial(1e-5);
  parameters.setlambdaFactor(10.0);
  const gtsam::Values optimized =
      gtsam::LevenbergMarquardtOptimizer(graph, initial, parameters)
          .optimize();
  const gtsam::Values dense = gpmp2::interpolateArmTraj(
      optimized, qc_model, delta_t,
      settings.collision_checks_per_interval);

  const auto positions = Sequence(dense, 'x');
  const auto velocities = Sequence(dense, 'v');
  if (positions.size() < 2 || positions.size() != velocities.size()) {
    throw std::runtime_error(
        "GPMP2 returned mismatched position and velocity sequences");
  }
  std::vector<JointTrajectoryPoint> points;
  points.reserve(positions.size());
  const double dense_dt =
      settings.duration_s / static_cast<double>(positions.size() - 1);
  std::size_t sample = 0;
  for (const auto& [index, position] : positions) {
    const auto velocity = velocities.find(index);
    if (velocity == velocities.end()) {
      throw std::runtime_error("GPMP2 dense trajectory is missing a velocity");
    }
    points.push_back(JointTrajectoryPoint{
        static_cast<double>(sample++) * dense_dt,
        FromGtsam(position, "GPMP2 position"),
        FromGtsam(velocity->second, "GPMP2 velocity"),
    });
  }

  Gpmp2Result result;
  result.trajectory =
      std::make_shared<const JointTrajectory>(std::move(points));
  result.initial_graph_error = graph.error(initial);
  result.final_graph_error = graph.error(optimized);
  result.minimum_planner_sphere_clearance_m =
      MinimumPlannerClearance(arm, sdf, positions);
  return result;
}

}  // namespace srl::planning
