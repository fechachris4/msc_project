// Standalone GPMP2 joint-space planning demonstration.
//
// This file does not connect to MuJoCo, the reactive Runner, or a robot.  It
// demonstrates the factor graph that should sit above the real-time safety
// filter: start/goal joint priors, GP smoothness, joint and velocity limits,
// and sphere-to-human signed-distance costs at support and interpolated states.

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
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kDof = 7;
constexpr double kPi = 3.14159265358979323846;

struct HumanCylinder {
  double center_x_m = 0.0;
  double center_y_m = 0.0;
  double radius_m = 0.25;
  double z_min_m = -1.10;
  double z_max_m = 0.70;
};

struct PlannerSettings {
  std::size_t support_intervals = 20;
  std::size_t collision_checks_per_interval = 4;
  double duration_s = 6.0;
  double sdf_cell_size_m = 0.04;
  double required_clearance_m = 0.02;
  double planning_margin_m = 0.02;
  double obstacle_cost_sigma_m = 0.005;
  double joint_limit_sigma_rad = 0.01;
  double velocity_limit_sigma_rad_s = 0.01;
  double endpoint_sigma_rad = 1e-4;
  double endpoint_velocity_sigma_rad_s = 1e-3;
  std::size_t max_optimizer_iterations = 300;
};

struct PlanResult {
  gtsam::Values optimized_support_states;
  gtsam::Values dense_states;
  double initial_graph_error;
  double final_graph_error;
};

gtsam::Vector vector7(const std::array<double, kDof>& values) {
  gtsam::Vector result(kDof);
  for (std::size_t index = 0; index < kDof; ++index) {
    result(static_cast<Eigen::Index>(index)) = values[index];
  }
  return result;
}

double cappedCylinderSignedDistance(
    const gtsam::Point3& point,
    const HumanCylinder& cylinder) {
  const double dx = point.x() - cylinder.center_x_m;
  const double dy = point.y() - cylinder.center_y_m;
  const double radial_signed =
      std::hypot(dx, dy) - cylinder.radius_m;
  const double vertical_signed = std::max(
      cylinder.z_min_m - point.z(),
      point.z() - cylinder.z_max_m);

  if (radial_signed > 0.0 && vertical_signed > 0.0) {
    return std::hypot(radial_signed, vertical_signed);
  }
  return std::max(radial_signed, vertical_signed);
}

gpmp2::SignedDistanceField makeHumanSdf(
    const HumanCylinder& cylinder,
    double cell_size_m) {
  // The field and the arm base pose are both expressed in the torso frame.
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
        layers[z](
            static_cast<Eigen::Index>(y),
            static_cast<Eigen::Index>(x)) =
            cappedCylinderSignedDistance(point, cylinder);
      }
    }
  }
  return gpmp2::SignedDistanceField(origin, cell_size_m, layers);
}

gpmp2::BodySphereVector makeDemonstrationSpheres() {
  // Minimal DH-frame geometry keeps this file focused on the GPMP2 graph.
  // It is deliberately not presented as a collision model of the real arm.
  // The mount links are omitted because they intersect the torso by design.
  return {
      gpmp2::BodySphere(2, 0.02, gtsam::Point3(0.0, 0.0, 0.0)),
      gpmp2::BodySphere(4, 0.02, gtsam::Point3(0.0, 0.0, 0.0)),
      gpmp2::BodySphere(6, 0.02, gtsam::Point3(0.0, 0.0, 0.0)),
  };
}

gpmp2::ArmModel makeLeftKinovaModel() {
  const gtsam::Vector a = gtsam::Vector::Zero(kDof);
  const gtsam::Vector alpha = vector7({
      kPi / 2.0, kPi / 2.0, kPi / 2.0, kPi / 2.0,
      kPi / 2.0, kPi / 2.0, kPi,
  });
  const gtsam::Vector d = vector7({
      -0.2848, -0.0118, -0.4208, -0.0128,
      -0.3143, 0.0, -0.3074,
  });
  const gtsam::Vector theta_bias = vector7({
      0.0, kPi, kPi, kPi, kPi, kPi, kPi,
  });

  // Left arm mount from sim/scene.xml, expressed in the torso frame.
  const gtsam::Pose3 torso_T_base(
      gtsam::Rot3::Rx(-0.845708),
      gtsam::Point3(-0.16, 0.10, 0.14));
  const gpmp2::Arm arm(
      kDof, a, alpha, d, torso_T_base, theta_bias);
  return gpmp2::ArmModel(arm, makeDemonstrationSpheres());
}

gtsam::Vector lowerJointLimits() {
  return vector7({
      -2.0 * kPi, -2.24, -2.0 * kPi, -2.57,
      -2.0 * kPi, -2.09, -2.0 * kPi,
  });
}

gtsam::Vector upperJointLimits() {
  return vector7({
      2.0 * kPi, 2.24, 2.0 * kPi, 2.57,
      2.0 * kPi, 2.09, 2.0 * kPi,
  });
}

gtsam::Vector velocityLimits() {
  return vector7({
      1.3892820845874863,
      1.3892820845874863,
      1.3892820845874863,
      1.3892820845874863,
      1.2199851471440364,
      1.2199851471440364,
      1.2199851471440364,
  });
}

void checkInputConfiguration(
    const gtsam::Vector& q,
    const std::string& name) {
  const gtsam::Vector lower = lowerJointLimits();
  const gtsam::Vector upper = upperJointLimits();
  if (q.size() != static_cast<Eigen::Index>(kDof) ||
      !q.allFinite()) {
    throw std::invalid_argument(name + " must contain seven finite joints");
  }
  if ((q.array() < lower.array()).any() ||
      (q.array() > upper.array()).any()) {
    throw std::invalid_argument(name + " violates a joint limit");
  }
}

PlanResult planJointTrajectory(
    const gpmp2::ArmModel& arm,
    const gpmp2::SignedDistanceField& sdf,
    const gtsam::Vector& start_q,
    const gtsam::Vector& goal_q,
    const PlannerSettings& settings) {
  checkInputConfiguration(start_q, "start_q");
  checkInputConfiguration(goal_q, "goal_q");

  const double delta_t =
      settings.duration_s /
      static_cast<double>(settings.support_intervals);
  const gtsam::Vector zero_velocity = gtsam::Vector::Zero(kDof);
  const gtsam::Vector joint_threshold =
      gtsam::Vector::Constant(kDof, 0.10);
  const gtsam::Vector velocity_threshold =
      gtsam::Vector::Constant(kDof, 0.05);
  const auto endpoint_model = gtsam::noiseModel::Isotropic::Sigma(
      kDof, settings.endpoint_sigma_rad);
  const auto endpoint_velocity_model =
      gtsam::noiseModel::Isotropic::Sigma(
          kDof, settings.endpoint_velocity_sigma_rad_s);
  const auto joint_limit_model = gtsam::noiseModel::Isotropic::Sigma(
      kDof, settings.joint_limit_sigma_rad);
  const auto velocity_limit_model =
      gtsam::noiseModel::Isotropic::Sigma(
          kDof, settings.velocity_limit_sigma_rad_s);
  const auto qc_model = gtsam::noiseModel::Gaussian::Covariance(
      gtsam::Matrix::Identity(kDof, kDof));
  const double obstacle_epsilon =
      settings.required_clearance_m + settings.planning_margin_m;

  gtsam::NonlinearFactorGraph graph;
  for (std::size_t index = 0;
       index <= settings.support_intervals;
       ++index) {
    const gtsam::Symbol q_key('x', index);
    const gtsam::Symbol v_key('v', index);

    if (index == 0) {
      graph.add(gtsam::PriorFactor<gtsam::Vector>(
          q_key, start_q, endpoint_model));
      graph.add(gtsam::PriorFactor<gtsam::Vector>(
          v_key, zero_velocity, endpoint_velocity_model));
    }
    if (index == settings.support_intervals) {
      graph.add(gtsam::PriorFactor<gtsam::Vector>(
          q_key, goal_q, endpoint_model));
      graph.add(gtsam::PriorFactor<gtsam::Vector>(
          v_key, zero_velocity, endpoint_velocity_model));
    }

    graph.add(gpmp2::JointLimitFactorVector(
        q_key,
        joint_limit_model,
        lowerJointLimits(),
        upperJointLimits(),
        joint_threshold));
    graph.add(gpmp2::VelocityLimitFactorVector(
        v_key,
        velocity_limit_model,
        velocityLimits(),
        velocity_threshold));
    graph.add(gpmp2::ObstacleSDFFactorArm(
        q_key,
        arm,
        sdf,
        settings.obstacle_cost_sigma_m,
        obstacle_epsilon));

    if (index == 0) {
      continue;
    }
    const gtsam::Symbol previous_q('x', index - 1);
    const gtsam::Symbol previous_v('v', index - 1);
    graph.add(gpmp2::GaussianProcessPriorLinear(
        previous_q,
        previous_v,
        q_key,
        v_key,
        delta_t,
        qc_model));

    for (std::size_t check = 1;
         check <= settings.collision_checks_per_interval;
         ++check) {
      const double tau =
          delta_t * static_cast<double>(check) /
          static_cast<double>(
              settings.collision_checks_per_interval + 1);
      graph.add(gpmp2::ObstacleSDFFactorGPArm(
          previous_q,
          previous_v,
          q_key,
          v_key,
          arm,
          sdf,
          settings.obstacle_cost_sigma_m,
          obstacle_epsilon,
          qc_model,
          delta_t,
          tau));
    }
  }

  const gtsam::Values initial =
      gpmp2::initArmTrajStraightLine(
          start_q, goal_q, settings.support_intervals);
  gtsam::LevenbergMarquardtParams parameters;
  parameters.setVerbosity("ERROR");
  parameters.setRelativeErrorTol(1e-6);
  parameters.setAbsoluteErrorTol(1e-6);
  parameters.setMaxIterations(settings.max_optimizer_iterations);
  parameters.setlambdaInitial(1e-5);
  parameters.setlambdaFactor(10.0);

  gtsam::LevenbergMarquardtOptimizer optimizer(
      graph, initial, parameters);
  const gtsam::Values optimized = optimizer.optimize();
  const gtsam::Values dense = gpmp2::interpolateArmTraj(
      optimized,
      qc_model,
      delta_t,
      settings.collision_checks_per_interval);
  return PlanResult{
      optimized,
      dense,
      graph.error(initial),
      graph.error(optimized),
  };
}

std::vector<std::pair<std::uint64_t, gtsam::Vector>> symbolSequence(
    const gtsam::Values& values,
    unsigned char requested_symbol) {
  std::vector<std::pair<std::uint64_t, gtsam::Vector>> sequence;
  for (const gtsam::Key key : values.keys()) {
    const gtsam::Symbol symbol(key);
    if (symbol.chr() == requested_symbol) {
      sequence.emplace_back(
          symbol.index(), values.at<gtsam::Vector>(key));
    }
  }
  std::sort(
      sequence.begin(),
      sequence.end(),
      [](const auto& left, const auto& right) {
        return left.first < right.first;
      });
  return sequence;
}

std::vector<std::pair<std::uint64_t, gtsam::Vector>>
configurationSequence(const gtsam::Values& values) {
  return symbolSequence(values, 'x');
}

std::vector<std::pair<std::uint64_t, gtsam::Vector>>
velocitySequence(const gtsam::Values& values) {
  return symbolSequence(values, 'v');
}

double minimumSphereClearance(
    const gpmp2::ArmModel& arm,
    const gpmp2::SignedDistanceField& sdf,
    const std::vector<std::pair<std::uint64_t, gtsam::Vector>>& sequence) {
  double minimum = std::numeric_limits<double>::infinity();
  for (const auto& indexed_q : sequence) {
    std::vector<gtsam::Point3> centers;
    arm.sphereCenters(indexed_q.second, centers);
    for (std::size_t sphere = 0; sphere < centers.size(); ++sphere) {
      const double clearance =
          sdf.getSignedDistance(centers[sphere]) -
          arm.sphere_radius(sphere);
      minimum = std::min(minimum, clearance);
    }
  }
  return minimum;
}

double minimumJointLimitMargin(
    const std::vector<std::pair<std::uint64_t, gtsam::Vector>>& sequence) {
  const gtsam::Vector lower = lowerJointLimits();
  const gtsam::Vector upper = upperJointLimits();
  double minimum = std::numeric_limits<double>::infinity();
  for (const auto& indexed_q : sequence) {
    minimum = std::min(
        minimum,
        (indexed_q.second - lower).minCoeff());
    minimum = std::min(
        minimum,
        (upper - indexed_q.second).minCoeff());
  }
  return minimum;
}

double maximumVelocityLimitRatio(
    const std::vector<std::pair<std::uint64_t, gtsam::Vector>>& sequence) {
  const gtsam::Vector limits = velocityLimits();
  double maximum = 0.0;
  for (const auto& indexed_velocity : sequence) {
    maximum = std::max(
        maximum,
        (indexed_velocity.second.array().abs() /
         limits.array()).maxCoeff());
  }
  return maximum;
}

void printCsv(
    const std::vector<std::pair<std::uint64_t, gtsam::Vector>>& sequence,
    double duration_s) {
  std::cout << "time_s";
  for (std::size_t joint = 1; joint <= kDof; ++joint) {
    std::cout << ",q" << joint << "_rad";
  }
  std::cout << '\n';

  const double dt = sequence.size() <= 1
      ? 0.0
      : duration_s / static_cast<double>(sequence.size() - 1);
  std::cout << std::setprecision(12);
  for (std::size_t sample = 0; sample < sequence.size(); ++sample) {
    std::cout << static_cast<double>(sample) * dt;
    for (Eigen::Index joint = 0;
         joint < static_cast<Eigen::Index>(kDof);
         ++joint) {
      std::cout << ',' << sequence[sample].second(joint);
    }
    std::cout << '\n';
  }
}

std::pair<gtsam::Vector, gtsam::Vector> readStartAndGoal(
    int argc,
    char** argv) {
  // No arguments: a small illustrative joint-space request.  Supplying
  // fourteen numbers replaces it with q_start[0:7], q_goal[0:7].
  if (argc == 1) {
    return {
        vector7({
            0.22356, 0.73366, -0.39854, 1.71674,
            -0.53462, 0.91219, 0.62194,
        }),
        vector7({
            -0.70, 0.45, -0.20, 1.20,
            -0.20, 0.75, 0.30,
        }),
    };
  }
  if (argc != 15) {
    throw std::invalid_argument(
        "usage: gpmp2_joint_space_demo "
        "q1_start ... q7_start q1_goal ... q7_goal");
  }
  std::array<double, kDof> start{};
  std::array<double, kDof> goal{};
  for (std::size_t index = 0; index < kDof; ++index) {
    start[index] = std::stod(argv[index + 1]);
    goal[index] = std::stod(argv[index + 1 + kDof]);
  }
  return {vector7(start), vector7(goal)};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const PlannerSettings settings;
    const HumanCylinder human;
    const auto [start_q, goal_q] = readStartAndGoal(argc, argv);
    const gpmp2::ArmModel arm = makeLeftKinovaModel();
    const gpmp2::SignedDistanceField sdf =
        makeHumanSdf(human, settings.sdf_cell_size_m);
    const PlanResult result = planJointTrajectory(
        arm, sdf, start_q, goal_q, settings);
    const auto dense = configurationSequence(result.dense_states);
    if (dense.empty()) {
      throw std::runtime_error("GPMP2 returned no joint configurations");
    }

    const double endpoint_error =
        (dense.back().second - goal_q).norm();
    const double startpoint_error =
        (dense.front().second - start_q).norm();
    const double minimum_clearance =
        minimumSphereClearance(arm, sdf, dense);
    const double minimum_joint_margin =
        minimumJointLimitMargin(dense);
    const auto dense_velocities =
        velocitySequence(result.dense_states);
    const double maximum_velocity_ratio =
        maximumVelocityLimitRatio(dense_velocities);
    std::cerr << std::setprecision(8)
              << "initial graph error: "
              << result.initial_graph_error << '\n'
              << "final graph error:   "
              << result.final_graph_error << '\n'
              << "dense samples:       "
              << dense.size() << '\n'
              << "start joint error:   "
              << startpoint_error << " rad\n"
              << "goal joint error:    "
              << endpoint_error << " rad\n"
              << "joint-limit margin:  "
              << minimum_joint_margin << " rad\n"
              << "max velocity ratio:  "
              << maximum_velocity_ratio << '\n'
              << "minimum clearance:   "
              << minimum_clearance << " m\n";

    if (startpoint_error > 1e-3 || endpoint_error > 1e-3) {
      throw std::runtime_error(
          "optimized trajectory did not satisfy its endpoint priors");
    }
    if (minimum_joint_margin < -1e-6) {
      throw std::runtime_error(
          "optimized trajectory violates a joint-position limit");
    }
    if (maximum_velocity_ratio > 1.0 + 1e-6) {
      throw std::runtime_error(
          "optimized trajectory violates a joint-velocity limit");
    }
    if (minimum_clearance + 1e-6 <
        settings.required_clearance_m) {
      throw std::runtime_error(
          "optimized trajectory violates required sphere clearance");
    }

    printCsv(dense, settings.duration_s);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "planning failed: " << error.what() << '\n';
    return 2;
  }
}
