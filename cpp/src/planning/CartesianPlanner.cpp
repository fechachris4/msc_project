#include "planning/CartesianPlanner.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <utility>

#include <unsupported/Eigen/LevenbergMarquardt>

#include "control/HumanSafety.h"
#include "kinematics/Frames.h"
#include "math/Transforms.h"

namespace srl::planning {
namespace {

constexpr double kAxisEpsilon = 1e-12;
constexpr double kRetimeToleranceM = 1e-4;
constexpr int kMaxRetimePasses = 3;
constexpr int kDeliveredClearanceSamples = 120;

std::vector<double> UniformTimes(int count, double duration_s) {
  std::vector<double> times(static_cast<std::size_t>(count));
  for (int index = 0; index < count; ++index) {
    times[static_cast<std::size_t>(index)] =
        duration_s * static_cast<double>(index) /
        static_cast<double>(count - 1);
  }
  return times;
}

Eigen::MatrixXd SecondDifferenceMatrix(int count) {
  Eigen::MatrixXd matrix =
      Eigen::MatrixXd::Zero(std::max(0, count - 2), count);
  for (int row = 0; row < count - 2; ++row) {
    matrix(row, row) = 1.0;
    matrix(row, row + 1) = -2.0;
    matrix(row, row + 2) = 1.0;
  }
  return matrix;
}

Eigen::MatrixXd KnotsFromVector(const Eigen::VectorXd& values,
                                int free_count) {
  Eigen::MatrixXd knots(free_count, 3);
  for (int knot = 0; knot < free_count; ++knot) {
    for (int axis = 0; axis < 3; ++axis) {
      knots(knot, axis) = values(3 * knot + axis);
    }
  }
  return knots;
}

Eigen::VectorXd VectorFromKnots(const Eigen::MatrixXd& knots) {
  Eigen::VectorXd values(knots.rows() * 3);
  for (int knot = 0; knot < knots.rows(); ++knot) {
    for (int axis = 0; axis < 3; ++axis) {
      values(3 * knot + axis) = knots(knot, axis);
    }
  }
  return values;
}

struct ResidualEvaluation {
  Eigen::VectorXd residual;
  Eigen::MatrixXd jacobian;
};

class KnotProblem {
 public:
  KnotProblem(const Eigen::MatrixXd& straight,
              const std::vector<double>& knot_times_s,
              const CartesianObstacleSet& obstacles,
              const config::PlanningConfig& config)
      : straight_(straight), obstacles_(obstacles), config_(config) {
    knot_count_ = static_cast<int>(straight_.rows());
    free_count_ = knot_count_ - 2;

    const std::vector<double> sample_times =
        UniformTimes(config_.dense_samples, knot_times_s.back());
    const Eigen::MatrixXd basis =
        CartesianSplineBasis(knot_times_s, sample_times);
    basis_free_ = basis.middleCols(1, free_count_);
    basis_fixed_.resize(config_.dense_samples, 3);
    for (int row = 0; row < config_.dense_samples; ++row) {
      basis_fixed_.row(row) =
          basis(row, 0) * straight_.row(0) +
          basis(row, knot_count_ - 1) * straight_.row(knot_count_ - 1);
    }

    const Eigen::MatrixXd smoothness =
        SecondDifferenceMatrix(knot_count_);
    smoothness_free_ = smoothness.middleCols(1, free_count_);
    smoothness_fixed_.resize(smoothness.rows(), 3);
    for (int row = 0; row < smoothness.rows(); ++row) {
      smoothness_fixed_.row(row) =
          smoothness(row, 0) * straight_.row(0) +
          smoothness(row, knot_count_ - 1) *
              straight_.row(knot_count_ - 1);
    }
  }

  Eigen::MatrixXd DensePoints(const Eigen::VectorXd& values) const {
    return basis_free_ * KnotsFromVector(values, free_count_) + basis_fixed_;
  }

  ResidualEvaluation Evaluate(const Eigen::VectorXd& values,
                              bool with_jacobian) const {
    const Eigen::MatrixXd free_knots = KnotsFromVector(values, free_count_);
    const Eigen::MatrixXd points = basis_free_ * free_knots + basis_fixed_;
    const Eigen::MatrixXd smooth =
        smoothness_free_ * free_knots + smoothness_fixed_;
    const Eigen::MatrixXd deviation =
        free_knots - straight_.middleRows(1, free_count_);

    const int smooth_rows = static_cast<int>(smooth.rows()) * 3;
    const int deviation_rows = free_count_ * 3;
    const int obstacle_start = smooth_rows + deviation_rows;
    const int residual_count = obstacle_start + config_.dense_samples;

    ResidualEvaluation result;
    result.residual = Eigen::VectorXd::Zero(residual_count);
    if (with_jacobian) {
      result.jacobian =
          Eigen::MatrixXd::Zero(residual_count, free_count_ * 3);
    }

    for (int row = 0; row < smooth.rows(); ++row) {
      for (int axis = 0; axis < 3; ++axis) {
        result.residual(3 * row + axis) =
            config_.smoothness_weight * smooth(row, axis);
        if (!with_jacobian) continue;
        for (int knot = 0; knot < free_count_; ++knot) {
          result.jacobian(3 * row + axis, 3 * knot + axis) =
              config_.smoothness_weight * smoothness_free_(row, knot);
        }
      }
    }

    for (int knot = 0; knot < free_count_; ++knot) {
      for (int axis = 0; axis < 3; ++axis) {
        const int row = smooth_rows + 3 * knot + axis;
        result.residual(row) =
            config_.deviation_weight * deviation(knot, axis);
        if (with_jacobian) {
          result.jacobian(row, 3 * knot + axis) =
              config_.deviation_weight;
        }
      }
    }

    for (int sample = 0; sample < config_.dense_samples; ++sample) {
      const DistanceGradient obstacle =
          obstacles_.Evaluate(points.row(sample).transpose());
      const double clearance = obstacle.distance_m - config_.tool_radius_m;
      if (clearance >= config_.clearance_margin_m) continue;
      const int row = obstacle_start + sample;
      result.residual(row) =
          config_.obstacle_weight *
          (config_.clearance_margin_m - clearance);
      if (!with_jacobian) continue;
      for (int knot = 0; knot < free_count_; ++knot) {
        const double weight = basis_free_(sample, knot);
        for (int axis = 0; axis < 3; ++axis) {
          result.jacobian(row, 3 * knot + axis) =
              -config_.obstacle_weight * obstacle.gradient(axis) * weight;
        }
      }
    }
    return result;
  }

 private:
  Eigen::MatrixXd straight_;
  const CartesianObstacleSet& obstacles_;
  const config::PlanningConfig& config_;
  int knot_count_{0};
  int free_count_{0};
  Eigen::MatrixXd basis_free_;
  Eigen::MatrixXd basis_fixed_;
  Eigen::MatrixXd smoothness_free_;
  Eigen::MatrixXd smoothness_fixed_;
};

struct SolverResult {
  Eigen::VectorXd values;
  int evaluations{0};
  bool converged{false};
  std::string message;
};

class BoundedKnotFunctor : public Eigen::DenseFunctor<double> {
 public:
  BoundedKnotFunctor(const KnotProblem& problem, Eigen::VectorXd lower,
                     Eigen::VectorXd upper, int residual_count)
      : Eigen::DenseFunctor<double>(
            static_cast<int>(lower.size()), residual_count),
        problem_(problem),
        midpoint_(0.5 * (lower + upper)),
        half_range_(0.5 * (upper - lower)) {}

  Eigen::VectorXd ToValues(const Eigen::VectorXd& parameters) const {
    return midpoint_ +
           (half_range_.array() * parameters.array().tanh()).matrix();
  }

  Eigen::VectorXd ToParameters(const Eigen::VectorXd& values) const {
    Eigen::ArrayXd unit =
        ((values - midpoint_).array() / half_range_.array())
            .max(-1.0 + 1e-12)
            .min(1.0 - 1e-12);
    return unit.atanh().matrix();
  }

  int operator()(const Eigen::VectorXd& parameters,
                 Eigen::VectorXd& residual) const {
    residual = problem_.Evaluate(ToValues(parameters), false).residual;
    return 0;
  }

  int df(const Eigen::VectorXd& parameters,
         Eigen::MatrixXd& jacobian) const {
    const ResidualEvaluation evaluation =
        problem_.Evaluate(ToValues(parameters), true);
    const Eigen::ArrayXd tanh = parameters.array().tanh();
    const Eigen::VectorXd derivative =
        (half_range_.array() * (1.0 - tanh.square())).matrix();
    jacobian = evaluation.jacobian * derivative.asDiagonal();
    return 0;
  }

 private:
  const KnotProblem& problem_;
  Eigen::VectorXd midpoint_;
  Eigen::VectorXd half_range_;
};

SolverResult SolveKnotProblem(const KnotProblem& problem,
                              const Eigen::VectorXd& initial,
                              const Eigen::VectorXd& lower,
                              const Eigen::VectorXd& upper,
                              int max_evaluations) {
  const int residual_count =
      static_cast<int>(problem.Evaluate(initial, false).residual.size());
  BoundedKnotFunctor functor(problem, lower, upper, residual_count);
  Eigen::VectorXd parameters = functor.ToParameters(initial);
  Eigen::LevenbergMarquardt<BoundedKnotFunctor> solver(functor);
  solver.setMaxfev(max_evaluations);
  solver.setFtol(1e-10);
  solver.setXtol(1e-10);
  solver.setGtol(1e-10);
  const Eigen::LevenbergMarquardtSpace::Status status =
      solver.minimize(parameters);

  SolverResult result;
  result.values = functor.ToValues(parameters);
  result.evaluations = static_cast<int>(solver.nfev());
  result.converged =
      status >= Eigen::LevenbergMarquardtSpace::RelativeReductionTooSmall &&
      status <= Eigen::LevenbergMarquardtSpace::CosinusTooSmall;
  switch (status) {
    case Eigen::LevenbergMarquardtSpace::RelativeReductionTooSmall:
      result.message = "cost tolerance reached";
      break;
    case Eigen::LevenbergMarquardtSpace::RelativeErrorTooSmall:
      result.message = "step tolerance reached";
      break;
    case Eigen::LevenbergMarquardtSpace::RelativeErrorAndReductionTooSmall:
      result.message = "cost and step tolerances reached";
      break;
    case Eigen::LevenbergMarquardtSpace::CosinusTooSmall:
      result.message = "gradient tolerance reached";
      break;
    case Eigen::LevenbergMarquardtSpace::TooManyFunctionEvaluation:
      result.message = "maximum function evaluations reached";
      break;
    default:
      result.message = "Levenberg-Marquardt failed with status " +
                       std::to_string(static_cast<int>(status));
      break;
  }
  return result;
}

double MinimumClearance(const Eigen::MatrixXd& points_torso_m,
                        const CartesianObstacleSet& obstacles,
                        double tool_radius_m) {
  double smallest = std::numeric_limits<double>::infinity();
  for (int row = 0; row < points_torso_m.rows(); ++row) {
    smallest =
        std::min(smallest,
                 obstacles.Evaluate(points_torso_m.row(row).transpose())
                         .distance_m -
                     tool_radius_m);
  }
  return smallest;
}

double TrajectoryMinimumClearance(
    const trajectory::CartesianWaypointTrajectory& trajectory,
    const Pose& torso_pose_world, const CartesianObstacleSet& obstacles,
    double tool_radius_m) {
  Eigen::MatrixXd points(kDeliveredClearanceSamples, 3);
  const Eigen::Matrix3d rotation_torso_world =
      torso_pose_world.rotation.transpose();
  for (int index = 0; index < kDeliveredClearanceSamples; ++index) {
    const double time =
        trajectory.duration_s() * static_cast<double>(index) /
        static_cast<double>(kDeliveredClearanceSamples - 1);
    const Eigen::Vector3d point_world =
        trajectory.Sample(time).pose.position_m;
    points.row(index) =
        (rotation_torso_world *
         (point_world - torso_pose_world.position_m))
            .transpose();
  }
  return MinimumClearance(points, obstacles, tool_radius_m);
}

std::shared_ptr<trajectory::CartesianWaypointTrajectory> BuildTrajectory(
    const Eigen::MatrixXd& knots_torso_m, const Pose& start_pose_world,
    const Pose& goal_pose_world, const Pose& torso_pose_world,
    const trajectory::TrajectoryLimits& limits,
    std::vector<Eigen::Vector3d>& knots_world_m) {
  const int knot_count = static_cast<int>(knots_torso_m.rows());
  knots_world_m.clear();
  knots_world_m.reserve(static_cast<std::size_t>(knot_count));
  std::vector<Pose> poses;
  poses.reserve(static_cast<std::size_t>(knot_count));
  const Eigen::Vector3d relative_rotation = trajectory::RotationVector(
      start_pose_world.rotation.transpose() * goal_pose_world.rotation);
  for (int index = 0; index < knot_count; ++index) {
    const Eigen::Vector3d point_world =
        torso_pose_world.rotation * knots_torso_m.row(index).transpose() +
        torso_pose_world.position_m;
    knots_world_m.push_back(point_world);
    const double fraction =
        static_cast<double>(index) / static_cast<double>(knot_count - 1);
    Pose pose;
    pose.position_m = point_world;
    pose.rotation =
        start_pose_world.rotation *
        trajectory::RotationFromVector(fraction * relative_rotation);
    poses.push_back(pose);
  }
  return trajectory::TimedWaypointTrajectory(
      TargetFrame::World, poses, std::nullopt, limits);
}

bool SameTimes(const std::vector<double>& left,
               const std::vector<double>& right) {
  if (left.size() != right.size()) return false;
  for (std::size_t index = 0; index < left.size(); ++index) {
    if (std::abs(left[index] - right[index]) > 1e-9) return false;
  }
  return true;
}

class LeadCompensatedTargetSource final : public control::TargetSource {
 public:
  LeadCompensatedTargetSource(
      std::shared_ptr<trajectory::CartesianWaypointTrajectory> source,
      const config::ReactivePoseConfig& controller)
      : source_(std::move(source)),
        kp_position_s_inv_(controller.kp_position_s_inv),
        kp_rotation_s_inv_(controller.kp_rotation_s_inv) {
    if (kp_position_s_inv_ <= 0.0 || kp_rotation_s_inv_ <= 0.0) {
      throw std::invalid_argument(
          "lead compensation needs positive position and rotation gains");
    }
  }

  TargetFrame reference_frame() const override {
    return source_->reference_frame();
  }

  control::KinematicTargetSample SampleKinematics(
      double elapsed_time_s) const override {
    control::KinematicTargetSample sample =
        source_->SampleKinematics(elapsed_time_s);
    sample.target.pose.position_m +=
        sample.target.twist.linear_m_s / kp_position_s_inv_;
    sample.target.pose.rotation =
        trajectory::RotationFromVector(
            sample.target.twist.angular_rad_s / kp_rotation_s_inv_) *
        sample.target.pose.rotation;
    return sample;
  }

 private:
  std::shared_ptr<trajectory::CartesianWaypointTrajectory> source_;
  double kp_position_s_inv_{0.0};
  double kp_rotation_s_inv_{0.0};
};

}  // namespace

CartesianObstacleSet::CartesianObstacleSet(
    config::HumanSafetyConfig human_safety,
    config::PlanningConfig planning, Pose torso_pose_world)
    : human_safety_(std::move(human_safety)),
      planning_(std::move(planning)),
      torso_pose_world_(std::move(torso_pose_world)) {}

DistanceGradient CartesianObstacleSet::Evaluate(
    const Eigen::Vector3d& point_torso_m) const {
  DistanceGradient nearest;
  control::FiniteCylinderDistanceGradient(
      point_torso_m, human_safety_, nearest.distance_m, nearest.gradient);
  nearest.distance_m -= human_safety_.clearance_m;

  if (planning_.include_torso_box) {
    const Eigen::Vector3d excess =
        point_torso_m.cwiseAbs() - planning_.torso_box_half_extent_m;
    const Eigen::Vector3d outside = excess.cwiseMax(0.0);
    const double outside_norm = outside.norm();
    DistanceGradient box;
    if (outside_norm > kAxisEpsilon) {
      box.distance_m = outside_norm;
      for (int axis = 0; axis < 3; ++axis) {
        const double sign = point_torso_m(axis) < 0.0 ? -1.0 : 1.0;
        box.gradient(axis) = sign * outside(axis) / outside_norm;
      }
    } else {
      Eigen::Index axis = 0;
      box.distance_m = excess.maxCoeff(&axis);
      box.gradient.setZero();
      box.gradient(axis) = point_torso_m(axis) < 0.0 ? -1.0 : 1.0;
    }
    if (box.distance_m < nearest.distance_m) nearest = box;
  }

  if (planning_.include_floor) {
    const Eigen::Vector3d normal_torso =
        torso_pose_world_.rotation.transpose() * Eigen::Vector3d::UnitZ();
    const double offset_torso =
        planning_.floor_height_world_m - torso_pose_world_.position_m(2);
    DistanceGradient floor;
    floor.distance_m =
        point_torso_m.dot(normal_torso) - offset_torso;
    floor.gradient = normal_torso;
    if (floor.distance_m < nearest.distance_m) nearest = floor;
  }
  return nearest;
}

bool PlanningIncludesSide(const config::PlanningConfig& config, Side side) {
  if (config.arm == "both") return true;
  return config.arm == SideName(side);
}

trajectory::TrajectoryLimits PlanningTrajectoryLimits(
    const config::PlanningConfig& config) {
  trajectory::TrajectoryLimits limits;
  limits.max_linear_speed_m_s = config.max_linear_speed_m_s;
  limits.max_linear_acceleration_m_s2 =
      config.max_linear_acceleration_m_s2;
  limits.max_angular_speed_rad_s = config.max_angular_speed_rad_s;
  limits.max_angular_acceleration_rad_s2 =
      config.max_angular_acceleration_rad_s2;
  return limits;
}

Eigen::MatrixXd CartesianSplineBasis(
    const std::vector<double>& knot_times_s,
    const std::vector<double>& sample_times_s) {
  if (knot_times_s.size() < 2) {
    throw std::invalid_argument("knot_times_s needs at least two values");
  }
  Eigen::MatrixXd basis(sample_times_s.size(), knot_times_s.size());
  for (std::size_t knot = 0; knot < knot_times_s.size(); ++knot) {
    std::vector<trajectory::CartesianWaypoint> waypoints;
    waypoints.reserve(knot_times_s.size());
    for (std::size_t index = 0; index < knot_times_s.size(); ++index) {
      Pose pose;
      pose.position_m =
          Eigen::Vector3d(index == knot ? 1.0 : 0.0, 0.0, 0.0);
      pose.rotation = Eigen::Matrix3d::Identity();
      waypoints.push_back({knot_times_s[index], pose});
    }
    const trajectory::CartesianWaypointTrajectory trajectory(
        TargetFrame::World, std::move(waypoints));
    for (std::size_t sample = 0; sample < sample_times_s.size(); ++sample) {
      basis(sample, knot) =
          trajectory.Sample(sample_times_s[sample]).pose.position_m(0);
    }
  }
  return basis;
}

CartesianPlanResult OptimizeCartesianPath(
    const Pose& start_pose_world, const Pose& goal_pose_world,
    const Pose& torso_pose_world, const CartesianObstacleSet& obstacles,
    const trajectory::TrajectoryLimits& limits,
    const config::PlanningConfig& planning_config) {
  const int knot_count = planning_config.waypoint_count + 2;
  const int free_count = knot_count - 2;
  const Eigen::Matrix3d rotation_torso_world =
      torso_pose_world.rotation.transpose();
  const Eigen::Vector3d start_torso =
      rotation_torso_world *
      (start_pose_world.position_m - torso_pose_world.position_m);
  const Eigen::Vector3d goal_torso =
      rotation_torso_world *
      (goal_pose_world.position_m - torso_pose_world.position_m);

  Eigen::MatrixXd straight(knot_count, 3);
  for (int index = 0; index < knot_count; ++index) {
    const double fraction =
        static_cast<double>(index) / static_cast<double>(knot_count - 1);
    straight.row(index) =
        ((1.0 - fraction) * start_torso + fraction * goal_torso).transpose();
  }
  const Eigen::VectorXd initial =
      VectorFromKnots(straight.middleRows(1, free_count));

  Eigen::VectorXd lower(initial.size());
  Eigen::VectorXd upper(initial.size());
  const Eigen::Vector3d corner_low =
      start_torso.cwiseMin(goal_torso).array() -
      planning_config.reach_allowance_m;
  const Eigen::Vector3d corner_high =
      start_torso.cwiseMax(goal_torso).array() +
      planning_config.reach_allowance_m;
  for (int knot = 0; knot < free_count; ++knot) {
    for (int axis = 0; axis < 3; ++axis) {
      lower(3 * knot + axis) = corner_low(axis);
      upper(3 * knot + axis) = corner_high(axis);
    }
  }

  CartesianPlanResult result;
  std::vector<double> times = UniformTimes(knot_count, 1.0);
  {
    const KnotProblem initial_problem(straight, times, obstacles,
                                      planning_config);
    result.initial_min_clearance_m =
        MinimumClearance(initial_problem.DensePoints(initial), obstacles,
                         planning_config.tool_radius_m);
  }

  bool have_best = false;
  double best_clearance = -std::numeric_limits<double>::infinity();
  SolverResult best_solver;
  Eigen::MatrixXd best_knots_torso;
  std::vector<Eigen::Vector3d> best_knots_world;
  std::shared_ptr<trajectory::CartesianWaypointTrajectory> best_trajectory;
  int total_evaluations = 0;
  int passes = 0;

  for (int pass = 0; pass < kMaxRetimePasses; ++pass) {
    ++passes;
    const KnotProblem problem(straight, times, obstacles, planning_config);
    const SolverResult solver =
        SolveKnotProblem(problem, initial, lower, upper,
                         planning_config.max_iterations);
    total_evaluations += solver.evaluations;

    Eigen::MatrixXd knots_torso = straight;
    knots_torso.middleRows(1, free_count) =
        KnotsFromVector(solver.values, free_count);
    std::vector<Eigen::Vector3d> knots_world;
    const auto trajectory =
        BuildTrajectory(knots_torso, start_pose_world, goal_pose_world,
                        torso_pose_world, limits, knots_world);
    const double clearance = TrajectoryMinimumClearance(
        *trajectory, torso_pose_world, obstacles,
        planning_config.tool_radius_m);
    if (!have_best || clearance > best_clearance) {
      have_best = true;
      best_clearance = clearance;
      best_solver = solver;
      best_knots_torso = knots_torso;
      best_knots_world = knots_world;
      best_trajectory = trajectory;
    }
    if (clearance >=
        planning_config.clearance_margin_m - kRetimeToleranceM) {
      break;
    }
    std::vector<double> actual_times = {0.0};
    const std::vector<double> boundaries = trajectory->boundary_times_s();
    actual_times.insert(actual_times.end(), boundaries.begin(),
                        boundaries.end());
    if (SameTimes(actual_times, times)) break;
    times = std::move(actual_times);
  }

  result.trajectory = std::move(best_trajectory);
  result.knots_world_m = std::move(best_knots_world);
  result.knots_torso_m.reserve(
      static_cast<std::size_t>(best_knots_torso.rows()));
  for (int row = 0; row < best_knots_torso.rows(); ++row) {
    result.knots_torso_m.push_back(best_knots_torso.row(row).transpose());
  }
  result.final_min_clearance_m = best_clearance;
  result.evaluations = total_evaluations;
  result.solver_converged = best_solver.converged;
  result.collision_free = best_clearance >= 0.0;
  result.margin_met =
      best_clearance >=
      planning_config.clearance_margin_m - kRetimeToleranceM;
  result.success = result.solver_converged && result.margin_met;
  result.message = best_solver.message;
  if (passes > 1) {
    result.message += " (" + std::to_string(passes) +
                      " timing passes, best kept)";
  }
  return result;
}

CartesianArmPlan PlanCartesianArm(
    kinematics::PinModel& pin, const PlantState& plant,
    const MountCalibration& calibration, Side side,
    const config::ProjectConfig& config) {
  const DualArmControllerStates states =
      kinematics::ControllerStates(pin, plant, calibration);
  const Pose start_pose = states.for_arm(side).ee_pose_world;

  const config::TargetConfig& target_config = config.target(side);
  FramedTarget framed_goal;
  framed_goal.reference_frame =
      TargetFrameFromName(target_config.reference_frame);
  framed_goal.pose.position_m = target_config.position_m;
  framed_goal.pose.rotation =
      transforms::RotationFromRpy(target_config.rpy_rad);
  framed_goal.twist = Twist::Zero();
  const Pose goal_pose =
      kinematics::ResolveTargetWorld(plant, side, calibration, framed_goal)
          .pose_world;

  const CartesianObstacleSet obstacles(
      config.human_safety, config.planning, plant.torso_pose_world);
  CartesianArmPlan plan;
  plan.side = side;
  plan.start_pose_world = start_pose;
  plan.goal_pose_world = goal_pose;
  plan.torso_pose_world = plant.torso_pose_world;
  plan.result = OptimizeCartesianPath(
      start_pose, goal_pose, plant.torso_pose_world, obstacles,
      PlanningTrajectoryLimits(config.planning), config.planning);
  plan.lead_compensation_enabled =
      config.planning.lead_compensation_enabled;
  if (plan.lead_compensation_enabled) {
    plan.source = std::make_shared<LeadCompensatedTargetSource>(
        plan.result.trajectory, config.reactive_pose);
  } else {
    plan.source = plan.result.trajectory;
  }
  return plan;
}

void PrintCartesianPlans(const config::PlanningConfig& config,
                         const std::vector<CartesianArmPlan>& plans) {
  std::printf(
      "planning: %s  arms=%s  waypoints=%d (%d knots incl. endpoints)  "
      "margin=%.0f mm\n",
      config.enabled ? "ENABLED" : "DISABLED",
      config.arm.c_str(), config.waypoint_count,
      config.waypoint_count + 2, config.clearance_margin_m * 1000.0);
  std::printf(
      "  collision optimised in the TORSO frame; reference delivered in "
      "the WORLD frame\n");
  for (const CartesianArmPlan& plan : plans) {
    const CartesianPlanResult& result = plan.result;
    std::printf(
        "  %-5s clearance %.1f -> %.1f mm  duration=%.2f s  "
        "evaluations=%d  %s%s%s\n",
        std::string(SideName(plan.side)).c_str(),
        result.initial_min_clearance_m * 1000.0,
        result.final_min_clearance_m * 1000.0, result.duration_s(),
        result.evaluations, result.success ? "OK" : "FAILED: ",
        result.success ? "" : result.message.c_str(),
        result.collision_free ? "" : " (trajectory penetrates obstacle)");
  }
  std::printf(
      "  lead compensation: %s; END-EFFECTOR path only, whole-arm "
      "protection remains with [human_safety]\n",
      config.lead_compensation_enabled ? "ENABLED" : "DISABLED");
}

}  // namespace srl::planning
