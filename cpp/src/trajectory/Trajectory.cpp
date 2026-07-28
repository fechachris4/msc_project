#include "trajectory/Trajectory.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/Eigenvalues>

#include "math/LinAlg.h"

namespace srl::trajectory {
namespace {

double RequirePositiveDuration(double value, const char* name) {
  if (!std::isfinite(value) || value <= 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and positive");
  }
  return value;
}

Eigen::Vector3d Vee(const Eigen::Matrix3d& skew) {
  return Eigen::Vector3d(skew(2, 1), skew(0, 2), skew(1, 0));
}

Eigen::Matrix3d Skew(const Eigen::Vector3d& vector) {
  Eigen::Matrix3d result;
  result << 0.0, -vector(2), vector(1), vector(2), 0.0, -vector(0), -vector(1),
      vector(0), 0.0;
  return result;
}

// Choose one deterministic sign for the otherwise ambiguous pi axis.
Eigen::Vector3d CanonicalAxis(const Eigen::Vector3d& axis) {
  Eigen::Vector3d result = axis;
  for (int index = 0; index < 3; ++index) {
    if (std::abs(result(index)) > 1e-12) {
      if (result(index) < 0.0) result *= -1.0;
      break;
    }
  }
  return result;
}

bool ZeroTwist(const FramedTarget& target) {
  return target.twist.linear_m_s.isZero(0.0) &&
         target.twist.angular_rad_s.isZero(0.0);
}

template <typename A, typename B>
bool AllClose(const A& left, const B& right, double atol) {
  return ((left - right).array().abs() <= atol).all();
}

bool SamplesContinuous(const KinematicTargetSample& left,
                       const KinematicTargetSample& right) {
  if (left.target.reference_frame != right.target.reference_frame) return false;
  return AllClose(left.target.pose.position_m, right.target.pose.position_m,
                  kProgramBoundaryAtol) &&
         AllClose(left.target.pose.rotation, right.target.pose.rotation,
                  kProgramBoundaryAtol) &&
         AllClose(left.target.twist.linear_m_s, right.target.twist.linear_m_s,
                  kProgramBoundaryAtol) &&
         AllClose(left.target.twist.angular_rad_s,
                  right.target.twist.angular_rad_s, kProgramBoundaryAtol) &&
         AllClose(left.linear_acceleration_m_s2,
                  right.linear_acceleration_m_s2, kProgramBoundaryAtol) &&
         AllClose(left.angular_acceleration_rad_s2,
                  right.angular_acceleration_rad_s2, kProgramBoundaryAtol);
}

// Ascending normalised-time coefficients for one quintic segment.
// `Components` is 1 for the basis-mapping use and 3 for the trajectory use.
// Returns a Components x 6 matrix: column p is the coefficient of u^p.
template <int Components>
Eigen::Matrix<double, Components, 6> QuinticCoefficients(
    const Eigen::Matrix<double, Components, 1>& start_position,
    const Eigen::Matrix<double, Components, 1>& end_position,
    const Eigen::Matrix<double, Components, 1>& start_velocity,
    const Eigen::Matrix<double, Components, 1>& end_velocity,
    const Eigen::Matrix<double, Components, 1>& start_acceleration,
    const Eigen::Matrix<double, Components, 1>& end_acceleration,
    double duration_s) {
  const double duration = RequirePositiveDuration(duration_s, "duration_s");
  using Vector = Eigen::Matrix<double, Components, 1>;
  const Vector c0 = start_position;
  const Vector c1 = duration * start_velocity;
  const Vector c2 = 0.5 * duration * duration * start_acceleration;

  // Rows are the three end-point constraints, columns the components -- the
  // same (3, N) right-hand side the NumPy builds with np.stack.
  Eigen::Matrix<double, 3, Components> rhs;
  rhs.row(0) = (end_position - c0 - c1 - c2).transpose();
  rhs.row(1) = (duration * end_velocity - c1 - 2.0 * c2).transpose();
  rhs.row(2) = (duration * duration * end_acceleration - 2.0 * c2).transpose();

  Eigen::Matrix3d m;
  m << 1.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 12.0, 20.0;
  const Eigen::MatrixXd solved =
      linalg::SolveGeneral(Eigen::MatrixXd(m), Eigen::MatrixXd(rhs));

  Eigen::Matrix<double, Components, 6> coefficients;
  coefficients.col(0) = c0;
  coefficients.col(1) = c1;
  coefficients.col(2) = c2;
  coefficients.col(3) = solved.row(0).transpose();
  coefficients.col(4) = solved.row(1).transpose();
  coefficients.col(5) = solved.row(2).transpose();
  return coefficients;
}

// np.polynomial.polynomial.polyval: Horner from the highest power down.
Eigen::Vector3d EvaluatePolynomial(const Eigen::Matrix<double, 3, 6>& ascending,
                                   double unit_time, int derivative_order) {
  // Repeated differentiation shrinks the ascending coefficient list.
  Eigen::Matrix<double, 3, 6> values = ascending;
  int width = 6;
  for (int order = 0; order < derivative_order; ++order) {
    for (int power = 1; power < width; ++power) {
      values.col(power - 1) = values.col(power) * static_cast<double>(power);
    }
    --width;
    values.col(width).setZero();
  }
  Eigen::Vector3d result = values.col(width - 1);
  for (int index = width - 2; index >= 0; --index) {
    result = values.col(index) + result * unit_time;
  }
  return result;
}

std::vector<double> PolynomialDerivative(const std::vector<double>& ascending) {
  std::vector<double> result;
  for (std::size_t power = 1; power < ascending.size(); ++power) {
    result.push_back(ascending[power] * static_cast<double>(power));
  }
  if (result.empty()) result.push_back(0.0);
  return result;
}

std::vector<double> PolynomialMultiply(const std::vector<double>& left,
                                       const std::vector<double>& right) {
  std::vector<double> result(left.size() + right.size() - 1, 0.0);
  for (std::size_t i = 0; i < left.size(); ++i) {
    for (std::size_t j = 0; j < right.size(); ++j) {
      result[i + j] += left[i] * right[j];
    }
  }
  return result;
}

// np.polynomial.polynomial.polyroots: trim, then companion-matrix eigenvalues.
std::vector<std::complex<double>> PolynomialRoots(std::vector<double> ascending) {
  while (ascending.size() > 1 && ascending.back() == 0.0) ascending.pop_back();
  if (ascending.size() < 2) return {};
  const int degree = static_cast<int>(ascending.size()) - 1;
  Eigen::MatrixXd companion = Eigen::MatrixXd::Zero(degree, degree);
  for (int row = 1; row < degree; ++row) companion(row, row - 1) = 1.0;
  for (int row = 0; row < degree; ++row) {
    companion(row, degree - 1) = -ascending[row] / ascending[degree];
  }
  Eigen::EigenSolver<Eigen::MatrixXd> solver(companion);
  std::vector<std::complex<double>> roots;
  for (int index = 0; index < degree; ++index) {
    roots.push_back(solver.eigenvalues()(index));
  }
  return roots;
}

// Exact maximum norm of a vector polynomial on normalised [0, 1].
double VectorPolynomialMaximum(const Eigen::Matrix<double, 3, 6>& ascending,
                               int width) {
  std::array<std::vector<double>, 3> components;
  for (int axis = 0; axis < 3; ++axis) {
    components[axis].assign(static_cast<std::size_t>(width), 0.0);
    for (int power = 0; power < width; ++power) {
      components[axis][static_cast<std::size_t>(power)] = ascending(axis, power);
    }
  }

  std::vector<double> norm_squared_derivative(
      static_cast<std::size_t>(2 * width - 2), 0.0);
  for (int axis = 0; axis < 3; ++axis) {
    const std::vector<double> derivative = PolynomialDerivative(components[axis]);
    const std::vector<double> product =
        PolynomialMultiply(components[axis], derivative);
    for (std::size_t index = 0;
         index < product.size() && index < norm_squared_derivative.size();
         ++index) {
      norm_squared_derivative[index] += 2.0 * product[index];
    }
  }

  std::vector<double> candidates = {0.0, 1.0};
  for (const auto& root : PolynomialRoots(norm_squared_derivative)) {
    if (std::abs(root.imag()) < 1e-9 && root.real() > kLimitTolerance &&
        root.real() < 1.0 - kLimitTolerance) {
      candidates.push_back(root.real());
    }
  }

  Eigen::Matrix<double, 3, 6> padded = Eigen::Matrix<double, 3, 6>::Zero();
  padded.leftCols(width) = ascending.leftCols(width);
  double largest = 0.0;
  for (double value : candidates) {
    // width < 6 leaves the tail zeroed, so evaluating the padded polynomial is
    // identical to evaluating the shorter one.
    largest = std::max(largest, EvaluatePolynomial(padded, value, 0).norm());
  }
  return largest;
}

double MinimumDuration(double distance, double angle,
                       const TrajectoryLimits& limits) {
  std::vector<double> candidates;
  if (distance > kLimitTolerance) {
    if (!limits.max_linear_speed_m_s && !limits.max_linear_acceleration_m_s2) {
      throw std::invalid_argument(
          "automatic timing for translation needs a linear speed or "
          "acceleration limit");
    }
    if (limits.max_linear_speed_m_s) {
      candidates.push_back(1.875 * distance / *limits.max_linear_speed_m_s);
    }
    if (limits.max_linear_acceleration_m_s2) {
      candidates.push_back(std::sqrt(10.0 / std::sqrt(3.0) * distance /
                                     *limits.max_linear_acceleration_m_s2));
    }
  }
  if (angle > kLimitTolerance) {
    if (!limits.max_angular_speed_rad_s &&
        !limits.max_angular_acceleration_rad_s2) {
      throw std::invalid_argument(
          "automatic timing for rotation needs an angular speed or "
          "acceleration limit");
    }
    if (limits.max_angular_speed_rad_s) {
      candidates.push_back(1.875 * angle / *limits.max_angular_speed_rad_s);
    }
    if (limits.max_angular_acceleration_rad_s2) {
      candidates.push_back(std::sqrt(10.0 / std::sqrt(3.0) * angle /
                                     *limits.max_angular_acceleration_rad_s2));
    }
  }
  if (candidates.empty()) {
    throw std::invalid_argument("automatic timing requires a moving segment");
  }
  return *std::max_element(candidates.begin(), candidates.end());
}

}  // namespace

// ------------------------------------------------------------------- math

Eigen::Vector3d RotationVector(const Eigen::Matrix3d& rotation) {
  const double cosine =
      std::clamp((rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
  const double angle = std::acos(cosine);
  const Eigen::Matrix3d antisymmetric = rotation - rotation.transpose();

  if (angle < kSmallAngleRad) return 0.5 * Vee(antisymmetric);

  if (M_PI - angle < kNearPiRad) {
    const Eigen::Matrix3d symmetric = 0.5 * (rotation + Eigen::Matrix3d::Identity());
    int index = 0;
    double largest = symmetric(0, 0);
    for (int candidate = 1; candidate < 3; ++candidate) {
      if (symmetric(candidate, candidate) > largest) {
        largest = symmetric(candidate, candidate);
        index = candidate;
      }
    }
    Eigen::Vector3d axis = Eigen::Vector3d::Zero();
    axis(index) = std::sqrt(std::max(symmetric(index, index), 0.0));
    if (axis(index) < 1e-12) {
      throw std::invalid_argument("could not determine the axis of a pi rotation");
    }
    for (int other = 0; other < 3; ++other) {
      if (other != index) axis(other) = symmetric(index, other) / axis(index);
    }
    axis /= axis.norm();
    const Eigen::Vector3d skew_axis = Vee(antisymmetric);
    if (skew_axis.norm() > 1e-10) {
      if (axis.dot(skew_axis) < 0.0) axis *= -1.0;
    } else {
      axis = CanonicalAxis(axis);
    }
    return axis * angle;
  }
  return angle / (2.0 * std::sin(angle)) * Vee(antisymmetric);
}

Eigen::Matrix3d RotationFromVector(const Eigen::Vector3d& rotation_vector) {
  const double angle = rotation_vector.norm();
  const Eigen::Matrix3d cross = Skew(rotation_vector);
  if (angle < kSmallAngleRad) {
    return Eigen::Matrix3d::Identity() + cross + 0.5 * cross * cross;
  }
  return Eigen::Matrix3d::Identity() + (std::sin(angle) / angle) * cross +
         ((1.0 - std::cos(angle)) / (angle * angle)) * cross * cross;
}

void MinimumJerk(double unit_time, double duration_s, double& progress,
                 double& progress_rate, double& progress_acceleration) {
  const double u = unit_time;
  progress = 10.0 * u * u * u - 15.0 * u * u * u * u +
             6.0 * u * u * u * u * u;
  progress_rate =
      (30.0 * u * u - 60.0 * u * u * u + 30.0 * u * u * u * u) / duration_s;
  progress_acceleration =
      (60.0 * u - 180.0 * u * u + 120.0 * u * u * u) / (duration_s * duration_s);
}

Eigen::Vector3d CleanSmall(const Eigen::Vector3d& value, double tolerance) {
  Eigen::Vector3d result = value;
  for (int index = 0; index < 3; ++index) {
    if (std::abs(result(index)) < tolerance) result(index) = 0.0;
  }
  return result;
}

double TrajectoryLimits::RequiredTimeScale(
    const TrajectoryRateBounds& bounds) const {
  double largest = 1.0;
  const auto consider = [&](double actual, const std::optional<double>& limit,
                            bool square_root) {
    if (!limit) return;
    const double ratio = actual / *limit;
    largest = std::max(largest, square_root ? std::sqrt(ratio) : ratio);
  };
  consider(bounds.max_linear_speed_m_s, max_linear_speed_m_s, false);
  consider(bounds.max_linear_acceleration_m_s2, max_linear_acceleration_m_s2,
           true);
  consider(bounds.max_angular_speed_rad_s, max_angular_speed_rad_s, false);
  consider(bounds.max_angular_acceleration_rad_s2,
           max_angular_acceleration_rad_s2, true);
  return largest;
}

void TrajectoryLimits::Validate(const TrajectoryRateBounds& bounds) const {
  const double scale = RequiredTimeScale(bounds);
  if (scale > 1.0 + kLimitTolerance) {
    throw std::invalid_argument(
        "explicit trajectory timing violates Cartesian limits; required time "
        "scale=" + std::to_string(scale));
  }
}

// ------------------------------------------------------------------- hold

HoldTrajectory::HoldTrajectory(FramedTarget target, double duration_s)
    : target_(std::move(target)),
      duration_s_(RequirePositiveDuration(duration_s, "duration_s")) {
  if (!ZeroTwist(target_)) {
    throw std::invalid_argument("a hold target must have zero twist");
  }
}

KinematicTargetSample HoldTrajectory::SampleKinematics(
    double elapsed_time_s) const {
  control::RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
  return KinematicTargetSample{target_, Eigen::Vector3d::Zero(),
                               Eigen::Vector3d::Zero()};
}

// --------------------------------------------------------------- waypoints

CartesianWaypointTrajectory::CartesianWaypointTrajectory(
    TargetFrame reference_frame, std::vector<CartesianWaypoint> waypoints)
    : reference_frame_(reference_frame), waypoints_(std::move(waypoints)) {
  if (waypoints_.size() < 2) {
    throw std::invalid_argument("a waypoint trajectory needs at least two poses");
  }
  if (waypoints_.front().time_s != 0.0) {
    throw std::invalid_argument("the first waypoint time must be 0.0 seconds");
  }
  for (const auto& waypoint : waypoints_) times_s_.push_back(waypoint.time_s);
  for (std::size_t index = 1; index < times_s_.size(); ++index) {
    if (times_s_[index] <= times_s_[index - 1]) {
      throw std::invalid_argument("waypoint times must be strictly increasing");
    }
  }
  for (std::size_t index = 0; index + 1 < waypoints_.size(); ++index) {
    rotation_vectors_.push_back(RotationVector(
        waypoints_[index].pose.rotation.transpose() *
        waypoints_[index + 1].pose.rotation));
  }

  // Globally minimum-integrated-jerk C2 spline: endpoint velocity and
  // acceleration are fixed to zero, interior values solved together rather
  // than guessed per segment (clause H1).
  const std::size_t count = waypoints_.size();
  std::vector<Eigen::Vector3d> derivatives(count, Eigen::Vector3d::Zero());
  std::vector<Eigen::Vector3d> accelerations(count, Eigen::Vector3d::Zero());

  if (count > 2) {
    const int interior = static_cast<int>(count) - 2;
    const int unknown_count = 2 * interior;
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(unknown_count, unknown_count);
    Eigen::MatrixXd gradients = Eigen::MatrixXd::Zero(3, unknown_count);
    Eigen::Matrix3d integral;
    integral << 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0,
        1.0 / 3.0, 1.0 / 4.0, 1.0 / 5.0;

    for (std::size_t segment = 0; segment + 1 < count; ++segment) {
      const double duration = times_s_[segment + 1] - times_s_[segment];

      // Linear map from the six boundary values (p0, p1, v0, a0, v1, a1) to
      // the three jerk-polynomial coefficients.
      using Scalar1 = Eigen::Matrix<double, 1, 1>;
      Eigen::Matrix<double, 3, 6> mapping;
      for (int basis = 0; basis < 6; ++basis) {
        std::array<Scalar1, 6> values{};
        for (auto& value : values) value.setZero();
        values[static_cast<std::size_t>(basis)](0) = 1.0;
        // Argument order matches the Python: p0, p1, v0, v1, a0, a1 read from
        // slots 0, 1, 2, 4, 3, 5 of the basis vector.
        const Eigen::Matrix<double, 1, 6> coefficients =
            QuinticCoefficients<1>(values[0], values[1], values[2], values[4],
                                   values[3], values[5], duration);
        mapping(0, basis) = 6.0 * coefficients(0, 3);
        mapping(1, basis) = 24.0 * coefficients(0, 4);
        mapping(2, basis) = 60.0 * coefficients(0, 5);
      }
      const Eigen::Matrix<double, 6, 6> cost =
          mapping.transpose() * integral * mapping /
          std::pow(duration, 5);

      Eigen::Matrix<double, 3, 6> constant = Eigen::Matrix<double, 3, 6>::Zero();
      constant.col(0) = waypoints_[segment].pose.position_m;
      constant.col(1) = waypoints_[segment + 1].pose.position_m;

      Eigen::MatrixXd local_map = Eigen::MatrixXd::Zero(6, unknown_count);
      if (segment > 0) {
        local_map(2, static_cast<int>(segment) - 1) = 1.0;
        local_map(3, interior + static_cast<int>(segment) - 1) = 1.0;
      }
      if (segment + 1 < count - 1) {
        local_map(4, static_cast<int>(segment)) = 1.0;
        local_map(5, interior + static_cast<int>(segment)) = 1.0;
      }
      hessian += local_map.transpose() * cost * local_map;
      gradients += constant * cost * local_map;
    }
    const Eigen::MatrixXd solved =
        linalg::SolveGeneral(hessian, Eigen::MatrixXd(-gradients.transpose()))
            .transpose();
    for (int index = 0; index < interior; ++index) {
      derivatives[static_cast<std::size_t>(index) + 1] = solved.col(index);
      accelerations[static_cast<std::size_t>(index) + 1] =
          solved.col(interior + index);
    }
  }

  for (std::size_t index = 0; index + 1 < count; ++index) {
    const double duration = times_s_[index + 1] - times_s_[index];
    translation_coefficients_.push_back(QuinticCoefficients<3>(
        waypoints_[index].pose.position_m, waypoints_[index + 1].pose.position_m,
        derivatives[index], derivatives[index + 1], accelerations[index],
        accelerations[index + 1], duration));
  }
}

CartesianWaypointTrajectory CartesianWaypointTrajectory::FromDurations(
    TargetFrame reference_frame, const std::vector<Pose>& poses,
    const std::vector<double>& durations_s) {
  if (durations_s.size() != poses.size() - 1) {
    throw std::invalid_argument(
        "durations_s must contain one duration between each pose");
  }
  std::vector<CartesianWaypoint> waypoints;
  double time = 0.0;
  waypoints.push_back({0.0, poses.front()});
  for (std::size_t index = 0; index < durations_s.size(); ++index) {
    time += RequirePositiveDuration(durations_s[index], "duration_s");
    waypoints.push_back({time, poses[index + 1]});
  }
  return CartesianWaypointTrajectory(reference_frame, std::move(waypoints));
}

KinematicTargetSample CartesianWaypointTrajectory::SampleKinematics(
    double elapsed_time_s) const {
  const double elapsed =
      control::RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
  if (elapsed <= 0.0) {
    return KinematicTargetSample{
        FramedTarget{reference_frame_, waypoints_.front().pose, Twist::Zero()},
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  }
  if (elapsed >= duration_s()) {
    return KinematicTargetSample{
        FramedTarget{reference_frame_, waypoints_.back().pose, Twist::Zero()},
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  }

  const std::size_t segment = static_cast<std::size_t>(
      std::upper_bound(times_s_.begin(), times_s_.end(), elapsed) -
      times_s_.begin() - 1);
  const CartesianWaypoint& left = waypoints_[segment];
  const CartesianWaypoint& right = waypoints_[segment + 1];
  const double duration = right.time_s - left.time_s;
  const double unit_time = (elapsed - left.time_s) / duration;

  double progress = 0.0, progress_rate = 0.0, progress_acceleration = 0.0;
  MinimumJerk(unit_time, duration, progress, progress_rate,
              progress_acceleration);

  const Eigen::Matrix<double, 3, 6>& coefficients =
      translation_coefficients_[segment];
  const Eigen::Vector3d position = EvaluatePolynomial(coefficients, unit_time, 0);
  const Eigen::Vector3d linear_velocity =
      EvaluatePolynomial(coefficients, unit_time, 1) / duration;
  const Eigen::Vector3d linear_acceleration =
      EvaluatePolynomial(coefficients, unit_time, 2) / (duration * duration);
  const Eigen::Vector3d& relative_vector = rotation_vectors_[segment];
  const Eigen::Vector3d world_axis = left.pose.rotation * relative_vector;

  FramedTarget target;
  target.reference_frame = reference_frame_;
  target.pose = Pose(CleanSmall(position),
                     left.pose.rotation *
                         RotationFromVector(progress * relative_vector));
  target.twist.linear_m_s = CleanSmall(linear_velocity);
  target.twist.angular_rad_s = CleanSmall(progress_rate * world_axis);
  return KinematicTargetSample{target, CleanSmall(linear_acceleration),
                               CleanSmall(progress_acceleration * world_axis)};
}

TrajectoryRateBounds CartesianWaypointTrajectory::MaximumRates() const {
  TrajectoryRateBounds bounds;
  for (std::size_t index = 0; index < translation_coefficients_.size(); ++index) {
    const double duration = times_s_[index + 1] - times_s_[index];
    const Eigen::Matrix<double, 3, 6>& coefficients =
        translation_coefficients_[index];

    Eigen::Matrix<double, 3, 6> velocity = Eigen::Matrix<double, 3, 6>::Zero();
    for (int power = 1; power < 6; ++power) {
      velocity.col(power - 1) =
          coefficients.col(power) * static_cast<double>(power) / duration;
    }
    Eigen::Matrix<double, 3, 6> acceleration =
        Eigen::Matrix<double, 3, 6>::Zero();
    for (int power = 1; power < 5; ++power) {
      acceleration.col(power - 1) = velocity.col(power) *
                                    static_cast<double>(power) / duration;
    }

    bounds.max_linear_speed_m_s =
        std::max(bounds.max_linear_speed_m_s, VectorPolynomialMaximum(velocity, 5));
    bounds.max_linear_acceleration_m_s2 = std::max(
        bounds.max_linear_acceleration_m_s2,
        VectorPolynomialMaximum(acceleration, 4));

    const double angle = rotation_vectors_[index].norm();
    bounds.max_angular_speed_rad_s =
        std::max(bounds.max_angular_speed_rad_s, 1.875 * angle / duration);
    bounds.max_angular_acceleration_rad_s2 =
        std::max(bounds.max_angular_acceleration_rad_s2,
                 10.0 / std::sqrt(3.0) * angle / (duration * duration));
  }
  return bounds;
}

// ------------------------------------------------------------------ circle

CircleTrajectory::CircleTrajectory(TargetFrame reference_frame, Pose start_pose,
                                   double radius_m, Eigen::Vector3d normal,
                                   Eigen::Vector3d start_direction,
                                   double duration_s, int revolutions,
                                   bool clockwise,
                                   std::optional<Eigen::Matrix3d> end_rotation)
    : reference_frame_(reference_frame),
      start_pose_(std::move(start_pose)),
      radius_m_(RequirePositiveDuration(radius_m, "radius_m")),
      duration_s_(RequirePositiveDuration(duration_s, "duration_s")),
      revolutions_(revolutions) {
  if (revolutions <= 0) {
    throw std::invalid_argument("revolutions must be a positive integer");
  }
  const double normal_norm = normal.norm();
  const double radial_norm = start_direction.norm();
  if (normal_norm < 1e-12 || radial_norm < 1e-12) {
    throw std::invalid_argument("circle directions must be non-zero");
  }
  normal_ = normal / normal_norm;
  radial_ = start_direction / radial_norm;
  if (std::abs(normal_.dot(radial_)) > 1e-9) {
    throw std::invalid_argument(
        "circle normal and start_direction must be orthogonal");
  }
  tangent_ = normal_.cross(radial_);
  if (clockwise) tangent_ *= -1.0;

  const Eigen::Matrix3d resolved_end =
      end_rotation ? *end_rotation : start_pose_.rotation;
  rotation_vector_ = RotationVector(start_pose_.rotation.transpose() * resolved_end);
}

KinematicTargetSample CircleTrajectory::SampleKinematics(
    double elapsed_time_s) const {
  const double elapsed =
      control::RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
  if (elapsed <= 0.0) {
    return KinematicTargetSample{
        FramedTarget{reference_frame_, start_pose_, Twist::Zero()},
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
  }
  const double clamped = std::min(elapsed, duration_s_);
  const double unit_time = clamped / duration_s_;
  double progress = 0.0, rate = 0.0, acceleration = 0.0;
  MinimumJerk(unit_time, duration_s_, progress, rate, acceleration);

  const double total_angle = 2.0 * M_PI * revolutions_;
  const double angle = total_angle * progress;
  const double angle_rate = total_angle * rate;
  const double angle_acceleration = total_angle * acceleration;

  const Eigen::Vector3d radial =
      std::cos(angle) * radial_ + std::sin(angle) * tangent_;
  const Eigen::Vector3d tangent =
      -std::sin(angle) * radial_ + std::cos(angle) * tangent_;

  const Eigen::Vector3d world_axis = start_pose_.rotation * rotation_vector_;
  FramedTarget target;
  target.reference_frame = reference_frame_;
  target.pose = Pose(centre_m() + radius_m_ * radial,
                     start_pose_.rotation *
                         RotationFromVector(progress * rotation_vector_));
  target.twist.linear_m_s = radius_m_ * angle_rate * tangent;
  target.twist.angular_rad_s = rate * world_axis;

  const Eigen::Vector3d linear_acceleration =
      radius_m_ * (angle_acceleration * tangent - angle_rate * angle_rate * radial);
  return KinematicTargetSample{target, linear_acceleration,
                               acceleration * world_axis};
}

TrajectoryRateBounds CircleTrajectory::MaximumRates() const {
  const double total_angle = 2.0 * M_PI * revolutions_;
  const double orientation_angle = rotation_vector_.norm();
  TrajectoryRateBounds bounds;
  bounds.max_linear_speed_m_s = radius_m_ * total_angle * 1.875 / duration_s_;
  bounds.max_linear_acceleration_m_s2 =
      radius_m_ *
      (total_angle * 10.0 / std::sqrt(3.0) + std::pow(total_angle * 1.875, 2)) /
      (duration_s_ * duration_s_);
  bounds.max_angular_speed_rad_s = 1.875 * orientation_angle / duration_s_;
  bounds.max_angular_acceleration_rad_s2 = 10.0 / std::sqrt(3.0) *
                                           orientation_angle /
                                           (duration_s_ * duration_s_);
  return bounds;
}

// ----------------------------------------------------------------- program

TargetProgram::TargetProgram(TargetFrame reference_frame,
                             std::vector<TargetProgramSegment> segments)
    : reference_frame_(reference_frame), segments_(std::move(segments)) {
  if (segments_.empty()) {
    throw std::invalid_argument("a target program needs at least one segment");
  }
  for (const auto& segment : segments_) {
    if (segment.source == nullptr) {
      throw std::invalid_argument("program segments must have a source");
    }
    if (segment.source->reference_frame() != reference_frame_) {
      throw std::invalid_argument(
          "all target program segments must use its declared frame");
    }
  }
  for (std::size_t index = 0; index + 1 < segments_.size(); ++index) {
    const KinematicTargetSample left =
        segments_[index].source->SampleKinematics(segments_[index].duration_s);
    const KinematicTargetSample right =
        segments_[index + 1].source->SampleKinematics(0.0);
    if (!SamplesContinuous(left, right)) {
      throw std::invalid_argument("target program boundary " +
                                  std::to_string(index) +
                                  " is not C2 continuous");
    }
  }
  const KinematicTargetSample final_sample =
      segments_.back().source->SampleKinematics(segments_.back().duration_s);
  if (!ZeroTwist(final_sample.target) ||
      !AllClose(final_sample.linear_acceleration_m_s2,
                Eigen::Vector3d::Zero(), kProgramBoundaryAtol) ||
      !AllClose(final_sample.angular_acceleration_rad_s2,
                Eigen::Vector3d::Zero(), kProgramBoundaryAtol)) {
    throw std::invalid_argument("the final program segment must end at rest");
  }

  double elapsed = 0.0;
  for (const auto& segment : segments_) {
    elapsed += segment.duration_s;
    end_times_s_.push_back(elapsed);
  }
}

KinematicTargetSample TargetProgram::SampleKinematics(
    double elapsed_time_s) const {
  const double elapsed =
      control::RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
  if (elapsed >= duration_s()) {
    return segments_.back().source->SampleKinematics(
        segments_.back().duration_s);
  }
  const std::size_t index = static_cast<std::size_t>(
      std::upper_bound(end_times_s_.begin(), end_times_s_.end(), elapsed) -
      end_times_s_.begin());
  const double start = index == 0 ? 0.0 : end_times_s_[index - 1];
  return segments_[index].source->SampleKinematics(elapsed - start);
}

TrajectoryRateBounds TargetProgram::MaximumRates() const {
  TrajectoryRateBounds bounds;
  for (const auto& segment : segments_) {
    const TrajectoryRateBounds item = segment.source->MaximumRates();
    bounds.max_linear_speed_m_s =
        std::max(bounds.max_linear_speed_m_s, item.max_linear_speed_m_s);
    bounds.max_linear_acceleration_m_s2 = std::max(
        bounds.max_linear_acceleration_m_s2, item.max_linear_acceleration_m_s2);
    bounds.max_angular_speed_rad_s =
        std::max(bounds.max_angular_speed_rad_s, item.max_angular_speed_rad_s);
    bounds.max_angular_acceleration_rad_s2 =
        std::max(bounds.max_angular_acceleration_rad_s2,
                 item.max_angular_acceleration_rad_s2);
  }
  return bounds;
}

// ---------------------------------------------------------------- periodic

PeriodicTargetSource::PeriodicTargetSource(
    std::shared_ptr<TimedTargetSource> source, double period_s)
    : source_(std::move(source)),
      period_s_(RequirePositiveDuration(period_s, "period_s")) {
  if (source_ == nullptr) {
    throw std::invalid_argument("source must not be null");
  }
  if (!SamplesContinuous(source_->SampleKinematics(0.0),
                         source_->SampleKinematics(period_s_))) {
    throw std::invalid_argument(
        "a periodic target source must close continuously in frame, pose, "
        "twist, and acceleration");
  }
}

KinematicTargetSample PeriodicTargetSource::SampleKinematics(
    double elapsed_time_s) const {
  const double elapsed =
      control::RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
  return source_->SampleKinematics(std::fmod(elapsed, period_s_));
}

// ------------------------------------------------------------------ timing

std::shared_ptr<CartesianWaypointTrajectory> TimedWaypointTrajectory(
    TargetFrame reference_frame, const std::vector<Pose>& poses,
    const std::optional<std::vector<double>>& durations_s,
    const TrajectoryLimits& limits) {
  if (poses.size() < 2) {
    throw std::invalid_argument("a waypoint path needs at least two poses");
  }
  std::vector<double> durations;
  bool explicit_timing = false;
  if (!durations_s) {
    for (std::size_t index = 0; index + 1 < poses.size(); ++index) {
      const double distance =
          (poses[index + 1].position_m - poses[index].position_m).norm();
      const double angle =
          RotationVector(poses[index].rotation.transpose() *
                         poses[index + 1].rotation)
              .norm();
      durations.push_back(MinimumDuration(distance, angle, limits));
    }
  } else {
    for (double value : *durations_s) {
      durations.push_back(RequirePositiveDuration(value, "duration_s"));
    }
    explicit_timing = true;
  }

  auto trajectory = std::make_shared<CartesianWaypointTrajectory>(
      CartesianWaypointTrajectory::FromDurations(reference_frame, poses,
                                                 durations));
  const double scale = limits.RequiredTimeScale(trajectory->MaximumRates());
  if (explicit_timing) {
    limits.Validate(trajectory->MaximumRates());
  } else if (scale > 1.0) {
    std::vector<double> scaled;
    for (double value : durations) scaled.push_back(scale * value);
    trajectory = std::make_shared<CartesianWaypointTrajectory>(
        CartesianWaypointTrajectory::FromDurations(reference_frame, poses,
                                                   scaled));
    limits.Validate(trajectory->MaximumRates());
  }
  return trajectory;
}

std::shared_ptr<CircleTrajectory> TimedCircleTrajectory(
    TargetFrame reference_frame, const Pose& start_pose, double radius_m,
    const Eigen::Vector3d& normal, const Eigen::Vector3d& start_direction,
    const std::optional<double>& duration_s, const TrajectoryLimits& limits,
    int revolutions, bool clockwise,
    const std::optional<Eigen::Matrix3d>& end_rotation) {
  if (!duration_s && !limits.max_linear_speed_m_s &&
      !limits.max_linear_acceleration_m_s2) {
    throw std::invalid_argument(
        "automatic circle timing needs a linear speed or acceleration limit");
  }
  const double candidate_duration = duration_s ? *duration_s : 1.0;
  auto trajectory = std::make_shared<CircleTrajectory>(
      reference_frame, start_pose, radius_m, normal, start_direction,
      candidate_duration, revolutions, clockwise, end_rotation);
  const double scale = limits.RequiredTimeScale(trajectory->MaximumRates());
  if (!duration_s) {
    trajectory = std::make_shared<CircleTrajectory>(
        reference_frame, start_pose, radius_m, normal, start_direction, scale,
        revolutions, clockwise, end_rotation);
  }
  limits.Validate(trajectory->MaximumRates());
  return trajectory;
}

}  // namespace srl::trajectory
