// Pure Cartesian target sources and timed waypoint trajectories.
//
// Fixed poses, explicit times, one declared frame, and a deterministic
// Sample(elapsed_time_s) boundary. No MuJoCo reads, marker writes, or user
// callbacks. One-to-one with controller/trajectory.py. Contract clauses H1-H10.

#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "control/TargetSource.h"
#include "core/Types.h"

namespace srl::trajectory {

using control::KinematicTargetSample;
using control::TargetSource;

inline constexpr double kSmallAngleRad = 1e-8;
inline constexpr double kNearPiRad = 1e-6;
inline constexpr double kProgramBoundaryAtol = 1e-10;
inline constexpr double kLimitTolerance = 1e-10;

// Principal SO(3) logarithm with norm in [0, pi]. This is trajectory.py's own
// implementation, deliberately NOT Pinocchio's log3 -- it has a distinct
// near-pi branch and a canonical-axis tie-break.
Eigen::Vector3d RotationVector(const Eigen::Matrix3d& rotation);

// SO(3) exponential for a rotation vector.
Eigen::Matrix3d RotationFromVector(const Eigen::Vector3d& rotation_vector);

// Quintic progress and its first two time derivatives.
void MinimumJerk(double unit_time, double duration_s, double& progress,
                 double& progress_rate, double& progress_acceleration);

// Zero out |value| < tolerance, as trajectory.py's _clean_small does.
Eigen::Vector3d CleanSmall(const Eigen::Vector3d& value,
                           double tolerance = 1e-14);

struct TrajectoryRateBounds {
  double max_linear_speed_m_s{0.0};
  double max_linear_acceleration_m_s2{0.0};
  double max_angular_speed_rad_s{0.0};
  double max_angular_acceleration_rad_s2{0.0};
};

// Optional Cartesian limits used for timing and strict validation.
struct TrajectoryLimits {
  std::optional<double> max_linear_speed_m_s;
  std::optional<double> max_linear_acceleration_m_s2;
  std::optional<double> max_angular_speed_rad_s;
  std::optional<double> max_angular_acceleration_rad_s2;

  double RequiredTimeScale(const TrajectoryRateBounds& bounds) const;
  void Validate(const TrajectoryRateBounds& bounds) const;
};

// Sources that can report their own extent and rate envelope.
class TimedTargetSource : public TargetSource {
 public:
  virtual double duration_s() const = 0;
  virtual std::vector<double> boundary_times_s() const = 0;
  virtual TrajectoryRateBounds MaximumRates() const = 0;
};

// Finite static segment with zero twist and acceleration.
class HoldTrajectory final : public TimedTargetSource {
 public:
  HoldTrajectory(FramedTarget target, double duration_s);

  TargetFrame reference_frame() const override {
    return target_.reference_frame;
  }
  KinematicTargetSample SampleKinematics(double elapsed_time_s) const override;
  double duration_s() const override { return duration_s_; }
  std::vector<double> boundary_times_s() const override {
    return {duration_s_};
  }
  TrajectoryRateBounds MaximumRates() const override { return {}; }

 private:
  FramedTarget target_;
  double duration_s_{0.0};
};

struct CartesianWaypoint {
  double time_s{0.0};
  Pose pose;
};

// C2 minimum-jerk translation and smooth SO(3) waypoint motion.
//
// Translation minimises integrated squared jerk over the complete path;
// endpoint linear velocity/acceleration are zero and interior values are
// solved globally and shared by adjacent quintics. Orientation reaches each
// exact waypoint using principal-log SO(3) quintics with zero angular velocity
// and acceleration at every knot.
class CartesianWaypointTrajectory final : public TimedTargetSource {
 public:
  CartesianWaypointTrajectory(TargetFrame reference_frame,
                              std::vector<CartesianWaypoint> waypoints);

  static CartesianWaypointTrajectory FromDurations(
      TargetFrame reference_frame, const std::vector<Pose>& poses,
      const std::vector<double>& durations_s);

  TargetFrame reference_frame() const override { return reference_frame_; }
  KinematicTargetSample SampleKinematics(double elapsed_time_s) const override;
  double duration_s() const override { return times_s_.back(); }
  std::vector<double> boundary_times_s() const override {
    return {times_s_.begin() + 1, times_s_.end()};
  }
  TrajectoryRateBounds MaximumRates() const override;

 private:
  TargetFrame reference_frame_{TargetFrame::World};
  std::vector<CartesianWaypoint> waypoints_;
  std::vector<double> times_s_;
  std::vector<Eigen::Vector3d> rotation_vectors_;
  // [segment][axis][power], ascending normalised-time powers.
  std::vector<Eigen::Matrix<double, 3, 6>> translation_coefficients_;
};

// One or more exact circular revolutions with smooth rest endpoints.
class CircleTrajectory final : public TimedTargetSource {
 public:
  CircleTrajectory(TargetFrame reference_frame, Pose start_pose, double radius_m,
                   Eigen::Vector3d normal, Eigen::Vector3d start_direction,
                   double duration_s, int revolutions, bool clockwise,
                   std::optional<Eigen::Matrix3d> end_rotation);

  TargetFrame reference_frame() const override { return reference_frame_; }
  KinematicTargetSample SampleKinematics(double elapsed_time_s) const override;
  double duration_s() const override { return duration_s_; }
  std::vector<double> boundary_times_s() const override {
    return {duration_s_};
  }
  TrajectoryRateBounds MaximumRates() const override;

  Eigen::Vector3d centre_m() const {
    return start_pose_.position_m - radius_m_ * radial_;
  }

 private:
  TargetFrame reference_frame_{TargetFrame::World};
  Pose start_pose_;
  double radius_m_{0.0};
  double duration_s_{0.0};
  int revolutions_{1};
  Eigen::Vector3d normal_{Eigen::Vector3d::UnitZ()};
  Eigen::Vector3d radial_{Eigen::Vector3d::UnitX()};
  Eigen::Vector3d tangent_{Eigen::Vector3d::UnitY()};
  Eigen::Vector3d rotation_vector_{Eigen::Vector3d::Zero()};
};

struct TargetProgramSegment {
  double duration_s{0.0};
  std::shared_ptr<TimedTargetSource> source;
};

// Sequence holds, waypoint trajectories, and future target shapes. All
// segments must declare the same frame and meet continuously in pose, twist
// and acceleration, and the last must end at rest.
class TargetProgram final : public TimedTargetSource {
 public:
  TargetProgram(TargetFrame reference_frame,
                std::vector<TargetProgramSegment> segments);

  TargetFrame reference_frame() const override { return reference_frame_; }
  KinematicTargetSample SampleKinematics(double elapsed_time_s) const override;
  double duration_s() const override { return end_times_s_.back(); }
  std::vector<double> boundary_times_s() const override { return end_times_s_; }
  TrajectoryRateBounds MaximumRates() const override;

 private:
  TargetFrame reference_frame_{TargetFrame::World};
  std::vector<TargetProgramSegment> segments_;
  std::vector<double> end_times_s_;
};

// Repeat one closed source with no pose or twist jump at the boundary.
class PeriodicTargetSource final : public TimedTargetSource {
 public:
  PeriodicTargetSource(std::shared_ptr<TimedTargetSource> source,
                       double period_s);

  TargetFrame reference_frame() const override {
    return source_->reference_frame();
  }
  KinematicTargetSample SampleKinematics(double elapsed_time_s) const override;
  double duration_s() const override { return period_s_; }
  std::vector<double> boundary_times_s() const override {
    return source_->boundary_times_s();
  }
  TrajectoryRateBounds MaximumRates() const override {
    return source_->MaximumRates();
  }

 private:
  std::shared_ptr<TimedTargetSource> source_;
  double period_s_{0.0};
};

// Build a waypoint spline, deriving timing when durations are omitted.
std::shared_ptr<CartesianWaypointTrajectory> TimedWaypointTrajectory(
    TargetFrame reference_frame, const std::vector<Pose>& poses,
    const std::optional<std::vector<double>>& durations_s,
    const TrajectoryLimits& limits);

// Build an exact circle, deriving duration when it is omitted.
std::shared_ptr<CircleTrajectory> TimedCircleTrajectory(
    TargetFrame reference_frame, const Pose& start_pose, double radius_m,
    const Eigen::Vector3d& normal, const Eigen::Vector3d& start_direction,
    const std::optional<double>& duration_s, const TrajectoryLimits& limits,
    int revolutions, bool clockwise,
    const std::optional<Eigen::Matrix3d>& end_rotation);

}  // namespace srl::trajectory
