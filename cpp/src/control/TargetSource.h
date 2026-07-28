// Pure Cartesian target sources sampled in trajectory-local elapsed time.
//
// No MuJoCo reads, marker writes, or user callbacks here. One-to-one with the
// source protocols in controller/trajectory.py.

#pragma once

#include <memory>
#include <stdexcept>

#include "core/Types.h"

namespace srl::control {

// Target plus acceleration, used to validate C2 program boundaries.
struct KinematicTargetSample {
  FramedTarget target;
  Eigen::Vector3d linear_acceleration_m_s2{Eigen::Vector3d::Zero()};
  Eigen::Vector3d angular_acceleration_rad_s2{Eigen::Vector3d::Zero()};
};

// One-arm Cartesian source.
class TargetSource {
 public:
  virtual ~TargetSource() = default;
  virtual TargetFrame reference_frame() const = 0;
  virtual KinematicTargetSample SampleKinematics(double elapsed_time_s) const = 0;

  FramedTarget Sample(double elapsed_time_s) const {
    return SampleKinematics(elapsed_time_s).target;
  }
};

// Two-arm source sampled exactly once at the Runner boundary.
class DualArmTargetSource {
 public:
  virtual ~DualArmTargetSource() = default;
  virtual DualArmFramedTargets Sample(double elapsed_time_s) = 0;
};

inline double RequireNonNegativeTime(double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0) {
    throw std::invalid_argument(std::string(name) +
                                " must be finite and non-negative");
  }
  return value;
}

// Compatibility adapter for retained DualArmFramedTargets.
class StaticDualArmTargetSource final : public DualArmTargetSource {
 public:
  explicit StaticDualArmTargetSource(DualArmFramedTargets targets);

  DualArmFramedTargets Sample(double elapsed_time_s) override {
    RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
    return targets_;
  }

 private:
  DualArmFramedTargets targets_;
};

// Adapter that presents one retained target through the source boundary.
class StaticTargetSource final : public TargetSource {
 public:
  explicit StaticTargetSource(FramedTarget target);

  TargetFrame reference_frame() const override {
    return target_.reference_frame;
  }
  KinematicTargetSample SampleKinematics(double elapsed_time_s) const override {
    RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
    return KinematicTargetSample{target_, Eigen::Vector3d::Zero(),
                                 Eigen::Vector3d::Zero()};
  }

 private:
  FramedTarget target_;
};

// Compose independently framed right/left Cartesian sources.
class IndependentArmTargetSource final : public DualArmTargetSource {
 public:
  IndependentArmTargetSource(std::shared_ptr<TargetSource> right,
                             std::shared_ptr<TargetSource> left)
      : right_(std::move(right)), left_(std::move(left)) {}

  DualArmFramedTargets Sample(double elapsed_time_s) override {
    const double elapsed =
        RequireNonNegativeTime(elapsed_time_s, "elapsed_time_s");
    DualArmFramedTargets targets;
    targets.right = right_->Sample(elapsed);
    targets.left = left_->Sample(elapsed);
    return targets;
  }

 private:
  std::shared_ptr<TargetSource> right_;
  std::shared_ptr<TargetSource> left_;
};

}  // namespace srl::control
