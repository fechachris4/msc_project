// Scripted EE-target motion: sinusoidal target trajectory per side, world
// frame -- the companion lever to Motion.h's base disturbance, used to
// characterise the reactive loop's own tracking bandwidth with the base held
// static. Off by default (zero amplitudes = no write).
//
// The sinusoid is added on top of the home pose captured by CaptureHome() and
// never re-resolved through the (possibly moving) torso, keeping this signal
// independent of base motion. One-to-one with sim/target_motion.py.

#pragma once

#include "core/Types.h"
#include "sim/MujocoBackend.h"

namespace srl::sim {

class TargetMotion {
 public:
  // Capture each side's current world-frame target pose as home. Call once,
  // after ApplyDesiredPos().
  void CaptureHome(const MujocoBackend& backend);

  // (position, rotation) of the scripted EE target at time t, world frame.
  // Composed as R_home * R_rpy(delta) -- matrix composition, not Motion.h's
  // rpy-sum shortcut, because this home orientation need not be identity.
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> PoseAt(
      double t, Side side, const Eigen::Vector3d& linear_amplitude,
      double linear_frequency, const Eigen::Vector3d& rotational_amplitude,
      double rotational_frequency) const;

  // Write the scripted EE-target pose into the target mocap body. All-zero
  // amplitudes: no write at all.
  void SetTargetPose(MujocoBackend& backend, double t, Side side,
                     const Eigen::Vector3d& linear_amplitude,
                     double linear_frequency,
                     const Eigen::Vector3d& rotational_amplitude,
                     double rotational_frequency) const;

 private:
  DualArm<Eigen::Vector3d> home_position_{};
  DualArm<Eigen::Matrix3d> home_rotation_{};
  bool captured_{false};
};

}  // namespace srl::sim
