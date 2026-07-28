// Stateful joint-velocity limiting and position-command integration.
// One-to-one with controller/position_actuation.py. Contract clauses F1-F5.

#pragma once

#include <array>

#include "core/Types.h"

namespace srl::control {

using JointFlags = std::array<bool, kJoints>;

struct PositionActuationLimits {
  Vector7 velocity_rad_s{Vector7::Zero()};
  double lead_rad{0.0};
  Vector7 lower_position_rad{Vector7::Zero()};
  Vector7 upper_position_rad{Vector7::Zero()};

  void Validate() const;
};

struct PositionActuation {
  Vector7 qdot_speed_clipped{Vector7::Zero()};
  Vector7 qdot_effective{Vector7::Zero()};
  Vector7 command_before_rad{Vector7::Zero()};
  Vector7 command_after_rad{Vector7::Zero()};
  JointFlags speed_saturated{};
  JointFlags lead_clamped{};
  JointFlags range_clamped{};
};

// Persistent commanded position; reset only by reconstruction (clause F1).
class PositionIntegrator {
 public:
  PositionIntegrator(const Vector7& measured_position_rad,
                     const PositionActuationLimits& limits);

  const Vector7& command_rad() const { return command_rad_; }

  // Exact velocity interval that avoids every downstream clamp (clause F4).
  void VelocityBounds(const Vector7& measured_position_rad, double dt_s,
                      Vector7& lower, Vector7& upper) const;

  PositionActuation Step(const Vector7& measured_position_rad,
                         const Vector7& requested_velocity_rad_s, double dt_s);

 private:
  PositionActuationLimits limits_;
  Vector7 command_rad_;
};

}  // namespace srl::control
