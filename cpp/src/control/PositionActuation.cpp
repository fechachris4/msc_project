#include "control/PositionActuation.h"

#include <cmath>
#include <stdexcept>

namespace srl::control {
namespace {

void RequireFinite(const Vector7& value, const char* name) {
  if (!value.allFinite()) {
    throw std::invalid_argument(std::string(name) +
                                " must be a finite shape-(7,) array");
  }
}

double Clamp(double value, double lower, double upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

}  // namespace

void PositionActuationLimits::Validate() const {
  RequireFinite(velocity_rad_s, "velocity_rad_s");
  if ((velocity_rad_s.array() <= 0.0).any()) {
    throw std::invalid_argument("velocity_rad_s must be positive");
  }
  if (!std::isfinite(lead_rad) || lead_rad <= 0.0) {
    throw std::invalid_argument("lead_rad must be finite and positive");
  }
  // NaN is rejected; +/-inf bounds are legal (an unlimited actuator).
  if (lower_position_rad.hasNaN() || upper_position_rad.hasNaN()) {
    throw std::invalid_argument("position bounds must not contain NaN");
  }
  if ((lower_position_rad.array() > upper_position_rad.array()).any()) {
    throw std::invalid_argument("lower_position_rad must not exceed upper");
  }
}

PositionIntegrator::PositionIntegrator(const Vector7& measured_position_rad,
                                       const PositionActuationLimits& limits)
    : limits_(limits), command_rad_(measured_position_rad) {
  limits_.Validate();
  RequireFinite(measured_position_rad, "measured_position_rad");
}

void PositionIntegrator::VelocityBounds(const Vector7& measured_position_rad,
                                        double dt_s, Vector7& lower,
                                        Vector7& upper) const {
  RequireFinite(measured_position_rad, "measured_position_rad");
  if (!std::isfinite(dt_s) || dt_s <= 0.0) {
    throw std::invalid_argument("dt_s must be finite and positive");
  }
  for (int index = 0; index < kJoints; ++index) {
    const double command_lower =
        std::max(measured_position_rad(index) - limits_.lead_rad,
                 limits_.lower_position_rad(index));
    const double command_upper =
        std::min(measured_position_rad(index) + limits_.lead_rad,
                 limits_.upper_position_rad(index));
    lower(index) = std::max(-limits_.velocity_rad_s(index),
                            (command_lower - command_rad_(index)) / dt_s);
    upper(index) = std::min(limits_.velocity_rad_s(index),
                            (command_upper - command_rad_(index)) / dt_s);
  }
  if ((lower.array() > upper.array()).any()) {
    throw std::runtime_error(
        "persistent command has no clamp-free velocity interval");
  }
}

PositionActuation PositionIntegrator::Step(const Vector7& measured_position_rad,
                                           const Vector7& requested_velocity_rad_s,
                                           double dt_s) {
  RequireFinite(measured_position_rad, "measured_position_rad");
  RequireFinite(requested_velocity_rad_s, "requested_velocity_rad_s");
  if (!std::isfinite(dt_s) || dt_s <= 0.0) {
    throw std::invalid_argument("dt_s must be finite and positive");
  }

  PositionActuation result;
  result.command_before_rad = command_rad_;
  for (int index = 0; index < kJoints; ++index) {
    const double limit = limits_.velocity_rad_s(index);
    const double clipped =
        Clamp(requested_velocity_rad_s(index), -limit, limit);
    result.qdot_speed_clipped(index) = clipped;
    // Exact inequality, matching NumPy's `requested != qdot_speed_clipped`.
    result.speed_saturated[index] = requested_velocity_rad_s(index) != clipped;

    const double integrated = command_rad_(index) + clipped * dt_s;
    const double lead_limited =
        Clamp(integrated, measured_position_rad(index) - limits_.lead_rad,
              measured_position_rad(index) + limits_.lead_rad);
    result.lead_clamped[index] = integrated != lead_limited;

    const double command_after =
        Clamp(lead_limited, limits_.lower_position_rad(index),
              limits_.upper_position_rad(index));
    result.range_clamped[index] = lead_limited != command_after;

    result.command_after_rad(index) = command_after;
    result.qdot_effective(index) =
        (command_after - result.command_before_rad(index)) / dt_s;
  }
  command_rad_ = result.command_after_rad;
  return result;
}

}  // namespace srl::control
