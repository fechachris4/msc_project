// Equation 7: project the requested joint velocity into the joint-rate box and
// the safe half-spaces implied by the human envelope.
//
// For every active arm sphere the control-barrier inequality is
//     distance_jacobian . qdot >= -recovery_gain * signed_clearance
// Positive clearance permits bounded approach; penetration requires outward
// recovery. A result is used only after every constraint is independently
// re-checked; otherwise the filter returns a hold/catch-up command and marks a
// genuine safety stop. One-to-one with reactive_controller.constrain_velocity_for_human.
// Contract clauses E7-E12.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "config/RuntimeConfig.h"
#include "core/Types.h"

namespace srl::control {

struct SafetyVelocitySolve {
  Vector7 qdot_safe{Vector7::Zero()};
  double minimum_clearance_m{0.0};
  int active_constraint_count{0};
  int projection_iterations{0};
  double max_constraint_violation_m_s{0.0};
  bool human_adjusted{false};
  bool limit_adjusted{false};
  bool stopped{false};
  std::string reason;
  std::vector<std::string> limiting_points;
};

// Reusable fixed-structure OSQP workspace for one arm. The sparsity pattern
// never changes, so each cycle updates only numeric values.
class SafetyVelocityProjector {
 public:
  SafetyVelocityProjector(std::size_t max_human_constraints,
                          const config::HumanSafetyConfig& config);
  ~SafetyVelocityProjector();

  SafetyVelocityProjector(const SafetyVelocityProjector&) = delete;
  SafetyVelocityProjector& operator=(const SafetyVelocityProjector&) = delete;

  // Returns the candidate, whether one was produced, and the iteration count.
  bool Project(const Vector7& requested,
               const std::vector<Vector7>& constraint_rows,
               const std::vector<double>& constraint_bounds,
               const Vector7& lower, const Vector7& upper, Vector7& solution,
               int& iterations);

  // True when this build has a QP backend at all.
  static bool Available();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

SafetyVelocitySolve ConstrainVelocityForHuman(
    const Vector7& requested_velocity_rad_s, const Vector7& lower_velocity_rad_s,
    const Vector7& upper_velocity_rad_s, const ArmHumanSafetyState& safety_state,
    const config::HumanSafetyConfig& config,
    const Vector7& measured_velocity_rad_s, SafetyVelocityProjector* projector);

// Number of times the QP fallback was actually entered this process. The
// repair sweep usually resolves feasibility first, so this being zero is the
// evidence that omitting OSQP would not change behaviour for a given run.
long QpFallbackEntryCount();

}  // namespace srl::control
