#include "control/SafetyFilter.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "kinematics/LinkSpheres.h"

#if SRL_HAVE_OSQP
#include <osqp.h>
#endif

namespace srl::control {
namespace {

std::atomic<long> g_qp_fallback_entries{0};

double Clamp(double value, double lower, double upper) {
  return value < lower ? lower : (value > upper ? upper : value);
}

Vector7 ClipVector(const Vector7& value, const Vector7& lower,
                   const Vector7& upper) {
  Vector7 result;
  for (int index = 0; index < kJoints; ++index) {
    result(index) = Clamp(value(index), lower(index), upper(index));
  }
  return result;
}

// _maximum_constraint_violation
double MaximumConstraintViolation(const Vector7& value,
                                  const std::vector<Vector7>& rows,
                                  const std::vector<double>& bounds,
                                  const Vector7& lower, const Vector7& upper) {
  double worst = (lower - value).maxCoeff();
  worst = std::max(worst, (value - upper).maxCoeff());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    worst = std::max(worst, bounds[index] - rows[index].dot(value));
  }
  return std::max(0.0, worst);
}

// _repair_constraint_feasibility: Gauss-Seidel projection onto each violated
// half-space, re-clipped to the box after every sweep. Never bypasses the
// final verification.
int RepairConstraintFeasibility(Vector7& value, const std::vector<Vector7>& rows,
                                const std::vector<double>& bounds,
                                const Vector7& lower, const Vector7& upper,
                                double tolerance, int max_iterations) {
  value = ClipVector(value, lower, upper);
  std::vector<double> row_norm_squared(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    row_norm_squared[index] = rows[index].squaredNorm();
  }
  const double epsilon = std::numeric_limits<double>::epsilon();

  int completed = 0;
  for (int sweep = 1; sweep <= max_iterations; ++sweep) {
    completed = sweep;
    for (std::size_t index = 0; index < rows.size(); ++index) {
      const double deficit = bounds[index] - rows[index].dot(value);
      if (deficit > tolerance && row_norm_squared[index] > epsilon) {
        value += (deficit / row_norm_squared[index]) * rows[index];
      }
    }
    value = ClipVector(value, lower, upper);
    if (MaximumConstraintViolation(value, rows, bounds, lower, upper) <=
        tolerance) {
      break;
    }
  }
  return completed;
}

bool AllClose(const Vector7& left, const Vector7& right, double atol) {
  // np.allclose(rtol=0.0, atol=tolerance)
  return ((left - right).array().abs() <= atol).all();
}

}  // namespace

long QpFallbackEntryCount() { return g_qp_fallback_entries.load(); }

// ------------------------------------------------------------------- OSQP

#if SRL_HAVE_OSQP

struct SafetyVelocityProjector::Impl {
  std::size_t max_constraints{0};
  OSQPSolver* solver{nullptr};

  // Fixed sparsity: dense human rows plus one identity row per joint.
  std::vector<OSQPInt> row_indices;
  std::vector<OSQPInt> column_starts;
  std::vector<OSQPFloat> values;

  std::vector<OSQPFloat> lower_bounds;
  std::vector<OSQPFloat> upper_bounds;
  std::vector<OSQPFloat> linear_cost;

  // P = I(7)
  std::vector<OSQPInt> p_row_indices;
  std::vector<OSQPInt> p_column_starts;
  std::vector<OSQPFloat> p_values;

  ~Impl() {
    if (solver != nullptr) osqp_cleanup(solver);
  }
};

SafetyVelocityProjector::SafetyVelocityProjector(
    std::size_t max_human_constraints, const config::HumanSafetyConfig& config)
    : impl_(std::make_unique<Impl>()) {
  if (max_human_constraints == 0) {
    throw std::invalid_argument("max_human_constraints must be positive");
  }
  impl_->max_constraints = max_human_constraints;
  const OSQPInt count = static_cast<OSQPInt>(max_human_constraints);
  const OSQPInt rows = count + kJoints;

  impl_->column_starts.push_back(0);
  for (OSQPInt joint = 0; joint < kJoints; ++joint) {
    for (OSQPInt row = 0; row < count; ++row) {
      impl_->row_indices.push_back(row);
      impl_->values.push_back(0.0);
    }
    impl_->row_indices.push_back(count + joint);
    impl_->values.push_back(1.0);
    impl_->column_starts.push_back(
        static_cast<OSQPInt>(impl_->row_indices.size()));
  }

  for (OSQPInt joint = 0; joint < kJoints; ++joint) {
    impl_->p_column_starts.push_back(joint);
    impl_->p_row_indices.push_back(joint);
    impl_->p_values.push_back(1.0);
  }
  impl_->p_column_starts.push_back(kJoints);

  impl_->lower_bounds.assign(static_cast<std::size_t>(rows), -OSQP_INFTY);
  impl_->upper_bounds.assign(static_cast<std::size_t>(rows), OSQP_INFTY);
  impl_->linear_cost.assign(kJoints, 0.0);

  OSQPCscMatrix a_matrix{};
  a_matrix.m = rows;
  a_matrix.n = kJoints;
  a_matrix.nz = -1;
  a_matrix.nzmax = static_cast<OSQPInt>(impl_->values.size());
  a_matrix.p = impl_->column_starts.data();
  a_matrix.i = impl_->row_indices.data();
  a_matrix.x = impl_->values.data();

  OSQPCscMatrix p_matrix{};
  p_matrix.m = kJoints;
  p_matrix.n = kJoints;
  p_matrix.nz = -1;
  p_matrix.nzmax = kJoints;
  p_matrix.p = impl_->p_column_starts.data();
  p_matrix.i = impl_->p_row_indices.data();
  p_matrix.x = impl_->p_values.data();

  OSQPSettings* settings = OSQPSettings_new();
  settings->verbose = 0;
  settings->eps_abs = config.constraint_tolerance_m_s;
  settings->eps_rel = 0.0;
  settings->max_iter = config.projection_iterations;
  settings->polishing = 0;
  settings->warm_starting = 1;
  settings->adaptive_rho = 1;
  settings->rho = 1.0;
  settings->check_termination = 5;

  const OSQPInt status =
      osqp_setup(&impl_->solver, &p_matrix, impl_->linear_cost.data(),
                 &a_matrix, impl_->lower_bounds.data(),
                 impl_->upper_bounds.data(), rows, kJoints, settings);
  OSQPSettings_free(settings);
  if (status != 0) {
    throw std::runtime_error("osqp_setup failed with status " +
                             std::to_string(status));
  }
}

SafetyVelocityProjector::~SafetyVelocityProjector() = default;

bool SafetyVelocityProjector::Available() { return true; }

bool SafetyVelocityProjector::Project(const Vector7& requested,
                                      const std::vector<Vector7>& constraint_rows,
                                      const std::vector<double>& constraint_bounds,
                                      const Vector7& lower, const Vector7& upper,
                                      Vector7& solution, int& iterations) {
  const std::size_t count = constraint_rows.size();
  if (count > impl_->max_constraints) {
    throw std::invalid_argument("human constraint count exceeds fixed capacity");
  }
  const std::size_t capacity = impl_->max_constraints;

  // Rebuild the numeric values in the fixed pattern: padded rows then 1.0.
  std::size_t cursor = 0;
  for (int joint = 0; joint < kJoints; ++joint) {
    for (std::size_t row = 0; row < capacity; ++row) {
      impl_->values[cursor++] =
          row < count ? constraint_rows[row](joint) : 0.0;
    }
    impl_->values[cursor++] = 1.0;
  }
  for (std::size_t row = 0; row < capacity; ++row) {
    impl_->lower_bounds[row] =
        row < count ? constraint_bounds[row] : -OSQP_INFTY;
    impl_->upper_bounds[row] = OSQP_INFTY;
  }
  for (int joint = 0; joint < kJoints; ++joint) {
    impl_->lower_bounds[capacity + static_cast<std::size_t>(joint)] =
        lower(joint);
    impl_->upper_bounds[capacity + static_cast<std::size_t>(joint)] =
        upper(joint);
    impl_->linear_cost[static_cast<std::size_t>(joint)] = -requested(joint);
  }

  osqp_update_data_mat(impl_->solver, nullptr, nullptr, 0,
                       impl_->values.data(), nullptr,
                       static_cast<OSQPInt>(impl_->values.size()));
  osqp_update_data_vec(impl_->solver, impl_->linear_cost.data(),
                       impl_->lower_bounds.data(), impl_->upper_bounds.data());
  osqp_solve(impl_->solver);

  iterations = static_cast<int>(impl_->solver->info->iter);
  const OSQPFloat* x = impl_->solver->solution != nullptr
                           ? impl_->solver->solution->x
                           : nullptr;
  bool available = x != nullptr;
  if (available) {
    for (int joint = 0; joint < kJoints; ++joint) {
      if (!std::isfinite(x[joint])) {
        available = false;
        break;
      }
    }
  }
  if (available) {
    for (int joint = 0; joint < kJoints; ++joint) solution(joint) = x[joint];
  } else {
    solution.setZero();
  }
  return available;
}

#else  // !SRL_HAVE_OSQP

struct SafetyVelocityProjector::Impl {};

SafetyVelocityProjector::SafetyVelocityProjector(
    std::size_t max_human_constraints, const config::HumanSafetyConfig&)
    : impl_(std::make_unique<Impl>()) {
  if (max_human_constraints == 0) {
    throw std::invalid_argument("max_human_constraints must be positive");
  }
}

SafetyVelocityProjector::~SafetyVelocityProjector() = default;

bool SafetyVelocityProjector::Available() { return false; }

bool SafetyVelocityProjector::Project(const Vector7&,
                                      const std::vector<Vector7>&,
                                      const std::vector<double>&, const Vector7&,
                                      const Vector7&, Vector7& solution,
                                      int& iterations) {
  // No QP backend: report "no candidate", which drives the caller to the same
  // fail-safe hold the Python takes when OSQP cannot produce a usable answer.
  solution.setZero();
  iterations = 0;
  return false;
}

#endif

// ---------------------------------------------------------------- equation 7

SafetyVelocitySolve ConstrainVelocityForHuman(
    const Vector7& requested_velocity_rad_s, const Vector7& lower_velocity_rad_s,
    const Vector7& upper_velocity_rad_s, const ArmHumanSafetyState& safety_state,
    const config::HumanSafetyConfig& config,
    const Vector7& measured_velocity_rad_s,
    SafetyVelocityProjector* projector) {
  const Vector7& requested = requested_velocity_rad_s;
  const Vector7& lower = lower_velocity_rad_s;
  const Vector7& upper = upper_velocity_rad_s;
  if ((lower.array() > upper.array()).any()) {
    throw std::invalid_argument(
        "lower_velocity_rad_s must not exceed upper_velocity_rad_s");
  }

  SafetyVelocitySolve result;
  result.minimum_clearance_m = safety_state.minimum_clearance_m();

  const Vector7 bounded_request = ClipVector(requested, lower, upper);

  // (1) Disabled: the request passes through UNCLIPPED -- joint-limit clipping
  // is skipped entirely here (clause E9).
  if (!config.enabled) {
    result.qdot_safe = requested;
    result.reason = "disabled";
    return result;
  }

  const bool limit_adjusted = !(requested.array() == bounded_request.array()).all();

  // (2) Nothing active.
  std::vector<std::size_t> active_indices;
  if (safety_state.evaluated) {
    for (std::size_t index = 0; index < safety_state.constraints.size();
         ++index) {
      if (safety_state.constraints.active[index]) active_indices.push_back(index);
    }
  }
  if (active_indices.empty()) {
    result.qdot_safe = bounded_request;
    result.limit_adjusted = limit_adjusted;
    result.reason = "clear";
    return result;
  }

  std::vector<Vector7> rows;
  std::vector<double> active_clearance;
  rows.reserve(active_indices.size());
  active_clearance.reserve(active_indices.size());
  for (std::size_t index : active_indices) {
    rows.push_back(safety_state.constraints.distance_jacobian_m_rad[index]);
    active_clearance.push_back(safety_state.constraints.signed_clearance_m[index]);
  }

  std::vector<double> bounds(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    const double measured_distance_rate = rows[index].dot(measured_velocity_rad_s);
    bounds[index] = -config.recovery_gain_s_inv *
                        (active_clearance[index] - config.control_margin_m) -
                    config.approach_velocity_damping *
                        std::min(measured_distance_rate, 0.0);
  }

  const double tolerance = config.constraint_tolerance_m_s;
  const auto point_name = [&](std::size_t local) {
    return std::string(kinematics::kLinkSpheres[active_indices[local]].name);
  };

  // (3) Already feasible.
  double smallest_slack = std::numeric_limits<double>::infinity();
  std::vector<double> bounded_slack(rows.size());
  for (std::size_t index = 0; index < rows.size(); ++index) {
    bounded_slack[index] = rows[index].dot(bounded_request) - bounds[index];
    smallest_slack = std::min(smallest_slack, bounded_slack[index]);
  }
  if (smallest_slack >= -tolerance) {
    result.qdot_safe = bounded_request;
    result.active_constraint_count = static_cast<int>(active_indices.size());
    result.limit_adjusted = limit_adjusted;
    result.reason = limit_adjusted ? "joint_limit_filtered" : "clear";
    for (std::size_t index = 0; index < rows.size(); ++index) {
      if (bounded_slack[index] <= 10.0 * tolerance) {
        result.limiting_points.push_back(point_name(index));
      }
    }
    return result;
  }

  // (4) Repair sweep.
  Vector7 candidate = bounded_request;
  int iterations = RepairConstraintFeasibility(candidate, rows, bounds, lower,
                                               upper, tolerance, 4);
  double violation =
      MaximumConstraintViolation(candidate, rows, bounds, lower, upper);
  bool candidate_available = violation <= tolerance;

  // (5) QP fallback, then a longer repair sweep.
  if (!candidate_available) {
    g_qp_fallback_entries.fetch_add(1);
    std::vector<Vector7> solver_rows(rows.size());
    std::vector<double> solver_bounds(rows.size());
    for (std::size_t index = 0; index < rows.size(); ++index) {
      const double row_scale = std::max(rows[index].norm(), 1e-12);
      solver_rows[index] = rows[index] / row_scale;
      solver_bounds[index] = bounds[index] / row_scale;
    }
    int solver_iterations = 0;
    if (projector != nullptr) {
      candidate_available =
          projector->Project(requested, solver_rows, solver_bounds, lower,
                             upper, candidate, solver_iterations);
    } else {
      candidate.setZero();
      candidate_available = false;
    }
    iterations += solver_iterations;
    violation = MaximumConstraintViolation(candidate, rows, bounds, lower, upper);
    if (candidate_available && violation > tolerance) {
      iterations += RepairConstraintFeasibility(candidate, rows, bounds, lower,
                                                upper, tolerance, 16);
      violation =
          MaximumConstraintViolation(candidate, rows, bounds, lower, upper);
    }
  }

  const bool feasible = candidate_available && violation <= tolerance;
  result.stopped = !feasible;
  if (result.stopped) {
    // Holding joint position is the fail-safe. If the persistent command must
    // catch up to its bounds, use only that required velocity and still report
    // the stop (clause E10).
    candidate = ClipVector(Vector7::Zero(), lower, upper);
    violation = MaximumConstraintViolation(candidate, rows, bounds, lower, upper);
  }

  result.qdot_safe = candidate;
  result.active_constraint_count = static_cast<int>(active_indices.size());
  result.projection_iterations = iterations;
  result.max_constraint_violation_m_s = violation;
  result.human_adjusted = !AllClose(candidate, bounded_request, tolerance);
  result.limit_adjusted = limit_adjusted;
  for (std::size_t index = 0; index < rows.size(); ++index) {
    if (rows[index].dot(candidate) - bounds[index] <= 10.0 * tolerance) {
      result.limiting_points.push_back(point_name(index));
    }
  }
  if (result.stopped) {
    const bool penetrating =
        std::any_of(active_clearance.begin(), active_clearance.end(),
                    [](double value) { return value < 0.0; });
    result.reason = penetrating ? "unsafe_initial_state_hold"
                                : "constraint_projection_failed_hold";
  } else if (result.human_adjusted) {
    result.reason = "filtered";
  } else if (result.limit_adjusted) {
    result.reason = "joint_limit_filtered";
  } else {
    result.reason = "clear";
  }
  return result;
}

}  // namespace srl::control
