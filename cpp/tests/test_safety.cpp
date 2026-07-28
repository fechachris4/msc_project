// Human-envelope geometry and the velocity-projection ladder.
// Contract clauses E1-E12.

#include "TestSupport.h"

#include "control/HumanSafety.h"
#include "control/SafetyFilter.h"
#include "kinematics/LinkSpheres.h"

using namespace srl;
using namespace srl::control;

namespace {

config::HumanSafetyConfig MakeConfig() {
  config::HumanSafetyConfig config;
  config.enabled = true;
  config.center_xy_torso_m = {0.0, 0.0};
  config.radius_m = 0.25;
  config.z_min_torso_m = -1.10;
  config.z_max_torso_m = 0.70;
  config.clearance_m = 0.02;
  config.control_margin_m = 0.02;
  config.activation_distance_m = 0.10;
  config.recovery_gain_s_inv = 4.0;
  config.approach_velocity_damping = 0.0;
  config.projection_iterations = 50;
  config.constraint_tolerance_m_s = 1e-6;
  return config;
}

void RadialDistanceOutsideTheBand() {
  // Clause E2: purely radial exterior point.
  const config::HumanSafetyConfig config = MakeConfig();
  double distance = 0.0;
  Eigen::Vector3d gradient;
  FiniteCylinderDistanceGradient(Eigen::Vector3d(0.60, 0.0, 0.0), config,
                                 distance, gradient);
  CHECK_CLOSE(distance, 0.35, 1e-15, "radial signed distance");
  CHECK_MATRIX(gradient, Eigen::Vector3d(1.0, 0.0, 0.0), 1e-15,
               "radial gradient points outward");
}

void VerticalDistanceAboveTheCap() {
  const config::HumanSafetyConfig config = MakeConfig();
  double distance = 0.0;
  Eigen::Vector3d gradient;
  FiniteCylinderDistanceGradient(Eigen::Vector3d(0.0, 0.0, 1.20), config,
                                 distance, gradient);
  CHECK_CLOSE(distance, 0.50, 1e-15, "vertical signed distance above the cap");
  CHECK_MATRIX(gradient, Eigen::Vector3d(0.0, 0.0, 1.0), 1e-15,
               "vertical gradient points up");
}

void ExteriorCornerUsesTheHypotenuse() {
  // Clause E2: both excesses positive -> hypotenuse, blended normal.
  const config::HumanSafetyConfig config = MakeConfig();
  double distance = 0.0;
  Eigen::Vector3d gradient;
  // radial excess 0.15, vertical excess 0.20
  FiniteCylinderDistanceGradient(Eigen::Vector3d(0.40, 0.0, 0.90), config,
                                 distance, gradient);
  CHECK_CLOSE(distance, std::hypot(0.15, 0.20), 1e-15, "corner distance");
  CHECK_CLOSE(gradient.norm(), 1.0, 1e-15, "corner gradient is a unit normal");
  CHECK_CLOSE(gradient(0), 0.15 / distance, 1e-15, "corner gradient radial part");
  CHECK_CLOSE(gradient(2), 0.20 / distance, 1e-15,
              "corner gradient vertical part");
}

void OnAxisDegeneracyUsesXGradient() {
  // Clause E3.
  const config::HumanSafetyConfig config = MakeConfig();
  double distance = 0.0;
  Eigen::Vector3d gradient;
  FiniteCylinderDistanceGradient(Eigen::Vector3d(0.0, 0.0, 0.0), config,
                                 distance, gradient);
  CHECK_CLOSE(distance, -0.25, 1e-15, "on-axis point is inside by the radius");
  CHECK_MATRIX(gradient, Eigen::Vector3d(1.0, 0.0, 0.0), 1e-15,
               "on-axis gradient falls back to +x");
}

ArmControllerState MakeArmState(double x_offset) {
  ArmControllerState state;
  state.link_safety_points.position_world_m.assign(
      kinematics::kLinkSpheres.size(), Eigen::Vector3d(x_offset, 0.0, 0.0));
  state.link_safety_points.jacobian_world_m_rad.assign(
      kinematics::kLinkSpheres.size(), Matrix3x7::Zero());
  // Give each sphere a simple, non-degenerate radial sensitivity.
  for (auto& jacobian : state.link_safety_points.jacobian_world_m_rad) {
    jacobian(0, 0) = 1.0;
  }
  return state;
}

void ActivationMaskRespectsMountExemption() {
  // Clause E6: mount-exempt spheres are never active, whatever the clearance.
  const config::HumanSafetyConfig config = MakeConfig();
  Pose torso;  // identity at the origin
  const ArmHumanSafetyState state =
      EvaluateArm(torso, MakeArmState(0.30), config);

  CHECK_TRUE(state.evaluated, "arm safety state evaluated");
  for (std::size_t index = 0; index < state.constraints.size(); ++index) {
    const bool exempt = kinematics::kLinkSpheres[index].mount_exempt;
    CHECK_TRUE(exempt == static_cast<bool>(state.constraints.mount_exempt[index]),
               "mount exemption carried onto the constraint record");
    if (exempt) {
      CHECK_TRUE(!state.constraints.active[index],
                 "mount-exempt sphere never active");
    }
  }
  // Clause E4: clearance subtracts the sphere radius and the configured margin.
  const double expected = (0.30 - config.radius_m) -
                          kinematics::kLinkSpheres[0].radius_m -
                          config.clearance_m;
  CHECK_CLOSE(state.constraints.signed_clearance_m[0], expected, 1e-15,
              "signed clearance subtracts sphere radius and clearance");
}

void MinimumClearanceIgnoresExemptSpheres() {
  // Clause E4/E6: the reported minimum is taken over protected spheres only.
  // Placed at x = 0.60 every sphere clears the 0.25 m envelope even after its
  // own radius and the configured clearance are subtracted.
  const config::HumanSafetyConfig config = MakeConfig();
  Pose torso;
  ArmControllerState arm = MakeArmState(0.60);

  std::size_t exempt_index = 0;
  bool found = false;
  for (std::size_t index = 0; index < kinematics::kLinkSpheres.size(); ++index) {
    if (kinematics::kLinkSpheres[index].mount_exempt) {
      exempt_index = index;
      found = true;
      break;
    }
  }
  CHECK_TRUE(found, "the sphere table has at least one mount-exempt entry");

  const ArmHumanSafetyState before = EvaluateArm(torso, arm, config);
  CHECK_TRUE(before.minimum_clearance_m() > 0.0,
             "all protected spheres clear the envelope at x = 0.60");

  // Drive the exempt sphere deep inside the envelope; the minimum must not move.
  arm.link_safety_points.position_world_m[exempt_index] = Eigen::Vector3d::Zero();
  const ArmHumanSafetyState after = EvaluateArm(torso, arm, config);
  CHECK_CLOSE(after.minimum_clearance_m(), before.minimum_clearance_m(), 0.0,
              "mount-exempt penetration does not change the minimum clearance");
  CHECK_TRUE(after.constraints.signed_clearance_m[exempt_index] < 0.0,
             "the exempt sphere really is penetrating");
  CHECK_TRUE(!after.constraints.active[exempt_index],
             "the penetrating exempt sphere is still not an active constraint");
}

void DisabledFilterReturnsTheRequestUnclipped() {
  // Clause E9: with safety disabled, joint-limit clipping is skipped too.
  config::HumanSafetyConfig config = MakeConfig();
  config.enabled = false;
  ArmHumanSafetyState state;

  const Vector7 requested = Vector7::Constant(9.0);
  const SafetyVelocitySolve solve = ConstrainVelocityForHuman(
      requested, Vector7::Constant(-1.0), Vector7::Constant(1.0), state, config,
      Vector7::Zero(), nullptr);
  CHECK_MATRIX(solve.qdot_safe, requested, 0.0,
               "disabled filter passes the request through unclipped");
  CHECK_TRUE(solve.reason == "disabled", "disabled reason reported");
  CHECK_TRUE(!solve.stopped, "disabled filter does not stop");
}

void NoActiveConstraintsClipsToTheBox() {
  const config::HumanSafetyConfig config = MakeConfig();
  ArmHumanSafetyState state;
  state.evaluated = true;  // evaluated but nothing active

  const SafetyVelocitySolve solve = ConstrainVelocityForHuman(
      Vector7::Constant(9.0), Vector7::Constant(-1.0), Vector7::Constant(1.0),
      state, config, Vector7::Zero(), nullptr);
  CHECK_MATRIX(solve.qdot_safe, Vector7::Constant(1.0), 0.0,
               "no active constraints still clips to the joint-rate box");
  CHECK_TRUE(solve.reason == "clear", "clear reason reported");
  CHECK_TRUE(solve.limit_adjusted, "limit adjustment flagged");
}

ArmHumanSafetyState MakeConstraint(double clearance, const Vector7& row) {
  ArmHumanSafetyState state;
  state.evaluated = true;
  state.constraints.signed_clearance_m = {clearance};
  state.constraints.distance_jacobian_m_rad = {row};
  state.constraints.active = {1};
  state.constraints.mount_exempt = {0};
  return state;
}

void FeasibleRequestPassesThrough() {
  const config::HumanSafetyConfig config = MakeConfig();
  Vector7 row = Vector7::Zero();
  row(0) = 1.0;
  // Clearance well outside the margin: retreating is trivially allowed.
  const ArmHumanSafetyState state = MakeConstraint(0.05, row);

  Vector7 requested = Vector7::Zero();
  requested(0) = 0.10;  // moving away from the envelope
  const SafetyVelocitySolve solve = ConstrainVelocityForHuman(
      requested, Vector7::Constant(-1.0), Vector7::Constant(1.0), state, config,
      Vector7::Zero(), nullptr);
  CHECK_MATRIX(solve.qdot_safe, requested, 0.0,
               "already-feasible request is untouched");
  CHECK_TRUE(!solve.human_adjusted, "no human adjustment reported");
  CHECK_TRUE(solve.active_constraint_count == 1, "active constraint counted");
}

void ApproachIsRepairedToTheBarrier() {
  // Clause E8: a violating request is repaired onto the barrier, and the
  // result must satisfy the constraint it violated.
  const config::HumanSafetyConfig config = MakeConfig();
  Vector7 row = Vector7::Zero();
  row(0) = 1.0;
  const double clearance = 0.01;
  const ArmHumanSafetyState state = MakeConstraint(clearance, row);

  Vector7 requested = Vector7::Zero();
  requested(0) = -0.9;  // driving hard into the envelope
  const SafetyVelocitySolve solve = ConstrainVelocityForHuman(
      requested, Vector7::Constant(-1.0), Vector7::Constant(1.0), state, config,
      Vector7::Zero(), nullptr);

  const double bound =
      -config.recovery_gain_s_inv * (clearance - config.control_margin_m);
  CHECK_TRUE(row.dot(solve.qdot_safe) >= bound - config.constraint_tolerance_m_s,
             "repaired velocity satisfies the barrier inequality");
  CHECK_TRUE(solve.human_adjusted, "human adjustment reported");
  CHECK_TRUE(!solve.stopped, "repairable case does not trigger a stop");
  CHECK_TRUE(solve.reason == "filtered", "filtered reason reported");
}

void InfeasibleBoxTriggersAHoldNotAZero() {
  // Clause E10: on a stop the command is clip(0, lower, upper), which may be
  // non-zero when the persistent command must catch up.
  const config::HumanSafetyConfig config = MakeConfig();
  Vector7 row = Vector7::Zero();
  row(0) = 1.0;
  // Deeply penetrating: recovery demands more outward rate than the box allows.
  const ArmHumanSafetyState state = MakeConstraint(-5.0, row);

  Vector7 lower = Vector7::Constant(0.10);
  Vector7 upper = Vector7::Constant(0.20);
  const SafetyVelocitySolve solve = ConstrainVelocityForHuman(
      Vector7::Zero(), lower, upper, state, config, Vector7::Zero(), nullptr);

  CHECK_TRUE(solve.stopped, "infeasible projection reports a stop");
  CHECK_TRUE(solve.reason == "unsafe_initial_state_hold",
             "penetration reported as unsafe initial state");
  CHECK_MATRIX(solve.qdot_safe, lower, 0.0,
               "hold clips zero into the box rather than commanding zero");
}

}  // namespace

int main() {
  RadialDistanceOutsideTheBand();
  VerticalDistanceAboveTheCap();
  ExteriorCornerUsesTheHypotenuse();
  OnAxisDegeneracyUsesXGradient();
  ActivationMaskRespectsMountExemption();
  MinimumClearanceIgnoresExemptSpheres();
  DisabledFilterReturnsTheRequestUnclipped();
  NoActiveConstraintsClipsToTheBox();
  FeasibleRequestPassesThrough();
  ApproachIsRepairedToTheBarrier();
  InfeasibleBoxTriggersAHoldNotAZero();
  return srl::test::Finish("test_safety");
}
