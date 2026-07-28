// Persistent position integration and its exact clamp-free velocity interval.
// Contract clauses F1-F5.

#include "TestSupport.h"

#include "control/PositionActuation.h"

using namespace srl;
using namespace srl::control;

namespace {

PositionActuationLimits MakeLimits() {
  PositionActuationLimits limits;
  limits.velocity_rad_s = Vector7::Constant(1.0);
  limits.lead_rad = 0.2;
  limits.lower_position_rad = Vector7::Constant(-2.0);
  limits.upper_position_rad = Vector7::Constant(2.0);
  return limits;
}

void SeededFromMeasuredPosition() {
  // Clause F1: the command starts at the measured position, and "reset" means
  // constructing a new integrator.
  const Vector7 measured = Vector7::LinSpaced(0.1, 0.7);
  PositionIntegrator integrator(measured, MakeLimits());
  CHECK_MATRIX(integrator.command_rad(), measured, 0.0,
               "command seeded from measured position");
}

void ClampOrderIsSpeedThenLeadThenRange() {
  // Clause F2. A huge request must be speed-clipped first, then lead-clamped
  // about the measured position, then clamped to the actuator range.
  PositionActuationLimits limits = MakeLimits();
  limits.upper_position_rad = Vector7::Constant(0.05);
  const Vector7 measured = Vector7::Zero();
  PositionIntegrator integrator(measured, limits);

  const Vector7 requested = Vector7::Constant(50.0);
  const PositionActuation actuation = integrator.Step(measured, requested, 0.002);

  CHECK_MATRIX(actuation.qdot_speed_clipped, Vector7::Constant(1.0), 0.0,
               "requested velocity clipped to the speed limit");
  for (int index = 0; index < kJoints; ++index) {
    CHECK_TRUE(actuation.speed_saturated[index], "speed saturation flagged");
  }
  // 0 + 1.0 * 0.002 = 0.002, inside both the lead band and the range.
  CHECK_MATRIX(actuation.command_after_rad, Vector7::Constant(0.002), 1e-18,
               "integrated command within lead and range");
  CHECK_MATRIX(actuation.qdot_effective, Vector7::Constant(1.0), 1e-12,
               "effective velocity equals the applied step over dt");
}

void RangeClampEngagesAtTheActuatorBound() {
  PositionActuationLimits limits = MakeLimits();
  limits.upper_position_rad = Vector7::Constant(0.001);
  const Vector7 measured = Vector7::Zero();
  PositionIntegrator integrator(measured, limits);
  const PositionActuation actuation =
      integrator.Step(measured, Vector7::Constant(1.0), 0.002);
  CHECK_MATRIX(actuation.command_after_rad, Vector7::Constant(0.001), 1e-18,
               "command clamped to the actuator range");
  for (int index = 0; index < kJoints; ++index) {
    CHECK_TRUE(actuation.range_clamped[index], "range clamp flagged");
  }
}

void LeadClampEngagesAwayFromMeasured() {
  PositionActuationLimits limits = MakeLimits();
  limits.lead_rad = 0.0005;
  const Vector7 measured = Vector7::Zero();
  PositionIntegrator integrator(measured, limits);
  const PositionActuation actuation =
      integrator.Step(measured, Vector7::Constant(1.0), 0.002);
  CHECK_MATRIX(actuation.command_after_rad, Vector7::Constant(0.0005), 1e-18,
               "command clamped to the lead band about measured");
  for (int index = 0; index < kJoints; ++index) {
    CHECK_TRUE(actuation.lead_clamped[index], "lead clamp flagged");
  }
}

void VelocityBoundsAvoidEveryClamp() {
  // Clause F4: stepping at exactly the returned bound must engage no clamp.
  PositionActuationLimits limits = MakeLimits();
  limits.lead_rad = 0.01;
  limits.upper_position_rad = Vector7::Constant(0.5);
  limits.lower_position_rad = Vector7::Constant(-0.5);
  const Vector7 measured = Vector7::Constant(0.3);

  for (double sign : {1.0, -1.0}) {
    PositionIntegrator integrator(measured, limits);
    Vector7 lower;
    Vector7 upper;
    integrator.VelocityBounds(measured, 0.002, lower, upper);
    const Vector7 requested = sign > 0 ? upper : lower;
    const PositionActuation actuation =
        integrator.Step(measured, requested, 0.002);
    for (int index = 0; index < kJoints; ++index) {
      CHECK_TRUE(!actuation.speed_saturated[index],
                 "bound velocity does not saturate speed");
      CHECK_TRUE(!actuation.lead_clamped[index],
                 "bound velocity does not hit the lead clamp");
      CHECK_TRUE(!actuation.range_clamped[index],
                 "bound velocity does not hit the range clamp");
    }
    CHECK_MATRIX(actuation.qdot_effective, requested, 1e-12,
                 "effective velocity equals the requested bound");
  }
}

void CommandPersistsAcrossSteps() {
  // Clause F1: the command integrates, it is not recomputed from measured.
  PositionIntegrator integrator(Vector7::Zero(), MakeLimits());
  const Vector7 measured = Vector7::Zero();
  integrator.Step(measured, Vector7::Constant(0.5), 0.002);
  const PositionActuation second =
      integrator.Step(measured, Vector7::Constant(0.5), 0.002);
  CHECK_MATRIX(second.command_before_rad, Vector7::Constant(0.001), 1e-18,
               "second step starts from the first step's command");
  CHECK_MATRIX(second.command_after_rad, Vector7::Constant(0.002), 1e-18,
               "command accumulates across steps");
}

void RejectsBadInputs() {
  PositionIntegrator integrator(Vector7::Zero(), MakeLimits());
  CHECK_THROWS(integrator.Step(Vector7::Zero(), Vector7::Zero(), 0.0),
               "zero dt rejected");
  CHECK_THROWS(integrator.Step(Vector7::Zero(), Vector7::Zero(), -1.0),
               "negative dt rejected");

  PositionActuationLimits bad = MakeLimits();
  bad.velocity_rad_s(3) = 0.0;
  CHECK_THROWS(PositionIntegrator(Vector7::Zero(), bad),
               "non-positive velocity limit rejected");

  PositionActuationLimits inverted = MakeLimits();
  inverted.lower_position_rad(2) = 5.0;
  CHECK_THROWS(PositionIntegrator(Vector7::Zero(), inverted),
               "inverted position bounds rejected");
}

void UnlimitedActuatorBoundsAreAllowed() {
  // MuJoCo reports +/-inf for actuators without a ctrlrange; that must be a
  // legal, non-clamping bound rather than an error.
  PositionActuationLimits limits = MakeLimits();
  limits.lower_position_rad = Vector7::Constant(
      -std::numeric_limits<double>::infinity());
  limits.upper_position_rad =
      Vector7::Constant(std::numeric_limits<double>::infinity());
  PositionIntegrator integrator(Vector7::Zero(), limits);
  const PositionActuation actuation =
      integrator.Step(Vector7::Zero(), Vector7::Constant(0.5), 0.002);
  for (int index = 0; index < kJoints; ++index) {
    CHECK_TRUE(!actuation.range_clamped[index],
               "infinite bounds never range-clamp");
  }
}

}  // namespace

int main() {
  SeededFromMeasuredPosition();
  ClampOrderIsSpeedThenLeadThenRange();
  RangeClampEngagesAtTheActuatorBound();
  LeadClampEngagesAwayFromMeasured();
  VelocityBoundsAvoidEveryClamp();
  CommandPersistsAcrossSteps();
  RejectsBadInputs();
  UnlimitedActuatorBoundsAreAllowed();
  return srl::test::Finish("test_actuation");
}
