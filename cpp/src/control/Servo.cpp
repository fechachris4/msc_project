#include "control/Servo.h"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "kinematics/LinkSpheres.h"

namespace srl::control {

ReactivePositionPipeline::ReactivePositionPipeline(
    const PlantState& plant_state, const DualArmPipelineSetup& setup,
    config::ReactivePoseConfig controller_config,
    config::HumanSafetyConfig human_safety_config)
    : human_safety_config_(human_safety_config) {
  for (Side side : kSides) {
    const ArmPipelineSetup& arm_setup = setup.for_arm(side);
    ArmPipeline& pipeline = arms_.for_arm(side);
    pipeline.controller.emplace(controller_config, arm_setup.centering);
    // Seeded from the measured joint position: reset means reconstruction.
    pipeline.integrator.emplace(plant_state.arm(side).position_rad,
                                arm_setup.actuation_limits);
    pipeline.actuation_limits = arm_setup.actuation_limits;
    pipeline.projector = std::make_unique<SafetyVelocityProjector>(
        kinematics::kMaxHumanConstraints, human_safety_config);
  }
}

JointPositionCommand ReactivePositionPipeline::Command() const {
  JointPositionCommand command;
  command.position_rad.right = arms_.right.integrator->command_rad();
  command.position_rad.left = arms_.left.integrator->command_rad();
  return command;
}

std::pair<JointPositionCommand, DualArmControlTraces>
ReactivePositionPipeline::Step(const DualArmControllerStates& states,
                               const DualArmWorldTargets& targets, double dt_s,
                               const std::vector<Side>& arms,
                               const DualArmHumanSafetyStates* human_safety_states) {
  const std::set<Side> unique(arms.begin(), arms.end());
  if (unique.size() != arms.size()) {
    throw std::invalid_argument("arms must not contain duplicates");
  }

  DualArmControlTraces traces;
  DualArm<std::optional<SafetyVelocitySolve>> statuses;

  for (Side side : arms) {
    ArmPipeline& pipeline = arms_.for_arm(side);
    const ArmControllerState& state = states.for_arm(side);
    const ReactiveOutput output =
        pipeline.controller->Compute(state, targets.for_arm(side));

    SafetyVelocitySolve safety;
    if (human_safety_states == nullptr) {
      // Direct pipeline users that did not supply evaluated geometry retain
      // the exact pre-safety actuation path.
      safety.qdot_safe = output.solve.qdot_raw;
      safety.minimum_clearance_m = std::numeric_limits<double>::infinity();
      safety.reason = "not_evaluated";
    } else {
      Vector7 lower_velocity;
      Vector7 upper_velocity;
      pipeline.integrator->VelocityBounds(state.joints.position_rad, dt_s,
                                          lower_velocity, upper_velocity);
      safety = ConstrainVelocityForHuman(
          output.solve.qdot_raw, lower_velocity, upper_velocity,
          human_safety_states->for_arm(side), human_safety_config_,
          state.joints.velocity_rad_s, pipeline.projector.get());
    }

    const PositionActuation actuation = pipeline.integrator->Step(
        state.joints.position_rad, safety.qdot_safe, dt_s);

    // The trace's speed fields describe the RAW request against the limits,
    // not the actuation path -- servo.py computes them from qdot_raw.
    const Vector7& velocity_limit = pipeline.actuation_limits.velocity_rad_s;
    ControlTrace trace;
    for (int index = 0; index < kJoints; ++index) {
      const double raw = output.solve.qdot_raw(index);
      const double limit = velocity_limit(index);
      const double clipped = raw < -limit ? -limit : (raw > limit ? limit : raw);
      trace.qdot_speed_clipped(index) = clipped;
      trace.speed_saturated[index] = raw != clipped;
    }
    trace.J = state.jacobian_world;
    trace.e_pos = output.e_pos;
    trace.e_rot = output.e_rot;
    trace.e_v = output.e_v;
    trace.e_w = output.e_w;
    trace.p_twist = output.solve.p_twist;
    trace.d_twist = output.solve.d_twist;
    trace.task_twist = output.solve.task_twist;
    trace.q = state.joints.position_rad;
    trace.qdot_measured = state.joints.velocity_rad_s;
    trace.qdot_raw = output.solve.qdot_raw;
    trace.qdot_safety_filtered = safety.qdot_safe;
    trace.qdot_effective = actuation.qdot_effective;
    trace.ctrl_before = actuation.command_before_rad;
    trace.ctrl_after = actuation.command_after_rad;
    trace.lead_clamped = actuation.lead_clamped;
    trace.range_clamped = actuation.range_clamped;

    traces.for_arm(side) = trace;
    statuses.for_arm(side) = safety;
  }
  last_human_safety_statuses_ = statuses;
  return {Command(), traces};
}

}  // namespace srl::control
