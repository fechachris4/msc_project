#include "TestSupport.h"

#include <memory>
#include <string>

#include "config/RuntimeConfig.h"
#include "control/PlannedJointRunner.h"
#include "kinematics/PinModel.h"
#include "planning/JointTrajectory.h"
#include "planning/PlanValidation.h"
#include "sim/MujocoBackend.h"

namespace {

std::string PythonRoot() { return std::string(SRL_PYTHON_ROOT); }

}  // namespace

int main() {
  const srl::config::ProjectConfig config =
      srl::config::LoadConfig(srl::config::DefaultConfigPath());
  srl::sim::MujocoBackend backend(PythonRoot() + "/sim/scene.xml", config);
  srl::kinematics::PinModel pin(
      PythonRoot() + "/sim/assets/kinova_gen3/gen3.xml", "base_link",
      "pinch_site");

  const srl::PlantState initial = backend.Takeover();
  backend.Release();
  srl::Vector7 goal = initial.arm(srl::Side::Left).position_rad;
  goal(0) += 0.02;
  const auto trajectory =
      std::make_shared<const srl::planning::JointTrajectory>(
          std::vector<srl::planning::JointTrajectoryPoint>{
              {0.0, initial.arm(srl::Side::Left).position_rad,
               srl::Vector7::Zero()},
              {0.10, goal, srl::Vector7::Zero()},
          });

  srl::config::HumanSafetyConfig safety = config.human_safety;
  safety.enabled = false;
  const srl::planning::PlanValidationReport validation =
      srl::planning::ValidateForExecution(
          *trajectory, srl::Side::Left, initial,
          backend.mount_calibration(), pin,
          backend.pipeline_setup()
              .for_arm(srl::Side::Left)
              .actuation_limits,
          safety, initial.nominal_dt_s);
  CHECK_TRUE(validation.valid,
             "the exact geometry validator accepts the small safe plan");
  CHECK_TRUE(validation.checked_samples == 51,
             "validation checks every 2 ms sample including both endpoints");

  srl::control::JointTrackingConfig tracking;
  srl::control::PlannedJointRunner runner(
      backend, pin, backend.mount_calibration(), backend.pipeline_setup(),
      trajectory, srl::Side::Left, tracking, safety);
  runner.Start();

  srl::control::PlannedRunnerCycle cycle;
  for (int index = 0; index < 300; ++index) {
    cycle = runner.Cycle();
    CHECK_TRUE(!cycle.execution_stopped,
               "a small disabled-safety trajectory keeps executing");
  }
  runner.Close();

  CHECK_TRUE(cycle.tracking.reference.finished,
             "the runner holds the final reference after its end time");
  CHECK_TRUE(cycle.human_safety_status.reason == "disabled",
             "the existing safety boundary is still traversed when disabled");
  CHECK_CLOSE(
      cycle.next_state.arm(srl::Side::Left).position_rad(0), goal(0), 0.01,
      "the planned arm reaches its joint-space goal");
  CHECK_MATRIX(
      cycle.command.for_arm(srl::Side::Right),
      initial.arm(srl::Side::Right).position_rad, 0.0,
      "the unplanned arm retains its takeover position command");

  return srl::test::Finish("planned runner");
}
