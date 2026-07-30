#include <cstdio>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>

#include "config/RuntimeConfig.h"
#include "kinematics/PinModel.h"
#include "planning/Gpmp2Planner.h"
#include "planning/JointTrajectoryCsv.h"
#include "planning/PlanValidation.h"
#include "sim/MujocoBackend.h"

namespace {

std::string PythonRoot() { return std::string(SRL_PYTHON_ROOT); }

struct Arguments {
  srl::Side side{srl::Side::Left};
  std::string output_path;
  std::optional<srl::Vector7> start;
  srl::Vector7 goal{srl::Vector7::Zero()};
};

Arguments ParseArguments(int argc, char** argv) {
  if (argc != 10 && (argc != 18 || std::string(argv[3]) != "--start")) {
    throw std::invalid_argument(
        "usage: srl_gpmp2_plan <left|right> <output.csv> "
        "[--start <start_q1_rad> ... <start_q7_rad>] "
        "<goal_q1_rad> ... <goal_q7_rad>");
  }
  Arguments arguments;
  arguments.side = srl::SideFromName(argv[1]);
  arguments.output_path = argv[2];
  int goal_offset = 3;
  if (argc == 18) {
    srl::Vector7 start;
    for (int joint = 0; joint < srl::kJoints; ++joint) {
      start(joint) = std::stod(argv[4 + joint]);
    }
    arguments.start = start;
    goal_offset = 11;
  }
  for (int joint = 0; joint < srl::kJoints; ++joint) {
    arguments.goal(joint) = std::stod(argv[goal_offset + joint]);
  }
  return arguments;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    const srl::config::ProjectConfig config =
        srl::config::LoadConfig(srl::config::DefaultConfigPath());
    srl::sim::MujocoBackend backend(PythonRoot() + "/sim/scene.xml", config);
    srl::kinematics::PinModel pin(
        PythonRoot() + "/sim/assets/kinova_gen3/gen3.xml", "base_link",
        "pinch_site");

    const std::optional<srl::Vector7>& configured_start =
        config.simulation.initial_joint_position(arguments.side);
    const std::optional<srl::Vector7>& selected_start =
        arguments.start ? arguments.start : configured_start;
    if (selected_start) {
      backend.SetJointPosition(arguments.side, *selected_start);
      backend.ZeroVelocities();
      backend.Forward();
    }
    const srl::PlantState start_state = backend.Takeover();
    backend.Release();
    srl::planning::Gpmp2Request request;
    request.side = arguments.side;
    request.start_position_rad =
        start_state.arm(arguments.side).position_rad;
    request.start_velocity_rad_s =
        start_state.arm(arguments.side).velocity_rad_s;
    request.goal_position_rad = arguments.goal;
    request.mount_calibration = backend.mount_calibration();
    request.limits =
        backend.pipeline_setup().for_arm(arguments.side).actuation_limits;
    request.human_safety = config.human_safety;

    const srl::planning::Gpmp2Result result =
        srl::planning::PlanWithGpmp2(request);
    const srl::planning::PlanValidationReport validation =
        srl::planning::ValidateForExecution(
            *result.trajectory, arguments.side, start_state,
            backend.mount_calibration(), pin, request.limits,
            config.human_safety, start_state.nominal_dt_s);
    if (!validation.valid) {
      std::fprintf(
          stderr,
          "rejected_plan_metrics: velocity_ratio=%.9g "
          "joint_margin_rad=%.9g exact_clearance_m=%.9g "
          "maximum_acceleration_rad_s2=%.9g "
          "human_sl_model_clearance_m=%.9g checked_samples=%zu\n",
          validation.maximum_velocity_ratio,
          validation.minimum_joint_margin_rad,
          validation.minimum_human_clearance_m,
          validation.maximum_acceleration_rad_s2,
          result.minimum_planner_sphere_clearance_m,
          validation.checked_samples);
      throw std::runtime_error(
          "optimized trajectory failed exact post-validation: " +
          validation.reason);
    }
    srl::planning::WriteJointTrajectoryCsv(*result.trajectory,
                                            arguments.output_path);

    std::printf("trajectory=%s\n", arguments.output_path.c_str());
    std::printf("samples=%zu duration_s=%.6f\n",
                result.trajectory->points().size(),
                result.trajectory->duration_s());
    std::printf(
        "human_sl_model_spheres=%zu external_obstacle_spheres=%zu "
        "support_points=%zu output_dt_s=%.9g retiming_scale=%.9g\n",
        result.planning_sphere_count,
        result.external_obstacle_sphere_count,
        result.support_point_count, result.output_sample_period_s,
        result.retiming_scale);
    std::printf("graph_error=%.9g -> %.9g\n", result.initial_graph_error,
                result.final_graph_error);
    std::printf("endpoint_error_rad=start %.9g goal %.9g\n",
                result.maximum_start_error_rad,
                result.maximum_goal_error_rad);
    std::printf("human_sl_model_clearance_m=%.9g\n",
                result.minimum_planner_sphere_clearance_m);
    std::printf(
        "exact_maximum_velocity_ratio=%.9g "
        "exact_minimum_clearance_m=%.9g checked_samples=%zu\n",
        validation.maximum_velocity_ratio,
        validation.minimum_human_clearance_m,
        validation.checked_samples);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "GPMP2 planning failed: %s\n", error.what());
    return 2;
  }
}
