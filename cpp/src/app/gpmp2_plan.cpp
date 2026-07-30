#include <cstdio>
#include <exception>
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
  srl::Vector7 goal{srl::Vector7::Zero()};
};

Arguments ParseArguments(int argc, char** argv) {
  if (argc != 10) {
    throw std::invalid_argument(
        "usage: srl_gpmp2_plan <left|right> <output.csv> "
        "<goal_q1_rad> ... <goal_q7_rad>");
  }
  Arguments arguments;
  arguments.side = srl::SideFromName(argv[1]);
  arguments.output_path = argv[2];
  for (int joint = 0; joint < srl::kJoints; ++joint) {
    arguments.goal(joint) = std::stod(argv[3 + joint]);
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
        "human_sl_model_spheres=%zu support_points=%zu output_dt_s=%.9g\n",
        result.planning_sphere_count, result.support_point_count,
        result.output_sample_period_s);
    std::printf("graph_error=%.9g -> %.9g\n", result.initial_graph_error,
                result.final_graph_error);
    std::printf("human_sl_model_clearance_m=%.9g\n",
                result.minimum_planner_sphere_clearance_m);
    std::printf("exact_minimum_clearance_m=%.9g checked_samples=%zu\n",
                validation.minimum_human_clearance_m,
                validation.checked_samples);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "GPMP2 planning failed: %s\n", error.what());
    return 2;
  }
}
