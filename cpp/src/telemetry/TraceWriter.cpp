#include "telemetry/TraceWriter.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "kinematics/Frames.h"
#include "math/LinAlg.h"
#include "sim/Targets.h"

namespace srl::telemetry {
namespace {

// format(float(value), ".17g") in the Python writer.
std::string Format(double value) {
  std::array<char, 64> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "%.17g", value);
  return std::string(buffer.data());
}

class RowBuilder {
 public:
  void Field(const std::string& text) { fields_.push_back(text); }
  void Number(double value) { fields_.push_back(Format(value)); }
  void Flag(bool value) { Number(value ? 1.0 : 0.0); }

  template <typename Derived>
  void Vector(const Eigen::MatrixBase<Derived>& value) {
    for (int index = 0; index < value.size(); ++index) Number(value(index));
  }

  // NumPy flattens row-major; Eigen stores column-major by default.
  template <typename Derived>
  void RowMajorMatrix(const Eigen::MatrixBase<Derived>& value) {
    for (int row = 0; row < value.rows(); ++row) {
      for (int column = 0; column < value.cols(); ++column) {
        Number(value(row, column));
      }
    }
  }

  void Flags(const control::JointFlags& value) {
    for (bool flag : value) Flag(flag);
  }

  std::string Line() const {
    std::string line;
    for (std::size_t index = 0; index < fields_.size(); ++index) {
      if (index > 0) line += ",";
      line += fields_[index];
    }
    return line + "\n";
  }

  std::size_t size() const { return fields_.size(); }

 private:
  std::vector<std::string> fields_;
};

void AppendNames(std::vector<std::string>& names, const std::string& prefix,
                 int count) {
  for (int index = 0; index < count; ++index) {
    names.push_back(prefix + "_" + std::to_string(index));
  }
}

}  // namespace

TraceWriter::TraceWriter(const std::string& path,
                         const config::ReactivePoseConfig& control,
                         MountCalibration calibration,
                         control::DualArmPipelineSetup pipeline_setup)
    : stream_(path, std::ios::binary),
      control_(control),
      calibration_(std::move(calibration)),
      pipeline_setup_(std::move(pipeline_setup)) {
  if (!stream_) throw std::runtime_error("cannot open trace file: " + path);
}

void TraceWriter::WriteHeader() {
  std::vector<std::string> names = {"cycle", "arm", "sample_time_s", "dt_s"};
  AppendNames(names, "torso_position_m", 3);
  AppendNames(names, "torso_rotation", 9);
  AppendNames(names, "torso_linear_velocity_m_s", 3);
  AppendNames(names, "torso_angular_velocity_rad_s", 3);
  AppendNames(names, "target_position_m", 3);
  AppendNames(names, "target_quaternion_wxyz", 4);
  AppendNames(names, "ee_position_m", 3);
  AppendNames(names, "ee_rotation", 9);
  AppendNames(names, "ee_linear_velocity_m_s", 3);
  AppendNames(names, "ee_angular_velocity_rad_s", 3);
  AppendNames(names, "qdot_task", 7);
  AppendNames(names, "qdot_null_objective", 7);
  AppendNames(names, "qdot_null_projected", 7);
  // ControlTrace dataclass field order, with qdot_safety_filtered omitted.
  AppendNames(names, "J", 42);
  AppendNames(names, "e_pos", 3);
  AppendNames(names, "e_rot", 3);
  AppendNames(names, "e_v", 3);
  AppendNames(names, "e_w", 3);
  AppendNames(names, "p_twist", 6);
  AppendNames(names, "d_twist", 6);
  AppendNames(names, "task_twist", 6);
  AppendNames(names, "q", 7);
  AppendNames(names, "qdot_measured", 7);
  AppendNames(names, "qdot_raw", 7);
  AppendNames(names, "qdot_speed_clipped", 7);
  AppendNames(names, "qdot_effective", 7);
  AppendNames(names, "ctrl_before", 7);
  AppendNames(names, "ctrl_after", 7);
  AppendNames(names, "speed_saturated", 7);
  AppendNames(names, "lead_clamped", 7);
  AppendNames(names, "range_clamped", 7);

  std::string line;
  for (std::size_t index = 0; index < names.size(); ++index) {
    if (index > 0) line += ",";
    line += names[index];
  }
  stream_ << line << "\n";
  header_written_ = true;
}

void TraceWriter::WriteCycle(int cycle, Side side, double dt_s,
                             const control::RunnerCycle& result,
                             const sim::MujocoBackend& backend,
                             kinematics::PinModel& pin) {
  if (!header_written_) WriteHeader();
  const control::ControlTrace& trace = *result.traces.for_arm(side);
  const PlantState& plant = result.input_state;

  // The Python recomputes the arm state from the input sample rather than
  // reusing the cycle's, so do the same.
  const ArmControllerState state =
      kinematics::ArmControllerStateOf(pin, plant, side, calibration_);

  RowBuilder row;
  row.Number(cycle);
  row.Field(std::string(SideName(side)));
  row.Number(plant.sample_time_s);
  row.Number(dt_s);
  row.Vector(plant.torso_pose_world.position_m);
  row.RowMajorMatrix(plant.torso_pose_world.rotation);
  row.Vector(plant.torso_twist_world.linear_m_s);
  row.Vector(plant.torso_twist_world.angular_rad_s);
  row.Vector(sim::TargetPosition(backend, side));
  row.Vector(sim::TargetQuat(backend, side));
  row.Vector(state.ee_pose_world.position_m);
  row.RowMajorMatrix(state.ee_pose_world.rotation);
  row.Vector(state.ee_twist_world.linear_m_s);
  row.Vector(state.ee_twist_world.angular_rad_s);

  // Stage reconstruction from the recorded J and task twist, mirroring
  // _cycle_rows -- and, like it, verified against qdot_raw.
  const control::JointCentering& centering =
      pipeline_setup_.for_arm(side).centering;
  const Eigen::Matrix<double, 6, 6> damped =
      trace.J * trace.J.transpose() +
      (control_.dls_damping * control_.dls_damping) *
          Eigen::Matrix<double, 6, 6>::Identity();
  const Vector7 qdot_task =
      trace.J.transpose() * linalg::Solve6(damped, trace.task_twist);

  Vector7 qdot_null_objective;
  for (int index = 0; index < kJoints; ++index) {
    const double gain =
        centering.enabled[index] ? control_.null_gain_s_inv : 0.0;
    qdot_null_objective(index) =
        -gain * (trace.q(index) - centering.midpoint_rad(index));
  }
  const Vector7 qdot_null_projected =
      (Eigen::Matrix<double, kJoints, kJoints>::Identity() -
       linalg::Pinv(trace.J) * trace.J) *
      qdot_null_objective;

  if (((qdot_task + qdot_null_projected) - trace.qdot_raw).cwiseAbs().maxCoeff() >
      1e-14) {
    throw std::runtime_error("trace stage reconstruction differs for cycle " +
                             std::to_string(cycle) + " " +
                             std::string(SideName(side)));
  }

  row.Vector(qdot_task);
  row.Vector(qdot_null_objective);
  row.Vector(qdot_null_projected);

  row.RowMajorMatrix(trace.J);
  row.Vector(trace.e_pos);
  row.Vector(trace.e_rot);
  row.Vector(trace.e_v);
  row.Vector(trace.e_w);
  row.Vector(trace.p_twist);
  row.Vector(trace.d_twist);
  row.Vector(trace.task_twist);
  row.Vector(trace.q);
  row.Vector(trace.qdot_measured);
  row.Vector(trace.qdot_raw);
  row.Vector(trace.qdot_speed_clipped);
  // qdot_safety_filtered is intentionally not a column; the Python asserts it
  // equals qdot_raw when safety is disabled, so assert the same here.
  if ((trace.qdot_safety_filtered - trace.qdot_raw).cwiseAbs().maxCoeff() != 0.0) {
    throw std::runtime_error("qdot_safety_filtered != qdot_raw at cycle " +
                             std::to_string(cycle));
  }
  row.Vector(trace.qdot_effective);
  row.Vector(trace.ctrl_before);
  row.Vector(trace.ctrl_after);
  row.Flags(trace.speed_saturated);
  row.Flags(trace.lead_clamped);
  row.Flags(trace.range_clamped);

  if (row.size() != 210) {
    throw std::runtime_error("trace row has " + std::to_string(row.size()) +
                             " fields, expected 210");
  }
  stream_ << row.Line();
}

void TraceWriter::Close() { stream_.close(); }

}  // namespace srl::telemetry
