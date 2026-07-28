// Writes the 210-column control trace in exactly the schema and formatting of
// tests/golden_trace.py: '%.17g' for every numeric field, booleans as 0/1
// (the Python passes them through float()), '\n' line endings, stable column
// order. Contract clauses M3-M4.

#pragma once

#include <fstream>
#include <string>

#include "config/RuntimeConfig.h"
#include "control/Runner.h"
#include "control/Servo.h"
#include "kinematics/PinModel.h"
#include "sim/MujocoBackend.h"

namespace srl::telemetry {

class TraceWriter {
 public:
  TraceWriter(const std::string& path, const config::ReactivePoseConfig& control,
              MountCalibration calibration,
              control::DualArmPipelineSetup pipeline_setup);

  // One CSV row for one arm of one completed cycle.
  void WriteCycle(int cycle, Side side, double dt_s,
                  const control::RunnerCycle& result,
                  const sim::MujocoBackend& backend, kinematics::PinModel& pin);

  void Close();

 private:
  void WriteHeader();

  std::ofstream stream_;
  bool header_written_{false};
  config::ReactivePoseConfig control_;
  MountCalibration calibration_;
  control::DualArmPipelineSetup pipeline_setup_;
};

}  // namespace srl::telemetry
