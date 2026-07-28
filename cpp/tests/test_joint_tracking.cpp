#include "TestSupport.h"

#include <limits>

#include "control/JointTrajectoryController.h"

namespace {

using srl::ArmJointState;
using srl::Vector7;
using srl::control::JointTrackingConfig;
using srl::control::JointTrajectoryController;
using srl::planning::JointTrajectorySample;

Vector7 Constant(double value) { return Vector7::Constant(value); }

}  // namespace

int main() {
  JointTrackingConfig config;
  config.kp_s_inv = Constant(3.0);
  config.start_tolerance_rad = 0.05;
  config.max_tracking_error_rad = 0.4;
  const JointTrajectoryController controller(config);

  JointTrajectorySample reference;
  reference.position_rad = Constant(1.0);
  reference.velocity_rad_s = Constant(0.2);
  ArmJointState measured;
  measured.position_rad = Constant(0.9);
  measured.velocity_rad_s = Constant(-0.1);

  const auto output = controller.Compute(reference, measured);
  CHECK_MATRIX(output.position_error_rad, Constant(0.1), 1e-15,
               "position error is reference minus measured");
  CHECK_MATRIX(output.feedback_velocity_rad_s, Constant(0.3), 1e-15,
               "feedback is Kp times position error");
  CHECK_MATRIX(output.requested_velocity_rad_s, Constant(0.5), 1e-15,
               "feedforward and feedback velocities are added");
  CHECK_TRUE(!output.stopped, "in-tolerance tracking continues");

  reference.position_rad = Constant(1.5);
  const auto stopped = controller.Compute(reference, measured);
  CHECK_TRUE(stopped.stopped, "large tracking error stops execution");
  CHECK_TRUE(stopped.reason == "tracking_error_limit",
             "tracking stop has an explicit reason");
  CHECK_MATRIX(stopped.requested_velocity_rad_s, Constant(0.0), 0.0,
               "a stopped controller requests no joint motion");

  JointTrackingConfig bad = config;
  bad.kp_s_inv(2) = -1.0;
  CHECK_THROWS((void)JointTrajectoryController{bad},
               "negative gain is rejected");
  bad = config;
  bad.start_tolerance_rad = 0.5;
  CHECK_THROWS((void)JointTrajectoryController{bad},
               "start tolerance cannot exceed tracking limit");
  bad = config;
  bad.max_tracking_error_rad =
      std::numeric_limits<double>::infinity();
  CHECK_THROWS((void)JointTrajectoryController{bad},
               "tracking limit must be finite");

  return srl::test::Finish("joint tracking");
}
