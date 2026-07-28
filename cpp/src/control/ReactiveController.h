// The complete mathematical policy for the reactive controller.
//
// Read the declarations below in order:
//   1. world-frame pose error,
//   2. world-frame twist error,
//   3. proportional and derivative task twist,
//   4. damped least-squares inverse kinematics,
//   5. null-space joint centering,
//   6. requested joint velocity.
// Equation 7 (safety projection) lives in SafetyFilter.h.
//
// One-to-one with controller/reactive_controller.py's equation ordering.
// No frame conversion, command integration, timing, or backend access here.
// Contract clauses D1-D10.

#pragma once

#include <array>

#include "config/RuntimeConfig.h"
#include "core/Types.h"

namespace srl::control {

struct JointCentering {
  Vector7 midpoint_rad{Vector7::Zero()};
  std::array<bool, kJoints> enabled{};
};

struct ReactiveSolve {
  Vector6 p_twist{Vector6::Zero()};
  Vector6 d_twist{Vector6::Zero()};
  Vector6 task_twist{Vector6::Zero()};
  Vector7 qdot_task{Vector7::Zero()};
  Vector7 qdot_null_objective{Vector7::Zero()};
  Vector7 qdot_null_projected{Vector7::Zero()};
  Vector7 qdot_raw{Vector7::Zero()};
};

struct ReactiveOutput {
  Eigen::Vector3d e_pos{Eigen::Vector3d::Zero()};
  Eigen::Vector3d e_rot{Eigen::Vector3d::Zero()};
  Eigen::Vector3d e_v{Eigen::Vector3d::Zero()};
  Eigen::Vector3d e_w{Eigen::Vector3d::Zero()};
  ReactiveSolve solve;
};

// Equation 1: world-frame pose error, reference minus actual.
// Rotation error is pin.log3(R_ref * R^T).
void PoseError(const ArmControllerState& state, const WorldTarget& target,
               Eigen::Vector3d& e_pos, Eigen::Vector3d& e_rot);

// Equation 2: world-frame twist error, reference minus actual.
void TwistError(const ArmControllerState& state, const WorldTarget& target,
                Eigen::Vector3d& e_v, Eigen::Vector3d& e_w);

// Equation 3: xdot_task = Kp * pose_error + Kd * twist_error.
void TaskTwistTerms(const Eigen::Vector3d& e_pos, const Eigen::Vector3d& e_rot,
                    const Eigen::Vector3d& e_v, const Eigen::Vector3d& e_w,
                    const config::ReactivePoseConfig& config, Vector6& p_twist,
                    Vector6& d_twist);

// Equations 3-6: PD + DLS + null-space requested joint velocity.
ReactiveSolve SolveReactiveVelocity(const Matrix6x7& jacobian_world,
                                    const Eigen::Vector3d& e_pos,
                                    const Eigen::Vector3d& e_rot,
                                    const Eigen::Vector3d& e_v,
                                    const Eigen::Vector3d& e_w,
                                    const Vector7& joint_position_rad,
                                    const Vector7& joint_midpoint_rad,
                                    const Vector7& null_gain_s_inv,
                                    const config::ReactivePoseConfig& config);

// Pure controller policy; no state persists between cycles.
class ReactiveController {
 public:
  ReactiveController(config::ReactivePoseConfig config, JointCentering centering)
      : config_(config), centering_(std::move(centering)) {}

  ReactiveOutput Compute(const ArmControllerState& state,
                         const WorldTarget& target) const;

 private:
  config::ReactivePoseConfig config_;
  JointCentering centering_;
};

}  // namespace srl::control
