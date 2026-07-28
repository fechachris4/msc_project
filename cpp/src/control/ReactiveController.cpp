#include "control/ReactiveController.h"

// Pinocchio 4.0 exposes log3 through the spatial umbrella header.
#include <pinocchio/spatial.hpp>

#include "math/LinAlg.h"

namespace srl::control {

void PoseError(const ArmControllerState& state, const WorldTarget& target,
               Eigen::Vector3d& e_pos, Eigen::Vector3d& e_rot) {
  e_pos = target.pose_world.position_m - state.ee_pose_world.position_m;
  e_rot = pinocchio::log3(Eigen::Matrix3d(target.pose_world.rotation *
                                          state.ee_pose_world.rotation.transpose()));
}

void TwistError(const ArmControllerState& state, const WorldTarget& target,
                Eigen::Vector3d& e_v, Eigen::Vector3d& e_w) {
  e_v = target.twist_world.linear_m_s - state.ee_twist_world.linear_m_s;
  e_w = target.twist_world.angular_rad_s - state.ee_twist_world.angular_rad_s;
}

void TaskTwistTerms(const Eigen::Vector3d& e_pos, const Eigen::Vector3d& e_rot,
                    const Eigen::Vector3d& e_v, const Eigen::Vector3d& e_w,
                    const config::ReactivePoseConfig& config, Vector6& p_twist,
                    Vector6& d_twist) {
  // Each 3-block is zeroed independently (clause D4); the derivative term is
  // all-or-nothing (clause D5).
  p_twist.head<3>() = config.position_enabled
                          ? Eigen::Vector3d(config.kp_position_s_inv * e_pos)
                          : Eigen::Vector3d::Zero();
  p_twist.tail<3>() = config.orientation_enabled
                          ? Eigen::Vector3d(config.kp_rotation_s_inv * e_rot)
                          : Eigen::Vector3d::Zero();
  if (config.velocity_enabled) {
    d_twist.head<3>() = config.kd_position * e_v;
    d_twist.tail<3>() = config.kd_rotation * e_w;
  } else {
    d_twist.setZero();
  }
}

ReactiveSolve SolveReactiveVelocity(const Matrix6x7& jacobian_world,
                                    const Eigen::Vector3d& e_pos,
                                    const Eigen::Vector3d& e_rot,
                                    const Eigen::Vector3d& e_v,
                                    const Eigen::Vector3d& e_w,
                                    const Vector7& joint_position_rad,
                                    const Vector7& joint_midpoint_rad,
                                    const Vector7& null_gain_s_inv,
                                    const config::ReactivePoseConfig& config) {
  ReactiveSolve solve;

  // Equation 3: desired world-frame task twist.
  TaskTwistTerms(e_pos, e_rot, e_v, e_w, config, solve.p_twist, solve.d_twist);
  solve.task_twist = solve.p_twist + solve.d_twist;

  // Equation 4: damped least-squares inverse kinematics.
  const double damping = config.dls_damping;
  const Eigen::Matrix<double, 6, 6> damped =
      jacobian_world * jacobian_world.transpose() +
      (damping * damping) * Eigen::Matrix<double, 6, 6>::Identity();
  solve.qdot_task =
      jacobian_world.transpose() * linalg::Solve6(damped, solve.task_twist);

  // Equation 5: joint-centering objective.
  solve.qdot_null_objective =
      -(null_gain_s_inv.array() *
        (joint_position_rad - joint_midpoint_rad).array())
           .matrix();

  // Equation 6: project centering into the Jacobian null space. The
  // pseudo-inverse here is deliberately UNDAMPED while the task solve above is
  // damped -- that asymmetry is part of the behaviour (clause D9).
  const Eigen::Matrix<double, kJoints, kJoints> projector =
      Eigen::Matrix<double, kJoints, kJoints>::Identity() -
      linalg::Pinv(jacobian_world) * jacobian_world;
  solve.qdot_null_projected = projector * solve.qdot_null_objective;
  solve.qdot_raw = solve.qdot_task + solve.qdot_null_projected;
  return solve;
}

ReactiveOutput ReactiveController::Compute(const ArmControllerState& state,
                                           const WorldTarget& target) const {
  ReactiveOutput output;
  PoseError(state, target, output.e_pos, output.e_rot);
  TwistError(state, target, output.e_v, output.e_w);

  // `centering.enabled * null_gain_s_inv` in the Python: a per-joint boolean
  // mask times the scalar gain, not a global on/off (clause D8).
  Vector7 null_gain;
  for (int index = 0; index < kJoints; ++index) {
    null_gain(index) =
        centering_.enabled[index] ? config_.null_gain_s_inv : 0.0;
  }
  output.solve = SolveReactiveVelocity(
      state.jacobian_world, output.e_pos, output.e_rot, output.e_v, output.e_w,
      state.joints.position_rad, centering_.midpoint_rad, null_gain, config_);
  return output;
}

}  // namespace srl::control
