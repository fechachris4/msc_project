#include "sim/Motion.h"

#include <stdexcept>

#include <mujoco/mujoco.h>

#include "math/Transforms.h"

namespace srl::sim {

std::pair<Eigen::Vector3d, Eigen::Vector3d> TorsoMotion::PoseAt(double t) const {
  // The element-wise rpy sum is only exact because the home rotation is
  // identity (contrast TargetMotion's matrix composition); CaptureTorsoHome
  // enforces that.
  const Eigen::Vector3d position =
      home_position +
      transforms::SineOffset(t, linear_amplitude, linear_frequency);
  const Eigen::Vector3d rpy =
      transforms::SineOffset(t, rotational_amplitude, rotational_frequency);
  return {position, rpy};
}

std::pair<Eigen::Vector3d, Eigen::Vector3d> TorsoMotion::TwistAt(double t) const {
  const Eigen::Vector3d linear =
      transforms::SineRate(t, linear_amplitude, linear_frequency);
  const Eigen::Vector3d rpy =
      transforms::SineOffset(t, rotational_amplitude, rotational_frequency);
  const Eigen::Vector3d rpy_dot =
      transforms::SineRate(t, rotational_amplitude, rotational_frequency);
  return {linear, transforms::AngularVelocityFromRpyRates(rpy, rpy_dot)};
}

TorsoMotion CaptureTorsoHome(const MujocoBackend& backend) {
  const mjModel* model = backend.model();
  const mjData* data = backend.data();
  const int torso_body = mj_name2id(model, mjOBJ_BODY, "torso");
  if (torso_body < 0) throw std::runtime_error("'torso' not in scene");
  const int mocap = model->body_mocapid[torso_body];

  TorsoMotion motion;
  motion.home_position = Eigen::Vector3d(data->mocap_pos[3 * mocap + 0],
                                         data->mocap_pos[3 * mocap + 1],
                                         data->mocap_pos[3 * mocap + 2]);
  const Eigen::Vector4d home_quat(
      data->mocap_quat[4 * mocap + 0], data->mocap_quat[4 * mocap + 1],
      data->mocap_quat[4 * mocap + 2], data->mocap_quat[4 * mocap + 3]);
  if ((home_quat - Eigen::Vector4d(1.0, 0.0, 0.0, 0.0)).cwiseAbs().maxCoeff() >
      1e-8) {
    throw std::runtime_error(
        "torso home rotation is not identity; HOME_RPY = zeros is invalid");
  }
  return motion;
}

void InstallTorsoDriver(MujocoBackend& backend, TorsoMotion motion) {
  backend.ConfigureTorsoDriver(
      [motion](double t) { return motion.PoseAt(t); },
      [motion](double t) { return motion.TwistAt(t); });
}

}  // namespace srl::sim
