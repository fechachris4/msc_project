// Scripted base motion: sinusoidal torso disturbance, world frame.
//
// The torso mocap body is driven as home + A*sin(2*pi*f*t) in position (world
// xyz) and orientation (rpy). Mocap writes teleport the torso -- mocap bodies
// carry no velocity state -- so the matching analytic twist is supplied
// separately. One-to-one with sim/motion.py.

#pragma once

#include "sim/MujocoBackend.h"

namespace srl::sim {

struct TorsoMotion {
  // Research levers. The Python module defaults are all-zero amplitudes, i.e.
  // a static torso for the default scenario (clause L1).
  Eigen::Vector3d linear_amplitude{Eigen::Vector3d::Zero()};
  Eigen::Vector3d rotational_amplitude{Eigen::Vector3d::Zero()};
  double linear_frequency{0.5};
  double rotational_frequency{0.5};

  // Captured from the scene's mocap home; HOME_RPY is zero and is only valid
  // because the torso home rotation is identity, which is asserted on capture.
  Eigen::Vector3d home_position{Eigen::Vector3d::Zero()};

  // (position, rpy) of the scripted torso at time t -- pure math.
  std::pair<Eigen::Vector3d, Eigen::Vector3d> PoseAt(double t) const;

  // (v, omega) at time t, world frame: the analytic derivative of PoseAt.
  // This is the base twist the mocap teleports never give the simulator; on
  // hardware the same quantity comes from Vicon.
  std::pair<Eigen::Vector3d, Eigen::Vector3d> TwistAt(double t) const;
};

// Capture the torso mocap home and verify its rotation is identity.
TorsoMotion CaptureTorsoHome(const MujocoBackend& backend);

// Install PoseAt/TwistAt on the backend as a matched pair.
void InstallTorsoDriver(MujocoBackend& backend, TorsoMotion motion);

}  // namespace srl::sim
