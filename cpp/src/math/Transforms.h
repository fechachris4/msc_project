// Pure rigid-transform math. One-to-one with controller/transforms.py.
// No MuJoCo, no Pinocchio, no scene knowledge.

#pragma once

#include <Eigen/Dense>

namespace srl::transforms {

Eigen::Matrix4d TransformFromPose(const Eigen::Vector3d& position,
                                  const Eigen::Matrix3d& rotation);

void PoseFromTransform(const Eigen::Matrix4d& transform,
                       Eigen::Vector3d& position, Eigen::Matrix3d& rotation);

// Rotation matrix from a MuJoCo quaternion [w, x, y, z]. The quaternion is
// normalised first, exactly as the Python does.
Eigen::Matrix3d RotationFromQuat(const Eigen::Vector4d& quat);

// Rotation matrix from [roll, pitch, yaw], composed as Rz @ Ry @ Rx.
Eigen::Matrix3d RotationFromRpy(const Eigen::Vector3d& rpy);

// World-frame angular velocity of a frame following RotationFromRpy(rpy(t)).
// Columns of E are the axes each rate spins about, expressed in the world.
Eigen::Vector3d AngularVelocityFromRpyRates(const Eigen::Vector3d& rpy,
                                            const Eigen::Vector3d& rpy_dot);

inline Eigen::Vector3d SineOffset(double t, const Eigen::Vector3d& amplitude,
                                  double frequency) {
  return amplitude * std::sin(2.0 * M_PI * frequency * t);
}

// d/dt of SineOffset.
inline Eigen::Vector3d SineRate(double t, const Eigen::Vector3d& amplitude,
                                double frequency) {
  const double w = 2.0 * M_PI * frequency;
  return amplitude * (w * std::cos(w * t));
}

}  // namespace srl::transforms
