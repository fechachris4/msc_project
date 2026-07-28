#include "math/Transforms.h"

#include <cmath>

namespace srl::transforms {

Eigen::Matrix4d TransformFromPose(const Eigen::Vector3d& position,
                                  const Eigen::Matrix3d& rotation) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.topLeftCorner<3, 3>() = rotation;
  transform.topRightCorner<3, 1>() = position;
  return transform;
}

void PoseFromTransform(const Eigen::Matrix4d& transform,
                       Eigen::Vector3d& position, Eigen::Matrix3d& rotation) {
  position = transform.topRightCorner<3, 1>();
  rotation = transform.topLeftCorner<3, 3>();
}

Eigen::Matrix3d RotationFromQuat(const Eigen::Vector4d& quat) {
  const Eigen::Vector4d unit = quat / quat.norm();
  const double w = unit(0), x = unit(1), y = unit(2), z = unit(3);
  Eigen::Matrix3d rotation;
  rotation << 1 - 2 * (y * y + z * z), 2 * (x * y - w * z),
      2 * (x * z + w * y),
      2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
      2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y);
  return rotation;
}

Eigen::Matrix3d RotationFromRpy(const Eigen::Vector3d& rpy) {
  const double cr = std::cos(rpy(0)), sr = std::sin(rpy(0));
  const double cp = std::cos(rpy(1)), sp = std::sin(rpy(1));
  const double cy = std::cos(rpy(2)), sy = std::sin(rpy(2));
  Eigen::Matrix3d rx;
  rx << 1, 0, 0, 0, cr, -sr, 0, sr, cr;
  Eigen::Matrix3d ry;
  ry << cp, 0, sp, 0, 1, 0, -sp, 0, cp;
  Eigen::Matrix3d rz;
  rz << cy, -sy, 0, sy, cy, 0, 0, 0, 1;
  // Same association order as NumPy's `Rz @ Ry @ Rx`.
  const Eigen::Matrix3d zy = rz * ry;
  return zy * rx;
}

Eigen::Vector3d AngularVelocityFromRpyRates(const Eigen::Vector3d& rpy,
                                            const Eigen::Vector3d& rpy_dot) {
  const double cp = std::cos(rpy(1)), sp = std::sin(rpy(1));
  const double cy = std::cos(rpy(2)), sy = std::sin(rpy(2));
  Eigen::Matrix3d e;
  e << cy * cp, -sy, 0.0, sy * cp, cy, 0.0, -sp, 0.0, 1.0;
  return e * rpy_dot;
}

}  // namespace srl::transforms
