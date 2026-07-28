// Pinocchio-backed forward kinematics for one Gen3 arm.
//
// Loads the arm-only MJCF (sim/assets/kinova_gen3/gen3.xml), where base_link
// is fixed at the origin -- so oMf of the EE site is directly T_K_E. The torso
// and world composition are Frames.cpp's concern; Pinocchio never sees them.
// Contract clause B1.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "core/Types.h"

namespace srl::kinematics {

// Owns one Pinocchio model/data pair. Construction is explicit -- the Python
// builds its model at import time, which the project's own risk list flags.
class PinModel {
 public:
  PinModel(const std::string& mjcf_path, const std::string& base_body_name,
           const std::string& ee_site_name);
  ~PinModel();

  PinModel(const PinModel&) = delete;
  PinModel& operator=(const PinModel&) = delete;

  // computeJointJacobians + updateFramePlacements, in that order, exactly as
  // controller/frames.py does before reading anything.
  void Update(const Vector7& joint_position_rad);

  // T_K_E as a homogeneous transform (valid after Update).
  Eigen::Matrix4d EeTransform() const;

  // 6x7 EE Jacobian in LOCAL_WORLD_ALIGNED (valid after Update).
  Matrix6x7 EeJacobian() const;

  // Resolve every link sphere's world position and 3x7 point Jacobian.
  // `base_position_world` is T_W_B's translation; `rotation_world_base` is
  // R_W_T * R_T_B. Contract clause B5.
  void ResolveLinkSafetyPoints(const Eigen::Vector3d& base_position_world,
                               const Eigen::Matrix3d& rotation_world_base,
                               LinkSafetyPoints& out) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace srl::kinematics
