#include "kinematics/PinModel.h"

#include <map>
#include <stdexcept>

#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "kinematics/LinkSpheres.h"

namespace srl::kinematics {

struct PinModel::Impl {
  pinocchio::Model model;
  pinocchio::Data data;
  pinocchio::FrameIndex ee_frame_id{0};

  // One entry per distinct sphere frame, in first-appearance order (the
  // Python builds this with dict.fromkeys, which preserves that order).
  struct FrameGroup {
    pinocchio::FrameIndex frame_id;
    std::vector<std::size_t> sphere_indices;
  };
  std::vector<FrameGroup> frame_groups;

  explicit Impl(pinocchio::Model built) : model(std::move(built)), data(model) {}
};

PinModel::PinModel(const std::string& mjcf_path,
                   const std::string& base_body_name,
                   const std::string& ee_site_name) {
  pinocchio::Model built;
  pinocchio::mjcf::buildModel(mjcf_path, built);

  if (!built.existFrame(base_body_name)) {
    throw std::invalid_argument("base frame '" + base_body_name +
                                "' not found in " + mjcf_path);
  }
  // base_body_name must be the fixed root of this MJCF (pose = identity),
  // otherwise oMf is not expressed in the base frame.
  if (!built.frames[built.getFrameId(base_body_name)].placement.isIdentity()) {
    throw std::invalid_argument("base frame '" + base_body_name +
                                "' is not fixed at identity");
  }
  if (!built.existFrame(ee_site_name)) {
    throw std::invalid_argument("MJCF site '" + ee_site_name +
                                "' did not map to a Pinocchio frame in " +
                                mjcf_path);
  }

  impl_ = std::make_unique<Impl>(std::move(built));
  impl_->ee_frame_id = impl_->model.getFrameId(ee_site_name);

  std::vector<std::string_view> seen;
  for (std::size_t index = 0; index < kLinkSpheres.size(); ++index) {
    const std::string_view frame_name = kLinkSpheres[index].frame_name;
    auto position = std::find(seen.begin(), seen.end(), frame_name);
    if (position == seen.end()) {
      seen.push_back(frame_name);
      const std::string name(frame_name);
      if (!impl_->model.existFrame(name)) {
        throw std::invalid_argument("link-sphere frame '" + name +
                                    "' not found in " + mjcf_path);
      }
      impl_->frame_groups.push_back({impl_->model.getFrameId(name), {index}});
    } else {
      const auto group = static_cast<std::size_t>(position - seen.begin());
      impl_->frame_groups[group].sphere_indices.push_back(index);
    }
  }
}

PinModel::~PinModel() = default;

void PinModel::Update(const Vector7& joint_position_rad) {
  const Eigen::VectorXd q = joint_position_rad;
  pinocchio::computeJointJacobians(impl_->model, impl_->data, q);
  pinocchio::updateFramePlacements(impl_->model, impl_->data);
}

Eigen::Matrix4d PinModel::EeTransform() const {
  return impl_->data.oMf[impl_->ee_frame_id].toHomogeneousMatrix();
}

Matrix6x7 PinModel::EeJacobian() const {
  pinocchio::Data::Matrix6x jacobian(6, impl_->model.nv);
  jacobian.setZero();
  pinocchio::getFrameJacobian(impl_->model, impl_->data, impl_->ee_frame_id,
                              pinocchio::LOCAL_WORLD_ALIGNED, jacobian);
  return jacobian;
}

void PinModel::ResolveLinkSafetyPoints(
    const Eigen::Vector3d& base_position_world,
    const Eigen::Matrix3d& rotation_world_base, LinkSafetyPoints& out) const {
  out.position_world_m.resize(kLinkSpheres.size());
  out.jacobian_world_m_rad.resize(kLinkSpheres.size());

  pinocchio::Data::Matrix6x frame_jacobian(6, impl_->model.nv);
  for (const auto& group : impl_->frame_groups) {
    const auto& placement = impl_->data.oMf[group.frame_id];
    frame_jacobian.setZero();
    pinocchio::getFrameJacobian(impl_->model, impl_->data, group.frame_id,
                                pinocchio::LOCAL_WORLD_ALIGNED,
                                frame_jacobian);
    const auto linear = frame_jacobian.topRows<3>();
    const auto angular = frame_jacobian.bottomRows<3>();

    for (std::size_t index : group.sphere_indices) {
      const auto& centre = kLinkSpheres[index].center_frame_m;
      const Eigen::Vector3d centre_frame(centre[0], centre[1], centre[2]);
      // offset_base = R_frame * c ; position_base = p_frame + offset_base
      const Eigen::Vector3d offset_base = placement.rotation() * centre_frame;
      const Eigen::Vector3d position_base =
          placement.translation() + offset_base;

      // Rigid-body point Jacobian: column j is J_lin_j + J_ang_j x offset.
      Matrix3x7 point_jacobian_base;
      for (int joint = 0; joint < kJoints; ++joint) {
        point_jacobian_base.col(joint) =
            linear.col(joint) +
            angular.col(joint).cross(offset_base);
      }

      out.position_world_m[index] =
          base_position_world + rotation_world_base * position_base;
      out.jacobian_world_m_rad[index] =
          rotation_world_base * point_jacobian_base;
    }
  }
}

}  // namespace srl::kinematics
