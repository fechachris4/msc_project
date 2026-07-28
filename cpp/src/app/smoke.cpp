// Stage-0 gate: prove we link and run against the *same* native MuJoCo and
// Pinocchio the Python venv uses, and that Accelerate LAPACK is reachable.
// If this binary runs clean, every later parity claim rests on real ground.

#include <array>
#include <cstdio>
#include <string>

#include <mujoco/mujoco.h>
#include <pinocchio/parsers/mjcf.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>

#include "math/LinAlg.h"
#include "math/Transforms.h"

namespace {

const std::string kPythonRoot = SRL_PYTHON_ROOT;

}  // namespace

int main() {
  std::printf("MuJoCo runtime version : %s\n", mj_versionString());

  // --- MuJoCo scene ---------------------------------------------------
  const std::string scene = kPythonRoot + "/sim/scene.xml";
  std::array<char, 1024> error{};
  mjModel* model =
      mj_loadXML(scene.c_str(), nullptr, error.data(), error.size());
  if (model == nullptr) {
    std::printf("FAILED to load %s: %s\n", scene.c_str(), error.data());
    return 1;
  }
  mjData* data = mj_makeData(model);
  mj_forward(model, data);
  std::printf("scene   : nq=%d nv=%d nu=%d nbody=%d timestep=%g\n", model->nq,
              model->nv, model->nu, model->nbody, model->opt.timestep);

  // --- Pinocchio arm model --------------------------------------------
  const std::string gen3 = kPythonRoot + "/sim/assets/kinova_gen3/gen3.xml";
  pinocchio::Model pin_model;
  pinocchio::mjcf::buildModel(gen3, pin_model);
  pinocchio::Data pin_data(pin_model);
  const auto base_id = pin_model.getFrameId("base_link");
  const auto ee_id = pin_model.getFrameId("pinch_site");
  std::printf("pinocchio: nq=%d nv=%d base_frame=%zu ee_frame=%zu\n",
              static_cast<int>(pin_model.nq), static_cast<int>(pin_model.nv),
              static_cast<std::size_t>(base_id),
              static_cast<std::size_t>(ee_id));
  if (!pin_model.frames[base_id].placement.isIdentity()) {
    std::printf("FAILED: base_link is not fixed at identity\n");
    return 1;
  }

  Eigen::VectorXd q = Eigen::VectorXd::Zero(pin_model.nq);
  pinocchio::computeJointJacobians(pin_model, pin_data, q);
  pinocchio::updateFramePlacements(pin_model, pin_data);
  const auto ee = pin_data.oMf[ee_id];
  std::printf("T_K_E(0) translation = [% .12f % .12f % .12f]\n",
              ee.translation()(0), ee.translation()(1), ee.translation()(2));

  // --- Accelerate LAPACK ----------------------------------------------
  Eigen::Matrix<double, 6, 6> a = Eigen::Matrix<double, 6, 6>::Identity() * 2.0;
  srl::Vector6 b = srl::Vector6::Ones();
  const srl::Vector6 x = srl::linalg::Solve6(a, b);
  std::printf("dgesv solve(2I, 1) = %g (expect 0.5)\n", x(0));

  srl::Matrix6x7 jacobian = srl::Matrix6x7::Random();
  const auto pinv = srl::linalg::Pinv(jacobian);
  const double residual =
      (jacobian * pinv - Eigen::Matrix<double, 6, 6>::Identity())
          .cwiseAbs()
          .maxCoeff();
  std::printf("dgesdd pinv residual |J J+ - I| = %.3e\n", residual);

  mj_deleteData(data);
  mj_deleteModel(model);
  std::printf("SMOKE OK\n");
  return 0;
}
