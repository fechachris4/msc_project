// Transforms and linear algebra. Contract clauses A5, D7, D9.

#include "TestSupport.h"

#include "math/LinAlg.h"
#include "math/Transforms.h"

using namespace srl;
using namespace srl::transforms;

namespace {

// Aliases keep template commas out of the assertion macros.
using M66 = Eigen::Matrix<double, 6, 6>;
using M76 = Eigen::Matrix<double, kJoints, 6>;
using M77 = Eigen::Matrix<double, kJoints, kJoints>;

void RpyComposition() {
  // Clause A5: R = Rz(yaw) Ry(pitch) Rx(roll), in that association order.
  const Eigen::Vector3d rpy(0.3, -0.7, 1.1);
  const Eigen::Matrix3d actual = RotationFromRpy(rpy);

  Eigen::Matrix3d rx, ry, rz;
  rx = Eigen::AngleAxisd(rpy(0), Eigen::Vector3d::UnitX()).toRotationMatrix();
  ry = Eigen::AngleAxisd(rpy(1), Eigen::Vector3d::UnitY()).toRotationMatrix();
  rz = Eigen::AngleAxisd(rpy(2), Eigen::Vector3d::UnitZ()).toRotationMatrix();
  CHECK_MATRIX(actual, Eigen::Matrix3d(rz * ry * rx), 1e-15,
               "rotation_from_rpy is Rz*Ry*Rx");

  CHECK_MATRIX(Eigen::Matrix3d(actual.transpose() * actual),
               Eigen::Matrix3d::Identity(), 1e-15, "rpy rotation orthonormal");
  CHECK_CLOSE(actual.determinant(), 1.0, 1e-15, "rpy rotation right-handed");
}

void QuaternionRoundTrip() {
  // MuJoCo order [w, x, y, z], normalised before use.
  const Eigen::Vector3d rpy(-0.2, 0.9, 0.4);
  const Eigen::Matrix3d rotation = RotationFromRpy(rpy);
  const Eigen::Quaterniond quaternion(rotation);
  const Eigen::Vector4d wxyz(quaternion.w(), quaternion.x(), quaternion.y(),
                             quaternion.z());
  CHECK_MATRIX(RotationFromQuat(wxyz), rotation, 1e-14,
               "rotation_from_quat inverts the rotation");

  // Deliberately unnormalised input must be normalised internally.
  CHECK_MATRIX(RotationFromQuat(Eigen::Vector4d(wxyz * 3.7)), rotation, 1e-14,
               "rotation_from_quat normalises its input");
}

void AngularVelocityIsTheRpyDerivative() {
  // The analytic twist must be the exact derivative of the scripted pose,
  // which is what makes the torso twist trustworthy (clause A5 / risk 1).
  const Eigen::Vector3d rpy(0.15, -0.35, 0.62);
  const Eigen::Vector3d rpy_dot(0.7, -1.3, 0.45);
  const Eigen::Vector3d omega = AngularVelocityFromRpyRates(rpy, rpy_dot);

  const double h = 1e-7;
  const Eigen::Matrix3d forward = RotationFromRpy(rpy + h * rpy_dot);
  const Eigen::Matrix3d backward = RotationFromRpy(rpy - h * rpy_dot);
  const Eigen::Matrix3d derivative = (forward - backward) / (2.0 * h);
  // omega_hat = Rdot * R^T
  const Eigen::Matrix3d omega_hat = derivative * RotationFromRpy(rpy).transpose();
  const Eigen::Vector3d numeric(omega_hat(2, 1), omega_hat(0, 2), omega_hat(1, 0));
  CHECK_MATRIX(omega, numeric, 1e-6,
               "angular velocity matches the finite-difference derivative");
}

void SineRateIsTheOffsetDerivative() {
  const Eigen::Vector3d amplitude(0.18, 0.04, 0.05);
  const double frequency = 0.5;
  const double t = 0.37;
  const double h = 1e-7;
  const Eigen::Vector3d numeric =
      (SineOffset(t + h, amplitude, frequency) -
       SineOffset(t - h, amplitude, frequency)) /
      (2.0 * h);
  CHECK_MATRIX(SineRate(t, amplitude, frequency), numeric, 1e-6,
               "sine_rate is the derivative of sine_offset");
}

void SolveMatchesDirectInverse() {
  // Clause D7: the DLS system is solved, never inverted explicitly, but the
  // answer must still satisfy the system.
  M66 a = M66::Random();
  a = a * a.transpose() + 6.0 * M66::Identity();
  const Vector6 b = Vector6::Random();
  const Vector6 x = linalg::Solve6(a, b);
  CHECK_MATRIX(Vector6(a * x), b, 1e-12, "dgesv solves A x = b");
}

void PseudoInverseSatisfiesMoorePenrose() {
  // Clause D9: an honest undamped pseudo-inverse of a full-row-rank 6x7.
  const Matrix6x7 j = Matrix6x7::Random();
  const M76 pinv = linalg::Pinv(j);
  const M66 jp = j * pinv;

  CHECK_MATRIX(M66(jp * jp), jp, 1e-12, "J J+ is idempotent");
  CHECK_MATRIX(Matrix6x7(j * pinv * j), j, 1e-12, "J J+ J = J");
  CHECK_MATRIX(jp, M66(jp.transpose()), 1e-12, "J J+ is symmetric");

  // The null-space projector must annihilate the row space and be idempotent.
  const M77 projector = M77::Identity() - pinv * j;
  CHECK_MATRIX(Matrix6x7(j * projector), Matrix6x7::Zero(), 1e-12,
               "null-space projector annihilates the row space");
  CHECK_MATRIX(M77(projector * projector), projector, 1e-12,
               "null-space projector is idempotent");
}

void SingularValuesAreDescending() {
  const Matrix6x7 j = Matrix6x7::Random();
  const Vector6 sigma = linalg::SingularValues(j);
  for (int index = 1; index < 6; ++index) {
    CHECK_TRUE(sigma(index) <= sigma(index - 1) + 1e-15,
               "singular values are non-increasing");
  }
  CHECK_TRUE(sigma(5) >= 0.0, "singular values are non-negative");
}

void SolveGeneralHandlesMultipleRightHandSides() {
  Eigen::MatrixXd a(3, 3);
  a << 1.0, 1.0, 1.0, 3.0, 4.0, 5.0, 6.0, 12.0, 20.0;
  Eigen::MatrixXd b = Eigen::MatrixXd::Random(3, 4);
  const Eigen::MatrixXd x = linalg::SolveGeneral(a, b);
  CHECK_MATRIX(Eigen::MatrixXd(a * x), b, 1e-12,
               "SolveGeneral handles several right-hand sides");
}

}  // namespace

int main() {
  RpyComposition();
  QuaternionRoundTrip();
  AngularVelocityIsTheRpyDerivative();
  SineRateIsTheOffsetDerivative();
  SolveMatchesDirectInverse();
  PseudoInverseSatisfiesMoorePenrose();
  SingularValuesAreDescending();
  SolveGeneralHandlesMultipleRightHandSides();
  return srl::test::Finish("test_math");
}
