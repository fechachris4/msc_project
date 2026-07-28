#include "math/LinAlg.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

// Match the LAPACK implementation NumPy is built against. NumPy's macOS
// wheels use Accelerate's *new* LAPACK (macOS 13.3+), not the legacy
// LAPACK 3.2.1 interface, so ask for the same one.
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

namespace srl::linalg {
namespace {

using LapackInt = __LAPACK_int;

}  // namespace

Vector6 Solve6(const Eigen::Matrix<double, 6, 6>& a, const Vector6& b) {
  // Eigen is column-major by default, which is already LAPACK's layout.
  std::array<double, 36> matrix{};
  std::copy(a.data(), a.data() + 36, matrix.begin());
  Vector6 rhs = b;

  LapackInt n = 6;
  LapackInt nrhs = 1;
  LapackInt lda = 6;
  LapackInt ldb = 6;
  LapackInt info = 0;
  std::array<LapackInt, 6> pivots{};

  dgesv_(&n, &nrhs, matrix.data(), &lda, pivots.data(), rhs.data(), &ldb,
         &info);
  if (info != 0) {
    throw std::runtime_error("dgesv failed with info=" + std::to_string(info));
  }
  return rhs;
}

Eigen::MatrixXd SolveGeneral(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b) {
  if (a.rows() != a.cols() || a.rows() != b.rows()) {
    throw std::invalid_argument("SolveGeneral requires a square A and matching B");
  }
  // Eigen is column-major, which is already LAPACK's layout.
  Eigen::MatrixXd matrix = a;
  Eigen::MatrixXd rhs = b;

  LapackInt n = static_cast<LapackInt>(a.rows());
  LapackInt nrhs = static_cast<LapackInt>(b.cols());
  LapackInt lda = n;
  LapackInt ldb = n;
  LapackInt info = 0;
  std::vector<LapackInt> pivots(static_cast<std::size_t>(n));

  dgesv_(&n, &nrhs, matrix.data(), &lda, pivots.data(), rhs.data(), &ldb, &info);
  if (info != 0) {
    throw std::runtime_error("dgesv failed with info=" + std::to_string(info));
  }
  return rhs;
}

namespace {

// Thin SVD of a 6x7 matrix via dgesdd (jobz='S'), the routine NumPy uses.
// Returns U (6x6), singular values (6), and V^T (6x7), all column-major.
struct ThinSvd {
  Eigen::Matrix<double, 6, 6> u;
  Vector6 singular_values;
  Eigen::Matrix<double, 6, kJoints> vt;
};

ThinSvd ComputeThinSvd(const Matrix6x7& a, bool want_vectors) {
  ThinSvd result{};
  // dgesdd overwrites its input.
  Eigen::Matrix<double, 6, kJoints> work_matrix = a;

  char jobz = want_vectors ? 'S' : 'N';
  LapackInt m = 6;
  LapackInt n = kJoints;
  LapackInt lda = 6;
  LapackInt ldu = 6;
  LapackInt ldvt = 6;
  LapackInt info = 0;

  // Workspace query.
  double optimal_work = 0.0;
  LapackInt lwork = -1;
  std::array<LapackInt, 8 * 6> iwork{};
  dgesdd_(&jobz, &m, &n, work_matrix.data(), &lda,
          result.singular_values.data(), result.u.data(), &ldu,
          result.vt.data(), &ldvt, &optimal_work, &lwork, iwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("dgesdd workspace query failed, info=" +
                             std::to_string(info));
  }

  lwork = static_cast<LapackInt>(optimal_work);
  std::vector<double> work(static_cast<std::size_t>(lwork));
  dgesdd_(&jobz, &m, &n, work_matrix.data(), &lda,
          result.singular_values.data(), result.u.data(), &ldu,
          result.vt.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
  if (info != 0) {
    throw std::runtime_error("dgesdd failed with info=" +
                             std::to_string(info));
  }
  return result;
}

}  // namespace

Eigen::Matrix<double, kJoints, 6> Pinv(const Matrix6x7& a) {
  const ThinSvd svd = ComputeThinSvd(a, /*want_vectors=*/true);

  // NumPy: cutoff = rcond * max(s); reciprocal where s > cutoff, else zero.
  const double cutoff = kPinvRcond * svd.singular_values.maxCoeff();
  Vector6 inverse_values = Vector6::Zero();
  for (int index = 0; index < 6; ++index) {
    const double value = svd.singular_values(index);
    inverse_values(index) = value > cutoff ? 1.0 / value : 0.0;
  }

  // NumPy: matmul(transpose(vt), multiply(s[..., newaxis], transpose(u))).
  const Eigen::Matrix<double, 6, 6> scaled_ut =
      inverse_values.asDiagonal() * svd.u.transpose();
  return svd.vt.transpose() * scaled_ut;
}

Vector6 SingularValues(const Matrix6x7& a) {
  return ComputeThinSvd(a, /*want_vectors=*/false).singular_values;
}

}  // namespace srl::linalg
