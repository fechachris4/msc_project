// Dense linear algebra that must agree with NumPy numerically.
//
// NumPy 2.4.4 on this machine is built on Apple Accelerate, so np.linalg.solve
// is LAPACK dgesv and np.linalg.pinv/svd are dgesdd. These wrappers call the
// same routines rather than re-deriving them through Eigen's decompositions,
// which is what makes equations D7 and D9 reproducible instead of merely
// close. See docs/03-cpp-design.md §5.

#pragma once

#include "core/Types.h"

namespace srl::linalg {

// np.linalg.solve for a 6x6 system (the damped-least-squares solve).
Vector6 Solve6(const Eigen::Matrix<double, 6, 6>& a, const Vector6& b);

// np.linalg.pinv for a 6x7 matrix, including NumPy's singular-value cutoff.
Eigen::Matrix<double, kJoints, 6> Pinv(const Matrix6x7& a);

// np.linalg.svd(..., compute_uv=False) for a 6x7 matrix.
Vector6 SingularValues(const Matrix6x7& a);

// np.linalg.solve for a general square system with one or more right-hand
// sides (the trajectory quintic and minimum-jerk spline solves).
Eigen::MatrixXd SolveGeneral(const Eigen::MatrixXd& a, const Eigen::MatrixXd& b);

// NumPy's default pinv cutoff factor; asserted against the installed NumPy by
// tests/test_linalg.cpp fixtures.
inline constexpr double kPinvRcond = 1e-15;

}  // namespace srl::linalg
