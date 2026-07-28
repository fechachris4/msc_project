// Minimal assertion helpers. A dependency-free harness keeps the test build as
// reproducible as the library build; every check reports file:line and the
// contract clause it is defending.

#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <Eigen/Dense>

namespace srl::test {

inline int g_failures = 0;
inline int g_checks = 0;

inline void Report(bool ok, const char* file, int line, const std::string& what) {
  ++g_checks;
  if (ok) return;
  ++g_failures;
  std::fprintf(stderr, "FAIL %s:%d  %s\n", file, line, what.c_str());
}

inline void CheckTrue(bool value, const char* file, int line,
                      const std::string& what) {
  Report(value, file, line, what);
}

inline void CheckClose(double actual, double expected, double tolerance,
                       const char* file, int line, const std::string& what) {
  const bool ok = std::abs(actual - expected) <= tolerance;
  Report(ok, file, line,
         what + " (actual=" + std::to_string(actual) +
             " expected=" + std::to_string(expected) + ")");
}

template <typename A, typename B>
void CheckMatrixClose(const A& actual, const B& expected, double tolerance,
                      const char* file, int line, const std::string& what) {
  const double worst = (actual - expected).cwiseAbs().maxCoeff();
  Report(worst <= tolerance, file, line,
         what + " (max_abs_diff=" + std::to_string(worst) + ")");
}

// Assert that `body` throws std::exception.
template <typename Callable>
void CheckThrows(Callable body, const char* file, int line,
                 const std::string& what) {
  bool threw = false;
  try {
    body();
  } catch (const std::exception&) {
    threw = true;
  }
  Report(threw, file, line, what + " (expected a throw)");
}

inline int Finish(const char* suite) {
  if (g_failures == 0) {
    std::printf("PASS %s (%d checks)\n", suite, g_checks);
    return 0;
  }
  std::printf("FAIL %s (%d/%d checks failed)\n", suite, g_failures, g_checks);
  return 1;
}

}  // namespace srl::test

#define CHECK_TRUE(value, what) \
  ::srl::test::CheckTrue((value), __FILE__, __LINE__, (what))
#define CHECK_CLOSE(actual, expected, tolerance, what) \
  ::srl::test::CheckClose((actual), (expected), (tolerance), __FILE__, \
                          __LINE__, (what))
#define CHECK_MATRIX(actual, expected, tolerance, what)                 \
  ::srl::test::CheckMatrixClose((actual), (expected), (tolerance),      \
                                __FILE__, __LINE__, (what))
#define CHECK_THROWS(body, what) \
  ::srl::test::CheckThrows([&] { body; }, __FILE__, __LINE__, (what))
