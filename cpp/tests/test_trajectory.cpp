// Trajectory generation. Contract clauses H1-H8.

#include "TestSupport.h"

#include "math/Transforms.h"
#include "trajectory/Trajectory.h"

using namespace srl;
using namespace srl::trajectory;

namespace {

Pose MakePose(const Eigen::Vector3d& position, const Eigen::Vector3d& rpy) {
  return Pose(position, transforms::RotationFromRpy(rpy));
}

void MinimumJerkEndpoints() {
  // Clause H3: rest-to-rest with unit progress at the end.
  double progress = 0.0, rate = 0.0, acceleration = 0.0;
  MinimumJerk(0.0, 2.0, progress, rate, acceleration);
  CHECK_CLOSE(progress, 0.0, 1e-15, "progress starts at zero");
  CHECK_CLOSE(rate, 0.0, 1e-15, "rate starts at zero");
  CHECK_CLOSE(acceleration, 0.0, 1e-15, "acceleration starts at zero");

  MinimumJerk(1.0, 2.0, progress, rate, acceleration);
  CHECK_CLOSE(progress, 1.0, 1e-15, "progress ends at one");
  CHECK_CLOSE(rate, 0.0, 1e-15, "rate ends at zero");
  CHECK_CLOSE(acceleration, 0.0, 1e-14, "acceleration ends at zero");

  MinimumJerk(0.5, 2.0, progress, rate, acceleration);
  CHECK_CLOSE(progress, 0.5, 1e-15, "progress is symmetric at the midpoint");
}

void RotationVectorRoundTrip() {
  // Clause H2: the port keeps trajectory.py's own principal log, so exp(log(R))
  // must return R across ordinary, tiny and near-pi rotations.
  //
  // Tolerances are per-case on purpose. Near pi the log is intrinsically
  // ill-conditioned -- cos(pi - e) = -1 + e^2/2, so acos recovers e only to
  // about sqrt(eps). The Python implementation measures 1.665e-16, 0.0 and
  // 1.0e-09 for these three cases respectively; the port must be no worse,
  // and the loose third bound records a real property of the algorithm rather
  // than a weakness of the port.
  struct Case {
    Eigen::Vector3d rpy;
    double tolerance;
    const char* what;
  };
  const Case cases[] = {
      {Eigen::Vector3d(0.3, -0.4, 0.9), 1e-14, "ordinary rotation"},
      {Eigen::Vector3d(1e-10, 0.0, 0.0), 1e-15, "tiny-angle branch"},
      {Eigen::Vector3d(0.0, 0.0, M_PI - 1e-9), 1e-8, "near-pi branch"},
  };
  for (const auto& item : cases) {
    const Eigen::Matrix3d rotation = transforms::RotationFromRpy(item.rpy);
    const Eigen::Vector3d log = RotationVector(rotation);
    CHECK_TRUE(log.norm() <= M_PI + 1e-9, "log norm stays within [0, pi]");
    CHECK_MATRIX(RotationFromVector(log), rotation, item.tolerance,
                 std::string("exp(log(R)) reproduces R: ") + item.what);
  }
}

void ExactPiRotationIsHandled() {
  // The near-pi branch is where a naive log divides by sin(pi) = 0.
  const Eigen::Matrix3d rotation =
      Eigen::AngleAxisd(M_PI, Eigen::Vector3d(0.0, 0.0, 1.0)).toRotationMatrix();
  const Eigen::Vector3d log = RotationVector(rotation);
  CHECK_CLOSE(log.norm(), M_PI, 1e-7, "a pi rotation has magnitude pi");
  CHECK_MATRIX(RotationFromVector(log), rotation, 1e-7,
               "the pi branch still round-trips");
}

void CleanSmallZeroesTinyValues() {
  // Clause H6.
  const Eigen::Vector3d value(1e-15, 1e-13, -0.5);
  const Eigen::Vector3d cleaned = CleanSmall(value);
  CHECK_CLOSE(cleaned(0), 0.0, 0.0, "sub-threshold value zeroed");
  CHECK_CLOSE(cleaned(1), 1e-13, 0.0, "above-threshold value retained");
  CHECK_CLOSE(cleaned(2), -0.5, 0.0, "large value untouched");
}

void HoldIsStationary() {
  FramedTarget target;
  target.reference_frame = TargetFrame::World;
  target.pose = MakePose(Eigen::Vector3d(0.4, 0.1, 1.2), Eigen::Vector3d::Zero());
  const HoldTrajectory hold(target, 3.0);

  for (double t : {0.0, 1.5, 3.0, 9.0}) {
    const auto sample = hold.SampleKinematics(t);
    CHECK_MATRIX(sample.target.pose.position_m, target.pose.position_m, 0.0,
                 "hold does not move");
    CHECK_MATRIX(sample.target.twist.linear_m_s, Eigen::Vector3d::Zero(), 0.0,
                 "hold has zero linear twist");
    CHECK_MATRIX(sample.linear_acceleration_m_s2, Eigen::Vector3d::Zero(), 0.0,
                 "hold has zero acceleration");
  }
  CHECK_THROWS(HoldTrajectory(target, 0.0), "non-positive hold duration rejected");
}

void WaypointSplineHitsItsKnotsAtRest() {
  // Clauses H1, H5: exact knots, rest endpoints, clamped outside the span.
  const std::vector<Pose> poses = {
      MakePose(Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d::Zero()),
      MakePose(Eigen::Vector3d(0.2, 0.1, 1.1), Eigen::Vector3d::Zero()),
      MakePose(Eigen::Vector3d(0.5, -0.1, 0.9), Eigen::Vector3d::Zero()),
  };
  const CartesianWaypointTrajectory trajectory =
      CartesianWaypointTrajectory::FromDurations(TargetFrame::World, poses,
                                                 {2.0, 3.0});
  CHECK_CLOSE(trajectory.duration_s(), 5.0, 1e-15, "durations accumulate");

  const auto start = trajectory.SampleKinematics(0.0);
  CHECK_MATRIX(start.target.pose.position_m, poses[0].position_m, 1e-12,
               "spline starts at the first knot");
  CHECK_MATRIX(start.target.twist.linear_m_s, Eigen::Vector3d::Zero(), 1e-12,
               "spline starts at rest");

  const auto middle = trajectory.SampleKinematics(2.0);
  CHECK_MATRIX(middle.target.pose.position_m, poses[1].position_m, 1e-9,
               "spline passes exactly through the interior knot");

  const auto end = trajectory.SampleKinematics(5.0);
  CHECK_MATRIX(end.target.pose.position_m, poses[2].position_m, 1e-12,
               "spline ends at the last knot");
  CHECK_MATRIX(end.target.twist.linear_m_s, Eigen::Vector3d::Zero(), 1e-12,
               "spline ends at rest");

  const auto beyond = trajectory.SampleKinematics(50.0);
  CHECK_MATRIX(beyond.target.pose.position_m, poses[2].position_m, 1e-12,
               "sampling past the end clamps to the final pose");

  // C1 continuity across the interior knot.
  const double h = 1e-6;
  const auto before = trajectory.SampleKinematics(2.0 - h);
  const auto after = trajectory.SampleKinematics(2.0 + h);
  CHECK_MATRIX(before.target.twist.linear_m_s, after.target.twist.linear_m_s,
               1e-5, "velocity is continuous across the knot");
}

void WaypointRejectsBadTiming() {
  const std::vector<CartesianWaypoint> single = {
      {0.0, MakePose(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())}};
  CHECK_THROWS(CartesianWaypointTrajectory(TargetFrame::World, single),
               "a single waypoint is rejected");

  const std::vector<CartesianWaypoint> unordered = {
      {0.0, MakePose(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())},
      {2.0, MakePose(Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero())},
      {1.0, MakePose(Eigen::Vector3d::UnitY(), Eigen::Vector3d::Zero())}};
  CHECK_THROWS(CartesianWaypointTrajectory(TargetFrame::World, unordered),
               "non-increasing waypoint times are rejected");

  const std::vector<CartesianWaypoint> late_start = {
      {1.0, MakePose(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero())},
      {2.0, MakePose(Eigen::Vector3d::UnitX(), Eigen::Vector3d::Zero())}};
  CHECK_THROWS(CartesianWaypointTrajectory(TargetFrame::World, late_start),
               "a non-zero first waypoint time is rejected");
}

void CircleIsGeometricallyExact() {
  // Clause H4.
  const Pose start = MakePose(Eigen::Vector3d(0.5, 0.0, 1.0),
                              Eigen::Vector3d::Zero());
  const CircleTrajectory circle(TargetFrame::World, start, 0.1,
                                Eigen::Vector3d(0.0, 0.0, 1.0),
                                Eigen::Vector3d(1.0, 0.0, 0.0), 4.0, 1, false,
                                std::nullopt);

  const Eigen::Vector3d centre = circle.centre_m();
  CHECK_MATRIX(centre, Eigen::Vector3d(0.4, 0.0, 1.0), 1e-15,
               "centre sits one radius back along the start direction");

  for (int step = 0; step <= 40; ++step) {
    const double t = 4.0 * step / 40.0;
    const auto sample = circle.SampleKinematics(t);
    const double radius = (sample.target.pose.position_m - centre).norm();
    CHECK_CLOSE(radius, 0.1, 1e-12, "every sample lies on the circle");
  }

  const auto begin = circle.SampleKinematics(0.0);
  CHECK_MATRIX(begin.target.pose.position_m, start.position_m, 1e-15,
               "the circle starts at the start pose");
  const auto finish = circle.SampleKinematics(4.0);
  CHECK_MATRIX(finish.target.pose.position_m, start.position_m, 1e-12,
               "one revolution returns to the start pose");
  CHECK_MATRIX(finish.target.twist.linear_m_s, Eigen::Vector3d::Zero(), 1e-12,
               "the circle ends at rest");

  CHECK_THROWS(CircleTrajectory(TargetFrame::World, start, 0.1,
                                Eigen::Vector3d(0.0, 0.0, 1.0),
                                Eigen::Vector3d(0.0, 0.0, 1.0), 4.0, 1, false,
                                std::nullopt),
               "a non-orthogonal normal/start_direction pair is rejected");
}

void ProgramRequiresContinuityAndRest() {
  // Clauses H7, H8.
  FramedTarget target;
  target.reference_frame = TargetFrame::World;
  target.pose = MakePose(Eigen::Vector3d(0.3, 0.0, 1.0), Eigen::Vector3d::Zero());
  auto hold = std::make_shared<HoldTrajectory>(target, 2.0);

  const TargetProgram program(TargetFrame::World, {{2.0, hold}});
  CHECK_CLOSE(program.duration_s(), 2.0, 1e-15, "program duration");
  const auto sample = program.SampleKinematics(1.0);
  CHECK_MATRIX(sample.target.pose.position_m, target.pose.position_m, 0.0,
               "program forwards to its segment");

  // A discontinuous jump between two holds must be rejected.
  FramedTarget moved = target;
  moved.pose = MakePose(Eigen::Vector3d(0.9, 0.0, 1.0), Eigen::Vector3d::Zero());
  auto elsewhere = std::make_shared<HoldTrajectory>(moved, 2.0);
  CHECK_THROWS(
      TargetProgram(TargetFrame::World, {{2.0, hold}, {2.0, elsewhere}}),
      "a discontinuous program boundary is rejected");

  // A periodic wrapper needs the source to close continuously; a hold does.
  auto periodic = std::make_shared<TargetProgram>(
      TargetFrame::World, std::vector<TargetProgramSegment>{{2.0, hold}});
  const PeriodicTargetSource repeat(periodic, 2.0);
  const auto wrapped = repeat.SampleKinematics(5.0);  // 5 mod 2 = 1
  CHECK_MATRIX(wrapped.target.pose.position_m, target.pose.position_m, 0.0,
               "periodic sampling wraps modulo the period");

  // A source that does not return to its start cannot be made periodic.
  const std::vector<Pose> open_path = {
      MakePose(Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d::Zero()),
      MakePose(Eigen::Vector3d(0.4, 0.0, 1.0), Eigen::Vector3d::Zero())};
  auto line = std::make_shared<CartesianWaypointTrajectory>(
      CartesianWaypointTrajectory::FromDurations(TargetFrame::World, open_path,
                                                 {2.0}));
  CHECK_THROWS(PeriodicTargetSource(line, 2.0),
               "a source that does not close is rejected as periodic");
}

void LimitsScaleAndValidate() {
  // Clauses H9, H10.
  TrajectoryLimits limits;
  limits.max_linear_speed_m_s = 0.5;

  TrajectoryRateBounds bounds;
  bounds.max_linear_speed_m_s = 1.0;
  CHECK_CLOSE(limits.RequiredTimeScale(bounds), 2.0, 1e-15,
              "twice the speed limit needs twice the time");
  CHECK_THROWS(limits.Validate(bounds), "an over-speed trajectory is rejected");

  bounds.max_linear_speed_m_s = 0.25;
  CHECK_CLOSE(limits.RequiredTimeScale(bounds), 1.0, 1e-15,
              "an under-limit trajectory needs no stretching");

  // Acceleration limits scale with the square root.
  TrajectoryLimits acceleration_only;
  acceleration_only.max_linear_acceleration_m_s2 = 1.0;
  TrajectoryRateBounds fast;
  fast.max_linear_acceleration_m_s2 = 4.0;
  CHECK_CLOSE(acceleration_only.RequiredTimeScale(fast), 2.0, 1e-15,
              "acceleration overshoot scales as the square root");

  // Derived timing: no durations given, so the limits set them.
  const std::vector<Pose> poses = {
      MakePose(Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Vector3d::Zero()),
      MakePose(Eigen::Vector3d(1.0, 0.0, 1.0), Eigen::Vector3d::Zero())};
  const auto derived =
      TimedWaypointTrajectory(TargetFrame::World, poses, std::nullopt, limits);
  CHECK_TRUE(derived->duration_s() > 0.0, "derived timing produces a duration");
  CHECK_TRUE(derived->MaximumRates().max_linear_speed_m_s <=
                 0.5 * (1.0 + 1e-9),
             "derived timing respects the speed limit");
}

}  // namespace

int main() {
  MinimumJerkEndpoints();
  RotationVectorRoundTrip();
  ExactPiRotationIsHandled();
  CleanSmallZeroesTinyValues();
  HoldIsStationary();
  WaypointSplineHitsItsKnotsAtRest();
  WaypointRejectsBadTiming();
  CircleIsGeometricallyExact();
  ProgramRequiresContinuityAndRest();
  LimitsScaleAndValidate();
  return srl::test::Finish("test_trajectory");
}
