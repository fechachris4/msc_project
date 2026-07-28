// Cylinder keep-out routing geometry. Contract clauses G1-G8.

#include "TestSupport.h"

#include "control/CylinderRouter.h"

using namespace srl;
using namespace srl::control;

namespace {

CylinderKeepout MakeKeepout() {
  CylinderKeepout keepout;
  keepout.enabled = true;
  keepout.center_xy_m = {0.0, 0.0};
  keepout.radius_m = 0.25;
  keepout.z_min_m = 0.0;
  keepout.z_max_m = 1.8;
  keepout.clearance_m = 0.10;
  keepout.waypoint_tolerance_m = 0.01;
  return keepout;
}

void InflationAndRouteRadius() {
  // Clause G2.
  const CylinderKeepout keepout = MakeKeepout();
  CHECK_CLOSE(keepout.obstacle_radius_m(), 0.35, 1e-15, "inflated radius");
  CHECK_CLOSE(keepout.obstacle_z_min_m(), -0.10, 1e-15, "inflated lower extent");
  CHECK_CLOSE(keepout.obstacle_z_max_m(), 1.90, 1e-15, "inflated upper extent");
  CHECK_CLOSE(keepout.route_radius_m(),
              0.35 / std::cos(kArcStepRad / 2.0) + kRoutePaddingM, 1e-15,
              "route radius compensates the 15-degree chord plus padding");
  CHECK_TRUE(keepout.route_radius_m() > keepout.obstacle_radius_m(),
             "arc waypoints sit outside the inflated obstacle");
}

void DisabledKeepoutRoutesDirect() {
  CylinderKeepout keepout = MakeKeepout();
  keepout.enabled = false;
  const CylinderRouter router(keepout);
  const Eigen::Vector3d start(-1.0, 0.0, 1.0);
  const Eigen::Vector3d target(1.0, 0.0, 1.0);

  CHECK_TRUE(!router.SegmentIntersects(start, target),
             "disabled keep-out never intersects");
  const CylinderRoute route = router.Plan(start, target);
  CHECK_TRUE(route.kind == CylinderRouteKind::Direct, "disabled route is direct");
  CHECK_TRUE(route.size() == 1, "disabled route has one waypoint");
  CHECK_MATRIX(route.waypoints[0], target, 0.0, "disabled route hits the target");
}

void ClearSegmentStaysDirect() {
  const CylinderRouter router(MakeKeepout());
  // Well outside the inflated radius, and never crossing the axis.
  const Eigen::Vector3d start(1.0, 1.0, 1.0);
  const Eigen::Vector3d target(1.0, -1.0, 1.0);
  CHECK_TRUE(!router.SegmentIntersects(start, target),
             "a clear segment does not intersect");
  const CylinderRoute route = router.Plan(start, target);
  CHECK_TRUE(route.kind == CylinderRouteKind::Direct, "clear route is direct");
  CHECK_TRUE(!route.target_adjusted, "clear target is not adjusted");
}

void SegmentBelowTheBandMisses() {
  // Clause G4: the segment is clipped to the inflated height band first.
  const CylinderRouter router(MakeKeepout());
  const Eigen::Vector3d start(-1.0, 0.0, -0.5);
  const Eigen::Vector3d target(1.0, 0.0, -0.5);
  CHECK_TRUE(!router.SegmentIntersects(start, target),
             "a segment entirely below the band misses");
}

void CrossingSegmentIsRouted() {
  // Clause G5: a segment straight through the axis must be routed around.
  const CylinderRouter router(MakeKeepout());
  const Eigen::Vector3d start(-1.0, 0.0, 1.0);
  const Eigen::Vector3d target(1.0, 0.0, 1.0);
  CHECK_TRUE(router.SegmentIntersects(start, target),
             "a segment through the axis intersects");

  const CylinderRoute route = router.Plan(start, target);
  CHECK_TRUE(route.kind != CylinderRouteKind::Direct,
             "a crossing segment is not routed direct");
  CHECK_TRUE(route.size() >= 2, "a routed path has several waypoints");
  CHECK_TRUE(route.size() <= kMaxWaypoints, "route respects the 32-point cap");
  CHECK_MATRIX(route.waypoints.back(), target, 1e-12,
               "the route still ends at the target");

  // Every arc waypoint must clear the inflated radius.
  for (std::size_t index = 0; index + 1 < route.waypoints.size(); ++index) {
    const Eigen::Vector2d radial = route.waypoints[index].head<2>();
    CHECK_TRUE(radial.norm() >= MakeKeepout().obstacle_radius_m() - 1e-9,
               "arc waypoints stay outside the inflated radius");
  }
  CHECK_TRUE(route.length_m > (target - start).norm(),
             "a detour is longer than the blocked straight line");
}

void TargetInsideIsMovedOutNotRefused() {
  // Clause G3.
  const CylinderKeepout keepout = MakeKeepout();
  const CylinderRouter router(keepout);
  const Eigen::Vector3d start(1.0, 0.0, 1.0);
  const Eigen::Vector3d inside(0.05, 0.0, 1.0);

  const CylinderRoute route = router.Plan(start, inside);
  CHECK_TRUE(route.target_adjusted, "an interior target is flagged as adjusted");
  CHECK_MATRIX(route.requested_target, inside, 0.0,
               "the originally requested target is retained");
  const Eigen::Vector2d effective_radial = route.effective_target.head<2>();
  CHECK_CLOSE(effective_radial.norm(), keepout.route_radius_m(), 1e-12,
              "the effective target sits on the route boundary");
  CHECK_CLOSE(route.effective_target(2), inside(2), 1e-15,
              "the adjustment is purely radial");
}

void FollowerAdvancesAndReports() {
  // Clause G7.
  CylinderRouteFollower follower(MakeKeepout());
  const Eigen::Vector3d start(-1.0, 0.0, 1.0);
  const Eigen::Vector3d target(1.0, 0.0, 1.0);
  follower.SetTarget(start, target);

  CHECK_TRUE(follower.index() == 0, "follower starts at the first waypoint");
  CHECK_TRUE(!follower.AtFinalWaypoint(), "a multi-point route is not final yet");

  const Eigen::Vector3d first = follower.Update(start);
  CHECK_MATRIX(first, follower.route().waypoints[0], 0.0,
               "the first update returns the first waypoint");

  // Standing exactly on waypoint 0 must advance to waypoint 1.
  const Eigen::Vector3d second = follower.Update(follower.route().waypoints[0]);
  CHECK_TRUE(follower.index() == 1, "reaching a waypoint advances the cursor");
  CHECK_MATRIX(second, follower.route().waypoints[1], 0.0,
               "the follower now serves the next waypoint");

  // Walking the whole route ends at the final waypoint and stays there.
  for (std::size_t step = 0; step < kMaxWaypoints + 2; ++step) {
    follower.Update(follower.route().waypoints[follower.index()]);
  }
  CHECK_TRUE(follower.AtFinalWaypoint(), "the follower settles on the last point");
}

void ResetPlansASinglePointRoute() {
  // Clause K3: seeding at the measured pose gives a valid one-waypoint route.
  CylinderRouteFollower follower(MakeKeepout());
  const Eigen::Vector3d here(0.8, 0.4, 1.2);
  follower.Reset(here);
  CHECK_TRUE(follower.route().size() == 1, "reset yields a single waypoint");
  CHECK_TRUE(follower.AtFinalWaypoint(), "a single-waypoint route is final");
  CHECK_MATRIX(follower.Update(here), here, 1e-12,
               "the seeded route returns the current position");
}

void RouteKindNames() {
  CHECK_TRUE(RouteKindName(CylinderRouteKind::Direct) == "direct", "direct name");
  CHECK_TRUE(RouteKindName(CylinderRouteKind::CounterClockwise) ==
                 "counter-clockwise",
             "counter-clockwise name");
  CHECK_TRUE(RouteKindName(CylinderRouteKind::Clockwise) == "clockwise",
             "clockwise name");
  CHECK_TRUE(RouteKindName(CylinderRouteKind::Over) == "over", "over name");
}

}  // namespace

int main() {
  InflationAndRouteRadius();
  DisabledKeepoutRoutesDirect();
  ClearSegmentStaysDirect();
  SegmentBelowTheBandMisses();
  CrossingSegmentIsRouted();
  TargetInsideIsMovedOutNotRefused();
  FollowerAdvancesAndReports();
  ResetPlansASinglePointRoute();
  RouteKindNames();
  return srl::test::Finish("test_router");
}
