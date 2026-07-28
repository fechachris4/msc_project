// Fixed-size Cartesian waypoint routing around one vertical keep-out cylinder.
//
// Pure geometry: no robot, no MuJoCo, no I/O. All inputs and outputs are metres
// in WORLD coordinates; the cylinder axis is world +z. Nothing here converts
// frames -- Runner supplies already-resolved world positions.
//
// SCOPE: routes the END EFFECTOR point only. Not whole-arm collision
// avoidance, and it never refuses a target -- a request inside the cylinder is
// moved radially to the route boundary instead. Contract clauses G1-G8.

#pragma once

#include <array>
#include <string_view>
#include <vector>

#include "config/RuntimeConfig.h"
#include "core/Types.h"

namespace srl::control {

// Constants shared with the hardware controller's CylinderRouter.cpp.
inline constexpr double kGeometryEpsilon = 1e-9;
inline constexpr double kArcStepRad = 15.0 * M_PI / 180.0;
inline constexpr double kRoutePaddingM = 0.01;
inline constexpr std::size_t kMaxWaypoints = 32;

enum class CylinderRouteKind { Direct, CounterClockwise, Clockwise, Over };

std::string_view RouteKindName(CylinderRouteKind kind);

// One finite world-vertical cylinder shared by both arms.
struct CylinderKeepout {
  bool enabled{false};
  std::array<double, 2> center_xy_m{0.0, 0.0};
  double radius_m{0.25};
  double z_min_m{0.0};
  double z_max_m{1.8};
  double clearance_m{0.10};
  double waypoint_tolerance_m{0.01};

  Eigen::Vector2d center() const {
    return Eigen::Vector2d(center_xy_m[0], center_xy_m[1]);
  }
  // Inflated radius the router must keep segments outside of.
  double obstacle_radius_m() const { return radius_m + clearance_m; }
  double obstacle_z_min_m() const { return z_min_m - clearance_m; }
  double obstacle_z_max_m() const { return z_max_m + clearance_m; }
  // Radius the arc waypoints sit on. The extra radial amount keeps the chords
  // between 15-degree waypoints outside the inflated obstacle.
  double route_radius_m() const {
    return obstacle_radius_m() / std::cos(kArcStepRad / 2.0) + kRoutePaddingM;
  }
};

CylinderKeepout KeepoutFromConfig(const config::CylinderKeepoutConfig& config);

struct CylinderRoute {
  std::vector<Eigen::Vector3d> waypoints;
  CylinderRouteKind kind{CylinderRouteKind::Direct};
  bool target_adjusted{false};
  Eigen::Vector3d requested_target{Eigen::Vector3d::Zero()};
  Eigen::Vector3d effective_target{Eigen::Vector3d::Zero()};
  double length_m{0.0};

  std::size_t size() const { return waypoints.size(); }
};

// Stateless planner.
class CylinderRouter {
 public:
  explicit CylinderRouter(CylinderKeepout keepout = {})
      : keepout_(std::move(keepout)) {}

  const CylinderKeepout& keepout() const { return keepout_; }

  // True when any part of the segment lies inside the inflated cylinder.
  bool SegmentIntersects(const Eigen::Vector3d& start,
                         const Eigen::Vector3d& end) const;

  // Direct when clear, otherwise the shortest of ccw / cw / over.
  CylinderRoute Plan(const Eigen::Vector3d& start,
                     const Eigen::Vector3d& requested_target) const;

 private:
  CylinderKeepout keepout_;
};

// Stateful cursor over one route. Route construction happens only on a new
// target; Update advances through the waypoints as the EE reaches each one.
class CylinderRouteFollower {
 public:
  explicit CylinderRouteFollower(CylinderKeepout keepout = {})
      : router_(std::move(keepout)) {}

  bool enabled() const { return router_.keepout().enabled; }
  const CylinderRoute& route() const { return route_; }
  std::size_t index() const { return index_; }

  void Reset(const Eigen::Vector3d& current);
  void SetTarget(const Eigen::Vector3d& current,
                 const Eigen::Vector3d& requested_target);
  Eigen::Vector3d Update(const Eigen::Vector3d& current);
  bool AtFinalWaypoint() const;

 private:
  CylinderRouter router_;
  CylinderRoute route_;
  std::size_t index_{0};
};

}  // namespace srl::control
