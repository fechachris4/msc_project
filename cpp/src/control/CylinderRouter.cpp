#include "control/CylinderRouter.h"

#include <cmath>

namespace srl::control {
namespace {

// PositiveAngle: wrap to [0, 2*pi).
double PositiveAngle(double angle) {
  const double two_pi = 2.0 * M_PI;
  angle = std::fmod(angle, two_pi);
  return angle < 0.0 ? angle + two_pi : angle;
}

// Append: drop duplicates, respect the fixed capacity.
void Append(CylinderRoute& route, const Eigen::Vector3d& point) {
  if (!route.waypoints.empty() &&
      (route.waypoints.back() - point).norm() < kGeometryEpsilon) {
    return;
  }
  if (route.waypoints.size() < kMaxWaypoints) route.waypoints.push_back(point);
}

// RouteLength: start-to-first plus waypoint-to-waypoint.
double RouteLength(const Eigen::Vector3d& start, const CylinderRoute& route) {
  double length = 0.0;
  Eigen::Vector3d previous = start;
  for (const auto& point : route.waypoints) {
    length += (point - previous).norm();
    previous = point;
  }
  return length;
}

// RadialDirection: unit radial, with two degenerate fallbacks.
Eigen::Vector2d RadialDirection(const Eigen::Vector3d& point,
                                const Eigen::Vector2d& center,
                                const Eigen::Vector2d& fallback) {
  const Eigen::Vector2d radial = point.head<2>() - center;
  if (radial.norm() > kGeometryEpsilon) return radial / radial.norm();
  if (fallback.norm() > kGeometryEpsilon) return fallback / fallback.norm();
  return Eigen::Vector2d(1.0, 0.0);
}

// ArcCandidate: 15-degree arc around the cylinder at route_radius.
CylinderRoute ArcCandidate(const CylinderKeepout& keepout,
                           const Eigen::Vector3d& start,
                           const Eigen::Vector3d& target, double delta_angle,
                           CylinderRouteKind kind, double route_radius) {
  CylinderRoute route;
  route.kind = kind;
  route.effective_target = target;

  const Eigen::Vector2d center = keepout.center();
  const Eigen::Vector2d target_fallback = target.head<2>() - center;
  const Eigen::Vector2d start_direction =
      RadialDirection(start, center, target_fallback);
  const Eigen::Vector2d target_direction =
      RadialDirection(target, center, start_direction);
  const double start_angle = std::atan2(start_direction(1), start_direction(0));
  const double target_angle =
      std::atan2(target_direction(1), target_direction(0));

  // The caller supplies direction but not magnitude. Recompute the exact sweep
  // so rounding around +/-pi cannot flip the route.
  const double counter_clockwise = PositiveAngle(target_angle - start_angle);
  double sweep = 0.0;
  if (delta_angle >= 0.0) {
    sweep = counter_clockwise;
  } else {
    sweep = counter_clockwise == 0.0 ? 0.0 : counter_clockwise - 2.0 * M_PI;
  }
  const int segments =
      std::max(1, static_cast<int>(std::ceil(std::abs(sweep) / kArcStepRad)));

  Append(route, Eigen::Vector3d(center(0) + route_radius * std::cos(start_angle),
                                center(1) + route_radius * std::sin(start_angle),
                                start(2)));
  for (int index = 1; index <= segments; ++index) {
    const double fraction = static_cast<double>(index) / segments;
    const double angle = start_angle + fraction * sweep;
    Append(route,
           Eigen::Vector3d(center(0) + route_radius * std::cos(angle),
                           center(1) + route_radius * std::sin(angle),
                           start(2) + fraction * (target(2) - start(2))));
  }
  Append(route, target);
  route.length_m = RouteLength(start, route);
  return route;
}

// OverCandidate: rise, translate, descend.
CylinderRoute OverCandidate(const CylinderKeepout& keepout,
                            const Eigen::Vector3d& start,
                            const Eigen::Vector3d& target) {
  CylinderRoute route;
  route.kind = CylinderRouteKind::Over;
  route.effective_target = target;

  const double top_z = std::max(
      {keepout.z_max_m + keepout.clearance_m + 0.05, start(2), target(2)});
  Append(route, Eigen::Vector3d(start(0), start(1), top_z));
  Append(route, Eigen::Vector3d(target(0), target(1), top_z));
  Append(route, target);
  route.length_m = RouteLength(start, route);
  return route;
}

}  // namespace

std::string_view RouteKindName(CylinderRouteKind kind) {
  switch (kind) {
    case CylinderRouteKind::Direct: return "direct";
    case CylinderRouteKind::CounterClockwise: return "counter-clockwise";
    case CylinderRouteKind::Clockwise: return "clockwise";
    case CylinderRouteKind::Over: return "over";
  }
  return "direct";
}

CylinderKeepout KeepoutFromConfig(const config::CylinderKeepoutConfig& config) {
  CylinderKeepout keepout;
  keepout.enabled = config.cylinder_keepout_enabled;
  keepout.center_xy_m = {config.cylinder_keepout_center_x_m,
                         config.cylinder_keepout_center_y_m};
  keepout.radius_m = config.cylinder_keepout_radius_m;
  keepout.z_min_m = config.cylinder_keepout_z_min_m;
  keepout.z_max_m = config.cylinder_keepout_z_max_m;
  keepout.clearance_m = config.cylinder_keepout_clearance_m;
  keepout.waypoint_tolerance_m = config.cylinder_waypoint_tolerance_m;
  return keepout;
}

bool CylinderRouter::SegmentIntersects(const Eigen::Vector3d& start,
                                       const Eigen::Vector3d& end) const {
  if (!keepout_.enabled) return false;

  double t_min = 0.0;
  double t_max = 1.0;
  const double dz = end(2) - start(2);
  const double obstacle_z_min = keepout_.obstacle_z_min_m();
  const double obstacle_z_max = keepout_.obstacle_z_max_m();
  if (std::abs(dz) < kGeometryEpsilon) {
    if (start(2) < obstacle_z_min || start(2) > obstacle_z_max) return false;
  } else {
    const double t_a = (obstacle_z_min - start(2)) / dz;
    const double t_b = (obstacle_z_max - start(2)) / dz;
    t_min = std::max(0.0, std::min(t_a, t_b));
    t_max = std::min(1.0, std::max(t_a, t_b));
    if (t_min > t_max) return false;
  }

  const Eigen::Vector2d p0 = start.head<2>() - keepout_.center();
  const Eigen::Vector2d direction = end.head<2>() - start.head<2>();
  double nearest_t = t_min;
  const double squared = direction.dot(direction);
  if (squared > kGeometryEpsilon) {
    nearest_t = std::min(std::max(-p0.dot(direction) / squared, t_min), t_max);
  }
  const Eigen::Vector2d nearest = p0 + nearest_t * direction;
  const double obstacle_radius = keepout_.obstacle_radius_m();
  return nearest.dot(nearest) <= obstacle_radius * obstacle_radius;
}

CylinderRoute CylinderRouter::Plan(const Eigen::Vector3d& start,
                                   const Eigen::Vector3d& requested_target) const {
  CylinderRoute direct;
  direct.requested_target = requested_target;
  direct.effective_target = requested_target;

  if (!keepout_.enabled) {
    Append(direct, requested_target);
    direct.length_m = (requested_target - start).norm();
    return direct;
  }

  const double obstacle_radius = keepout_.obstacle_radius_m();
  const double route_radius = keepout_.route_radius_m();
  const Eigen::Vector2d center = keepout_.center();

  Eigen::Vector3d effective_target = requested_target;
  const bool target_in_height =
      requested_target(2) >= keepout_.obstacle_z_min_m() &&
      requested_target(2) <= keepout_.obstacle_z_max_m();
  const Eigen::Vector2d target_radial = requested_target.head<2>() - center;
  if (target_in_height &&
      target_radial.norm() <= obstacle_radius + kGeometryEpsilon) {
    // Inside the keep-out: move radially out to the route boundary rather than
    // refusing the request (clause G3).
    const Eigen::Vector2d start_fallback = start.head<2>() - center;
    const Eigen::Vector2d direction =
        RadialDirection(requested_target, center, start_fallback);
    effective_target(0) = center(0) + route_radius * direction(0);
    effective_target(1) = center(1) + route_radius * direction(1);
    direct.target_adjusted = true;
  }
  direct.effective_target = effective_target;

  if (!SegmentIntersects(start, effective_target)) {
    Append(direct, effective_target);
    direct.length_m = (effective_target - start).norm();
    return direct;
  }

  const CylinderRoute counter_clockwise =
      ArcCandidate(keepout_, start, effective_target, 1.0,
                   CylinderRouteKind::CounterClockwise, route_radius);
  const CylinderRoute clockwise =
      ArcCandidate(keepout_, start, effective_target, -1.0,
                   CylinderRouteKind::Clockwise, route_radius);
  const CylinderRoute over = OverCandidate(keepout_, start, effective_target);

  // Strict '<' in ccw, cw, over order so ties resolve identically (clause G5).
  CylinderRoute best = counter_clockwise;
  if (clockwise.length_m < best.length_m) best = clockwise;
  if (over.length_m < best.length_m) best = over;
  best.requested_target = requested_target;
  best.effective_target = effective_target;
  best.target_adjusted = direct.target_adjusted;
  return best;
}

void CylinderRouteFollower::Reset(const Eigen::Vector3d& current) {
  route_ = router_.Plan(current, current);
  index_ = 0;
}

void CylinderRouteFollower::SetTarget(const Eigen::Vector3d& current,
                                      const Eigen::Vector3d& requested_target) {
  route_ = router_.Plan(current, requested_target);
  index_ = 0;
}

Eigen::Vector3d CylinderRouteFollower::Update(const Eigen::Vector3d& current) {
  const double tolerance = router_.keepout().waypoint_tolerance_m;
  while (index_ + 1 < route_.size() &&
         (current - route_.waypoints[index_]).norm() <= tolerance) {
    ++index_;
  }
  if (route_.size() == 0) return current;
  return route_.waypoints[index_];
}

bool CylinderRouteFollower::AtFinalWaypoint() const {
  return route_.size() > 0 && index_ + 1 == route_.size();
}

}  // namespace srl::control
