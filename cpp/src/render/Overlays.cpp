#include "render/Overlays.h"

#include <array>
#include <cstdio>

#include "kinematics/LinkSpheres.h"

namespace srl::render {
namespace {

// Physical cylinder: warm and fairly solid. Inflated clearance boundary: cool
// and near-transparent, so the two are unmistakable at a glance.
constexpr std::array<float, 4> kPhysicalRgba{0.90f, 0.35f, 0.20f, 0.40f};
constexpr std::array<float, 4> kKeepoutClearanceRgba{0.25f, 0.60f, 0.95f, 0.13f};
constexpr std::array<float, 4> kRouteRgba{1.00f, 0.85f, 0.10f, 0.90f};
constexpr std::array<float, 4> kActiveWaypointRgba{0.10f, 1.00f, 0.35f, 0.95f};
constexpr double kRouteWaypointRadiusM = 0.018;
constexpr double kActiveWaypointRadiusM = 0.030;
constexpr double kRouteLineWidthM = 0.006;

constexpr std::array<float, 4> kPlanPathRgba{0.20f, 0.85f, 1.00f, 0.90f};
constexpr std::array<float, 4> kPlanKnotRgba{1.00f, 0.85f, 0.10f, 0.95f};
constexpr std::array<float, 4> kPlanStartRgba{0.10f, 1.00f, 0.35f, 0.95f};
constexpr std::array<float, 4> kPlanGoalRgba{1.00f, 0.30f, 0.75f, 0.95f};
constexpr double kPlanKnotRadiusM = 0.018;
constexpr double kPlanEndpointRadiusM = 0.028;
constexpr double kPlanLineWidthM = 0.006;
constexpr int kPlanPathSamples = 24;

constexpr std::array<float, 4> kHumanRgba{0.95f, 0.30f, 0.15f, 0.22f};
constexpr std::array<float, 4> kSafetyClearanceRgba{0.20f, 0.65f, 1.00f, 0.09f};
constexpr std::array<float, 4> kSafeSphereRgba{0.15f, 0.95f, 0.35f, 0.13f};
constexpr std::array<float, 4> kActiveSphereRgba{1.00f, 0.80f, 0.10f, 0.28f};
constexpr std::array<float, 4> kUnsafeSphereRgba{1.00f, 0.05f, 0.05f, 0.55f};
constexpr std::array<float, 4> kMountSphereRgba{0.55f, 0.55f, 0.60f, 0.10f};

// Claim the next free scene slot, or null when the scene is full.
mjvGeom* AddGeom(mjvScene* scene) {
  if (scene->ngeom >= scene->maxgeom) return nullptr;
  return &scene->geoms[scene->ngeom++];
}

// MuJoCo geom frames are row-major 3x3; Eigen defaults to column-major.
void RowMajor(const Eigen::Matrix3d& rotation, mjtNum* out) {
  Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> map(out);
  map = rotation;
}

void AddSphere(mjvScene* scene, const Eigen::Vector3d& position, double radius,
               const std::array<float, 4>& rgba) {
  mjvGeom* geom = AddGeom(scene);
  if (geom == nullptr) return;
  const mjtNum size[3] = {radius, 0.0, 0.0};
  const mjtNum position_values[3] = {position(0), position(1), position(2)};
  mjtNum identity[9];
  RowMajor(Eigen::Matrix3d::Identity(), identity);
  mjv_initGeom(geom, mjGEOM_SPHERE, size, position_values, identity,
               rgba.data());
}

void AddLine(mjvScene* scene, const Eigen::Vector3d& start,
             const Eigen::Vector3d& end, double width_m,
             const std::array<float, 4>& rgba) {
  mjvGeom* geom = AddGeom(scene);
  if (geom == nullptr) return;
  const mjtNum zero[3] = {0.0, 0.0, 0.0};
  mjtNum identity[9];
  RowMajor(Eigen::Matrix3d::Identity(), identity);
  mjv_initGeom(geom, mjGEOM_CAPSULE, zero, zero, identity, rgba.data());
  const mjtNum from[3] = {start(0), start(1), start(2)};
  const mjtNum to[3] = {end(0), end(1), end(2)};
  mjv_connector(geom, mjGEOM_CAPSULE, width_m, from, to);
}

void AddCylinderSized(mjvScene* scene, const Eigen::Vector2d& center_xy,
                      double z_low, double z_high, double radius,
                      const std::array<float, 4>& rgba) {
  mjvGeom* geom = AddGeom(scene);
  if (geom == nullptr) return;
  const mjtNum size[3] = {radius, 0.5 * (z_high - z_low), 0.0};
  const mjtNum center[3] = {center_xy(0), center_xy(1), 0.5 * (z_low + z_high)};
  mjtNum identity[9];
  RowMajor(Eigen::Matrix3d::Identity(), identity);
  mjv_initGeom(geom, mjGEOM_CYLINDER, size, center, identity, rgba.data());
}

void AddTorsoCylinder(mjvScene* scene, const Pose& torso_pose_world,
                      const config::HumanSafetyConfig& config, double expansion,
                      const std::array<float, 4>& rgba) {
  mjvGeom* geom = AddGeom(scene);
  if (geom == nullptr) return;
  const double z_low = config.z_min_torso_m - expansion;
  const double z_high = config.z_max_torso_m + expansion;
  const Eigen::Vector3d center_torso(config.center_xy_torso_m[0],
                                     config.center_xy_torso_m[1],
                                     0.5 * (z_low + z_high));
  const Eigen::Vector3d center_world =
      torso_pose_world.position_m + torso_pose_world.rotation * center_torso;
  const mjtNum size[3] = {config.radius_m + expansion, 0.5 * (z_high - z_low),
                          0.0};
  const mjtNum center[3] = {center_world(0), center_world(1), center_world(2)};
  mjtNum rotation[9];
  RowMajor(torso_pose_world.rotation, rotation);
  mjv_initGeom(geom, mjGEOM_CYLINDER, size, center, rotation, rgba.data());
}

std::string JoinSides(const std::vector<Side>& sides) {
  std::string text;
  for (std::size_t index = 0; index < sides.size(); ++index) {
    if (index > 0) text += ", ";
    text += std::string(SideName(sides[index]));
  }
  return text;
}

}  // namespace

std::string DescribeKeepout(const control::CylinderKeepout& keepout,
                            const std::vector<Side>& sides) {
  if (!keepout.enabled) {
    return "cylinder keep-out: DISABLED (direct targets; no routing, nothing "
           "drawn)";
  }
  std::array<char, 768> buffer{};
  std::snprintf(
      buffer.data(), buffer.size(),
      "cylinder keep-out: ENABLED  centre=(%.3f, %.3f) m  radius=%.3f m  "
      "z=[%.3f, %.3f] m  clearance=%.3f m  tolerance=%.3f m\n"
      "  frame: WORLD, axis=+z; one central cylinder shared by %s\n"
      "  END-EFFECTOR routing only - NOT whole-arm/link collision avoidance",
      keepout.center_xy_m[0], keepout.center_xy_m[1], keepout.radius_m,
      keepout.z_min_m, keepout.z_max_m, keepout.clearance_m,
      keepout.waypoint_tolerance_m, JoinSides(sides).c_str());
  return std::string(buffer.data());
}

std::string DescribeHumanSafety(const config::HumanSafetyConfig& config) {
  std::size_t exempt = 0;
  for (const auto& sphere : kinematics::kLinkSpheres) {
    if (sphere.mount_exempt) ++exempt;
  }
  std::array<char, 768> buffer{};
  std::snprintf(
      buffer.data(), buffer.size(),
      "whole-arm human safety: %s  frame=TORSO  axis=torso +z  radius=%.3f m  "
      "z=[%.3f, %.3f] m  clearance=%.3f m  control_margin=%.3f m\n"
      "  %zu mesh-covering spheres per arm; %zu active, %zu intentional "
      "mount-interface exemptions",
      config.enabled ? "ENABLED" : "DISABLED", config.radius_m,
      config.z_min_torso_m, config.z_max_torso_m, config.clearance_m,
      config.control_margin_m, kinematics::kLinkSpheres.size(),
      kinematics::kLinkSpheres.size() - exempt, exempt);
  return std::string(buffer.data());
}

int DrawKeepout(
    mjvScene* scene, const control::CylinderKeepout& keepout,
    const DualArm<std::optional<control::CylinderRouteStatus>>& routes,
    const std::vector<Side>& sides) {
  if (!keepout.enabled) return 0;
  const int before = scene->ngeom;

  AddCylinderSized(scene, keepout.center(), keepout.z_min_m, keepout.z_max_m,
                   keepout.radius_m, kPhysicalRgba);
  AddCylinderSized(scene, keepout.center(), keepout.obstacle_z_min_m(),
                   keepout.obstacle_z_max_m(), keepout.obstacle_radius_m(),
                   kKeepoutClearanceRgba);

  for (Side side : sides) {
    const auto& status = routes.for_arm(side);
    if (!status) continue;
    const auto& points = status->waypoints_world_m;
    for (std::size_t index = 0; index + 1 < points.size(); ++index) {
      AddLine(scene, points[index], points[index + 1],
              kRouteLineWidthM, kRouteRgba);
    }
    for (const auto& point : points) {
      AddSphere(scene, point, kRouteWaypointRadiusM, kRouteRgba);
    }
    AddSphere(scene, status->active_waypoint_world_m, kActiveWaypointRadiusM,
              kActiveWaypointRgba);
  }
  return scene->ngeom - before;
}

int DrawCartesianPlans(
    mjvScene* scene,
    const std::vector<planning::CartesianArmPlan>& plans) {
  const int before = scene->ngeom;
  for (const planning::CartesianArmPlan& plan : plans) {
    Eigen::Vector3d previous =
        plan.result.trajectory->Sample(0.0).pose.position_m;
    for (int index = 1; index < kPlanPathSamples; ++index) {
      const double time =
          plan.result.duration_s() * static_cast<double>(index) /
          static_cast<double>(kPlanPathSamples - 1);
      const Eigen::Vector3d point =
          plan.result.trajectory->Sample(time).pose.position_m;
      AddLine(scene, previous, point, kPlanLineWidthM, kPlanPathRgba);
      previous = point;
    }
    for (std::size_t index = 0;
         index < plan.result.knots_world_m.size(); ++index) {
      const bool first = index == 0;
      const bool last = index + 1 == plan.result.knots_world_m.size();
      AddSphere(scene, plan.result.knots_world_m[index],
                first || last ? kPlanEndpointRadiusM : kPlanKnotRadiusM,
                first ? kPlanStartRgba
                      : (last ? kPlanGoalRgba : kPlanKnotRgba));
    }
  }
  return scene->ngeom - before;
}

int DrawHumanSafety(mjvScene* scene, const PlantState& plant,
                    const DualArmControllerStates& controller_states,
                    const DualArmHumanSafetyStates& safety_states,
                    const config::HumanSafetyConfig& config,
                    const std::vector<Side>& sides) {
  if (!config.enabled) return 0;
  const int before = scene->ngeom;

  AddTorsoCylinder(scene, plant.torso_pose_world, config, 0.0, kHumanRgba);
  AddTorsoCylinder(scene, plant.torso_pose_world, config,
                   config.clearance_m + config.control_margin_m,
                   kSafetyClearanceRgba);

  for (Side side : sides) {
    const auto& points = controller_states.for_arm(side).link_safety_points;
    const auto& constraints = safety_states.for_arm(side).constraints;
    for (std::size_t index = 0; index < points.size(); ++index) {
      const auto& sphere = kinematics::kLinkSpheres[index];
      std::array<float, 4> color = kSafeSphereRgba;
      if (sphere.mount_exempt) {
        color = kMountSphereRgba;
      } else if (constraints.signed_clearance_m[index] < 0.0) {
        color = kUnsafeSphereRgba;
      } else if (constraints.active[index]) {
        color = kActiveSphereRgba;
      }
      AddSphere(scene, points.position_world_m[index], sphere.radius_m, color);
    }
  }
  return scene->ngeom - before;
}

}  // namespace srl::render
