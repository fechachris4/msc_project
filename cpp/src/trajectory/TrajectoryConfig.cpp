#include "trajectory/TrajectoryConfig.h"

#include <stdexcept>

#include "math/Transforms.h"

namespace srl::trajectory {
namespace {

TrajectoryLimits LimitsOf(const config::TargetTrajectoryConfig& config) {
  TrajectoryLimits limits;
  limits.max_linear_speed_m_s = config.constraints.max_linear_speed_m_s;
  limits.max_linear_acceleration_m_s2 =
      config.constraints.max_linear_acceleration_m_s2;
  limits.max_angular_speed_rad_s = config.constraints.max_angular_speed_rad_s;
  limits.max_angular_acceleration_rad_s2 =
      config.constraints.max_angular_acceleration_rad_s2;
  return limits;
}

template <typename A, typename B>
bool AllClose(const A& left, const B& right, double atol) {
  return ((left - right).array().abs() <= atol).all();
}

struct SegmentResult {
  std::shared_ptr<TimedTargetSource> source;
  Pose end_pose;
};

SegmentResult BuildSegment(TargetFrame reference_frame, const Pose& start_pose,
                           const config::TrajectorySegmentConfig& segment,
                           const TrajectoryLimits& limits, std::size_t index) {
  switch (segment.type) {
    case config::SegmentType::Hold: {
      FramedTarget target;
      target.reference_frame = reference_frame;
      target.pose = start_pose;
      target.twist = Twist::Zero();
      return {std::make_shared<HoldTrajectory>(target, *segment.duration_s),
              start_pose};
    }

    case config::SegmentType::Line: {
      const Eigen::Vector3d end_position =
          segment.displacement_m
              ? Eigen::Vector3d(start_pose.position_m + *segment.displacement_m)
              : *segment.end_position_m;
      const Eigen::Matrix3d end_rotation =
          segment.end_rpy_rad
              ? transforms::RotationFromRpy(*segment.end_rpy_rad)
              : start_pose.rotation;
      const Pose end_pose(end_position, end_rotation);
      if (AllClose(end_pose.position_m, start_pose.position_m, 1e-12) &&
          AllClose(end_pose.rotation, start_pose.rotation, 1e-12)) {
        throw std::invalid_argument("line segment " + std::to_string(index) +
                                    " has no translation or rotation");
      }
      std::optional<std::vector<double>> durations;
      if (segment.duration_s) durations = std::vector<double>{*segment.duration_s};
      return {TimedWaypointTrajectory(reference_frame, {start_pose, end_pose},
                                      durations, limits),
              end_pose};
    }

    case config::SegmentType::Waypoints: {
      std::vector<Eigen::Vector3d> positions;
      if (segment.has_offsets) {
        if (!AllClose(segment.offsets_m.front(), Eigen::Vector3d::Zero(), 1e-12)) {
          throw std::invalid_argument("waypoint segment " +
                                      std::to_string(index) +
                                      " offsets_m must start at [0, 0, 0]");
        }
        for (const auto& offset : segment.offsets_m) {
          positions.push_back(start_pose.position_m + offset);
        }
      } else {
        positions = segment.positions_m;
        if (!AllClose(positions.front(), start_pose.position_m, 1e-9)) {
          throw std::invalid_argument(
              "waypoint segment " + std::to_string(index) +
              " positions_m must begin at the previous segment endpoint");
        }
      }

      std::vector<Eigen::Matrix3d> rotations;
      if (!segment.has_rpy) {
        rotations.assign(positions.size(), start_pose.rotation);
      } else {
        for (const auto& rpy : segment.rpy_rad) {
          rotations.push_back(transforms::RotationFromRpy(rpy));
        }
        if (!AllClose(rotations.front(), start_pose.rotation, 1e-9)) {
          throw std::invalid_argument(
              "waypoint segment " + std::to_string(index) +
              " rpy_rad must begin at the previous segment orientation");
        }
      }

      std::vector<Pose> poses;
      for (std::size_t point = 0; point < positions.size(); ++point) {
        poses.emplace_back(positions[point], rotations[point]);
      }
      std::optional<std::vector<double>> durations;
      if (segment.has_durations) durations = segment.durations_s;
      return {TimedWaypointTrajectory(reference_frame, poses, durations, limits),
              poses.back()};
    }

    case config::SegmentType::Circle: {
      const Eigen::Matrix3d end_rotation =
          segment.end_rpy_rad
              ? transforms::RotationFromRpy(*segment.end_rpy_rad)
              : start_pose.rotation;
      auto source = TimedCircleTrajectory(
          reference_frame, start_pose, *segment.radius_m, *segment.normal,
          *segment.start_direction, segment.duration_s, limits,
          *segment.revolutions, *segment.clockwise, end_rotation);
      return {source, Pose(start_pose.position_m, end_rotation)};
    }
  }
  throw std::invalid_argument("unsupported trajectory segment type");
}

}  // namespace

MaterializedTrajectory MaterializeTrajectory(
    const config::TargetTrajectoryConfig& config, const Pose& start_pose) {
  const TargetFrame reference_frame =
      TargetFrameFromName(config.reference_frame);
  const TrajectoryLimits limits = LimitsOf(config);

  std::vector<TargetProgramSegment> segments;
  Pose current_pose = start_pose;
  for (std::size_t index = 0; index < config.segments.size(); ++index) {
    SegmentResult result = BuildSegment(reference_frame, current_pose,
                                        config.segments[index], limits, index);
    current_pose = result.end_pose;
    segments.push_back({result.source->duration_s(), result.source});
  }

  auto program =
      std::make_shared<TargetProgram>(reference_frame, std::move(segments));
  const TrajectoryRateBounds bounds = program->MaximumRates();
  limits.Validate(bounds);

  MaterializedTrajectory materialized;
  materialized.program = program;
  materialized.source =
      config.loop ? std::static_pointer_cast<TimedTargetSource>(
                        std::make_shared<PeriodicTargetSource>(
                            program, program->duration_s()))
                  : std::static_pointer_cast<TimedTargetSource>(program);
  materialized.duration_s = program->duration_s();
  materialized.boundary_times_s = program->boundary_times_s();
  materialized.rate_bounds = bounds;
  return materialized;
}

}  // namespace srl::trajectory
