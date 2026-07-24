"""Compile validated structured trajectory intent into pure target sources.

This is the request-to-control seam used after Codex has resolved a user's
natural-language intent. It deliberately accepts structured data only; no
natural-language interpretation, MuJoCo access, or marker state belongs here.
"""

from dataclasses import dataclass

import numpy as np

from controller.state import FramedTarget, Pose, TargetFrame, Twist
from controller.trajectory import (
    HoldTrajectory,
    PeriodicTargetSource,
    TargetProgram,
    TargetProgramSegment,
    TrajectoryLimits,
    TrajectoryRateBounds,
    timed_circle_trajectory,
    timed_waypoint_trajectory,
)
from controller.transforms import rotation_from_rpy
from runtime_config import TargetTrajectoryConfig


@dataclass(frozen=True, slots=True)
class MaterializedTrajectory:
    """A complete one-arm trajectory ready for Runner composition."""

    source: object
    program: TargetProgram
    duration_s: float
    boundary_times_s: tuple[float, ...]
    rate_bounds: TrajectoryRateBounds


def _limits(config):
    return TrajectoryLimits(
        config.constraints.max_linear_speed_m_s,
        config.constraints.max_linear_acceleration_m_s2,
        config.constraints.max_angular_speed_rad_s,
        config.constraints.max_angular_acceleration_rad_s2,
    )


def _segment_source(reference_frame, start_pose, segment, limits, index):
    if segment.type == "hold":
        source = HoldTrajectory(
            FramedTarget(reference_frame, start_pose, Twist.zero()),
            segment.duration_s,
        )
        return source, start_pose

    if segment.type == "line":
        end_position = (
            start_pose.position_m
            + np.asarray(segment.displacement_m, dtype=float)
            if segment.displacement_m is not None
            else np.asarray(segment.end_position_m, dtype=float)
        )
        end_rotation = (
            start_pose.rotation
            if segment.end_rpy_rad is None
            else rotation_from_rpy(segment.end_rpy_rad)
        )
        end_pose = Pose(end_position, end_rotation)
        if (
            np.allclose(
                end_pose.position_m,
                start_pose.position_m,
                rtol=0.0,
                atol=1e-12,
            )
            and np.allclose(
                end_pose.rotation,
                start_pose.rotation,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise ValueError(
                f"line segment {index} has no translation or rotation"
            )
        source = timed_waypoint_trajectory(
            reference_frame,
            (start_pose, end_pose),
            (
                None
                if segment.duration_s is None
                else (segment.duration_s,)
            ),
            limits,
        )
        return source, end_pose

    if segment.type == "waypoints":
        if segment.offsets_m is not None:
            offsets = np.asarray(segment.offsets_m, dtype=float)
            if not np.allclose(
                offsets[0], 0.0, rtol=0.0, atol=1e-12
            ):
                raise ValueError(
                    f"waypoint segment {index} offsets_m must start at "
                    "[0, 0, 0]"
                )
            positions = start_pose.position_m + offsets
        else:
            positions = np.asarray(segment.positions_m, dtype=float)
            if not np.allclose(
                positions[0],
                start_pose.position_m,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"waypoint segment {index} positions_m must begin "
                    "at the previous segment endpoint"
                )
        if segment.rpy_rad is None:
            rotations = (start_pose.rotation,) * len(positions)
        else:
            rotations = tuple(
                rotation_from_rpy(value) for value in segment.rpy_rad
            )
            if not np.allclose(
                rotations[0],
                start_pose.rotation,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    f"waypoint segment {index} rpy_rad must begin at "
                    "the previous segment orientation"
                )
        poses = tuple(
            Pose(position, rotation)
            for position, rotation in zip(positions, rotations)
        )
        source = timed_waypoint_trajectory(
            reference_frame,
            poses,
            segment.durations_s,
            limits,
        )
        return source, poses[-1]

    end_rotation = (
        start_pose.rotation
        if segment.end_rpy_rad is None
        else rotation_from_rpy(segment.end_rpy_rad)
    )
    source = timed_circle_trajectory(
        reference_frame,
        start_pose,
        segment.radius_m,
        segment.normal,
        segment.start_direction,
        segment.duration_s,
        limits,
        revolutions=segment.revolutions,
        clockwise=segment.clockwise,
        end_rotation=end_rotation,
    )
    return source, Pose(start_pose.position_m, end_rotation)


def materialize_trajectory(config, start_pose):
    """Build and validate one structured trajectory from its start pose."""
    if not isinstance(config, TargetTrajectoryConfig):
        raise TypeError("config must be TargetTrajectoryConfig")
    if not isinstance(start_pose, Pose):
        raise TypeError("start_pose must be a Pose")
    reference_frame = TargetFrame(config.reference_frame)
    limits = _limits(config)
    segments = []
    current_pose = start_pose
    for index, segment_config in enumerate(config.segments):
        source, current_pose = _segment_source(
            reference_frame,
            current_pose,
            segment_config,
            limits,
            index,
        )
        segments.append(TargetProgramSegment(source.duration_s, source))
    program = TargetProgram(reference_frame, tuple(segments))
    bounds = program.maximum_rates()
    limits.validate(bounds)
    source = (
        PeriodicTargetSource(program, program.duration_s)
        if config.loop
        else program
    )
    return MaterializedTrajectory(
        source=source,
        program=program,
        duration_s=program.duration_s,
        boundary_times_s=program.boundary_times_s,
        rate_bounds=bounds,
    )
