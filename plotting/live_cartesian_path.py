"""Optional nonblocking live publisher for EE Cartesian path snapshots.

The control process owns only a bounded deque and a size-one queue. Matplotlib
and its GUI event loop run in a separate spawned process.
"""

from collections import deque
from dataclasses import dataclass
import multiprocessing
from queue import Empty, Full

import numpy as np

from plotting.style import SIDE_COLOR


DEFAULT_BUFFER_SAMPLES = 4000
DEFAULT_REFRESH_EVERY = 50


@dataclass(frozen=True, slots=True)
class CartesianPathSnapshot:
    arms: tuple[str, ...]
    source_frames: dict[str, str]
    desired_world_m: dict[str, np.ndarray]
    measured_world_m: dict[str, np.ndarray]


class CartesianPathBuffer:
    """Bounded same-cycle desired/measured position history."""

    def __init__(self, arms, max_samples=DEFAULT_BUFFER_SAMPLES):
        selected = tuple(arms)
        if not selected or any(
            side not in ("right", "left") for side in selected
        ):
            raise ValueError("arms must be a non-empty subset of right/left")
        if len(set(selected)) != len(selected):
            raise ValueError("arms must not contain duplicates")
        capacity = int(max_samples)
        if capacity <= 0:
            raise ValueError("max_samples must be positive")
        self._arms = selected
        self._desired = {
            side: deque(maxlen=capacity) for side in selected
        }
        self._measured = {
            side: deque(maxlen=capacity) for side in selected
        }
        self._source_frames = {}

    @property
    def sample_count(self):
        return len(self._desired[self._arms[0]])

    def append(self, cycle):
        for side in self._arms:
            sampled = cycle.sampled_targets.for_arm(side)
            frame = sampled.reference_frame.value
            previous = self._source_frames.get(side)
            if previous is not None and previous != frame:
                raise ValueError(
                    f"{side} target source changed frame within live trace"
                )
            self._source_frames[side] = frame
            desired = (
                cycle.resolved_targets.for_arm(side)
                .pose_world.position_m
            )
            measured = (
                cycle.controller_states.for_arm(side)
                .ee_pose_world.position_m
            )
            self._desired[side].append(np.asarray(desired).copy())
            self._measured[side].append(np.asarray(measured).copy())

    def snapshot(self):
        return CartesianPathSnapshot(
            arms=self._arms,
            source_frames=dict(self._source_frames),
            desired_world_m={
                side: np.asarray(self._desired[side], dtype=float)
                for side in self._arms
            },
            measured_world_m={
                side: np.asarray(self._measured[side], dtype=float)
                for side in self._arms
            },
        )


def offer_latest_nonblocking(channel, value):
    """Replace a pending snapshot without ever waiting on the channel."""
    try:
        channel.put_nowait(value)
        return True
    except Full:
        pass
    try:
        channel.get_nowait()
    except Empty:
        return False
    try:
        channel.put_nowait(value)
        return True
    except Full:
        return False


def _set_line_3d(line, values):
    line.set_data(values[:, 0], values[:, 1])
    line.set_3d_properties(values[:, 2])


def _set_point_3d(line, value):
    line.set_data([value[0]], [value[1]])
    line.set_3d_properties([value[2]])


def _equal_axes(axis, points):
    lower = np.min(points, axis=0)
    upper = np.max(points, axis=0)
    centre = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 0.005)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _plot_worker(channel, arms):
    """Own the Matplotlib GUI in a process separate from control timing."""
    import matplotlib.pyplot as plt

    plt.ion()
    figure = plt.figure(
        figsize=(7.0 * len(arms), 6.2), layout="constrained"
    )
    artists = {}
    axes = {}
    for column, side in enumerate(arms, start=1):
        color = SIDE_COLOR[side]
        axis = figure.add_subplot(
            1, len(arms), column, projection="3d"
        )
        desired, = axis.plot(
            [], [], [], "k--", linewidth=1.4,
            label="desired (resolved world)",
        )
        measured, = axis.plot(
            [], [], [], color=color, linewidth=1.2,
            label="measured FK",
        )
        desired_start, = axis.plot(
            [], [], [], "k^", markersize=6, label="desired start"
        )
        desired_end, = axis.plot(
            [], [], [], "k*", markersize=8, label="desired end"
        )
        measured_start, = axis.plot(
            [], [], [], marker="o", color=color, linestyle="None",
            markersize=5, label="measured start",
        )
        measured_end, = axis.plot(
            [], [], [], marker="X", color=color, linestyle="None",
            markersize=6, label="measured end",
        )
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.set_zlabel("world z [m]")
        axis.legend(loc="best", fontsize=8)
        axes[side] = axis
        artists[side] = (
            desired,
            measured,
            desired_start,
            desired_end,
            measured_start,
            measured_end,
        )
    figure.suptitle(
        "Live simulation EE paths — controller-aligned, "
        "not hardware validation"
    )
    plt.show(block=False)

    while plt.fignum_exists(figure.number):
        try:
            snapshot = channel.get(timeout=0.05)
        except Empty:
            plt.pause(0.001)
            continue
        if snapshot is None:
            break
        while True:
            try:
                newer = channel.get_nowait()
            except Empty:
                break
            if newer is None:
                snapshot = None
                break
            snapshot = newer
        if snapshot is None:
            break

        for side in arms:
            desired = snapshot.desired_world_m[side]
            measured = snapshot.measured_world_m[side]
            if not len(desired):
                continue
            (
                desired_line,
                measured_line,
                desired_start,
                desired_end,
                measured_start,
                measured_end,
            ) = artists[side]
            _set_line_3d(desired_line, desired)
            _set_line_3d(measured_line, measured)
            _set_point_3d(desired_start, desired[0])
            _set_point_3d(desired_end, desired[-1])
            _set_point_3d(measured_start, measured[0])
            _set_point_3d(measured_end, measured[-1])
            _equal_axes(axes[side], np.vstack((desired, measured)))
            error_mm = 1000.0 * np.linalg.norm(
                desired[-1] - measured[-1]
            )
            axes[side].set_title(
                f"{side} · source: "
                f"{snapshot.source_frames.get(side, 'unknown')}\n"
                f"current |position error| = {error_mm:.1f} mm"
            )
        figure.canvas.draw_idle()
        plt.pause(0.001)
    plt.close(figure)


class LiveCartesianPathPublisher:
    """Bounded producer with decimated, drop-stale GUI publication."""

    def __init__(
        self,
        arms,
        max_samples=DEFAULT_BUFFER_SAMPLES,
        refresh_every=DEFAULT_REFRESH_EVERY,
    ):
        refresh = int(refresh_every)
        if refresh <= 0:
            raise ValueError("refresh_every must be positive")
        self._buffer = CartesianPathBuffer(arms, max_samples)
        self._refresh_every = refresh
        self._context = multiprocessing.get_context("spawn")
        self._channel = self._context.Queue(maxsize=1)
        self._process = self._context.Process(
            target=_plot_worker,
            args=(self._channel, tuple(arms)),
            daemon=True,
        )
        self._started = False
        self._closed = False

    def start(self):
        if self._closed:
            raise RuntimeError("live path publisher is closed")
        if self._started:
            raise RuntimeError("live path publisher is already started")
        import matplotlib

        backend = str(matplotlib.get_backend()).lower()
        if backend in {
            "agg", "pdf", "pgf", "ps", "svg", "template", "cairo"
        }:
            raise RuntimeError(
                f"Matplotlib backend {backend!r} is not interactive"
            )
        self._process.start()
        self._started = True

    def append(self, cycle):
        if not self._started:
            raise RuntimeError("live path publisher is not started")
        if not self._process.is_alive():
            raise RuntimeError("live path window process has exited")
        self._buffer.append(cycle)
        count = self._buffer.sample_count
        if count == 1 or count % self._refresh_every == 0:
            offer_latest_nonblocking(
                self._channel, self._buffer.snapshot()
            )

    def close(self):
        if self._closed:
            return
        if self._started:
            offer_latest_nonblocking(self._channel, None)
            self._process.join(timeout=1.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
        self._channel.close()
        self._started = False
        self._closed = True
