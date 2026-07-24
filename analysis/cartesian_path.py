"""World-frame desired-versus-measured Cartesian path figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plotting.style import C_LEFT, C_RIGHT


MAX_RENDER_SAMPLES = 5000
_SIDE_COLOR = {"right": C_RIGHT, "left": C_LEFT}


def _render_indices(sample_count):
    if sample_count <= MAX_RENDER_SAMPLES:
        return np.arange(sample_count)
    return np.unique(
        np.linspace(
            0, sample_count - 1, MAX_RENDER_SAMPLES, dtype=int
        )
    )


def _equal_world_axes(axis, points):
    finite = points[np.all(np.isfinite(points), axis=1)]
    if not finite.size:
        raise ValueError("Cartesian path contains no finite points")
    lower = np.min(finite, axis=0)
    upper = np.max(finite, axis=0)
    centre = 0.5 * (lower + upper)
    radius = max(0.5 * float(np.max(upper - lower)), 0.005)
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1.0, 1.0, 1.0))


def _positions(log, side):
    fields = log.arm_data[side]
    try:
        desired = np.asarray(
            fields["target_position_world_m"], dtype=float
        )
        measured = np.asarray(
            fields["ee_position_world_m"], dtype=float
        )
    except KeyError as error:
        raise ValueError(
            "log does not contain aligned Cartesian path fields"
        ) from error
    if (
        desired.ndim != 2
        or measured.ndim != 2
        or desired.shape[1:] != (3,)
        or measured.shape != desired.shape
        or desired.shape[0] == 0
    ):
        raise ValueError(
            "desired and measured Cartesian paths must be non-empty Nx3 arrays"
        )
    return desired, measured


def make_cartesian_path_figure(log, path):
    """Plot full-run desired and measured EE positions in resolved world.

    Raw logs are never downsampled. Only the rendered lines are bounded to
    ``MAX_RENDER_SAMPLES`` while exact first/last samples remain marked.
    """
    arms = tuple(log.arms)
    if not arms:
        raise ValueError("log must contain at least one arm")
    fig = plt.figure(
        figsize=(7.0 * len(arms), 6.2), layout="constrained"
    )

    for column, side in enumerate(arms, start=1):
        desired, measured = _positions(log, side)
        indices = _render_indices(len(desired))
        desired_render = desired[indices]
        measured_render = measured[indices]
        color = _SIDE_COLOR.get(side, "tab:blue")
        axis = fig.add_subplot(1, len(arms), column, projection="3d")

        axis.plot(
            desired_render[:, 0],
            desired_render[:, 1],
            desired_render[:, 2],
            color="black",
            linestyle="--",
            linewidth=1.5,
            label="desired (resolved world)",
        )
        axis.plot(
            measured_render[:, 0],
            measured_render[:, 1],
            measured_render[:, 2],
            color=color,
            linewidth=1.2,
            label="measured FK",
        )
        axis.scatter(
            *desired[0],
            color="black",
            marker="^",
            s=45,
            label="desired start",
        )
        axis.scatter(
            *desired[-1],
            color="black",
            marker="*",
            s=70,
            label="desired end",
        )
        axis.scatter(
            *measured[0],
            color=color,
            marker="o",
            s=35,
            label="measured start",
        )
        axis.scatter(
            *measured[-1],
            color=color,
            marker="X",
            s=45,
            label="measured end",
        )
        _equal_world_axes(axis, np.vstack((desired, measured)))
        source_frame = log.target_reference_frames.get(side, "unknown")
        axis.set_title(
            f"{side} arm · source frame: {source_frame}\n"
            "compared after resolution to world"
        )
        axis.set_xlabel("world x [m]")
        axis.set_ylabel("world y [m]")
        axis.set_zlabel("world z [m]")
        axis.legend(loc="best", fontsize=8)

    fig.suptitle(
        "Simulation EE Cartesian paths — controller-aligned samples, "
        "not hardware validation"
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)
    return output
