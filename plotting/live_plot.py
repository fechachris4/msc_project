"""Generic live time-series plot: stacked subplots sharing a time axis.

Knows nothing about MuJoCo or any experiment. Construct with the row
labels and signal names, feed samples with add(), and it handles the
ring buffers, redraw throttling, and figure lifecycle:

    plot = LivePlot(rows=["X (m)", "Y (m)", "Z (m)"],
                    signals={"direct": {}, "fk": {"style": "--"}})
    while plot.is_open():
        plot.add(t, {"direct": (x, y, z), "fk": (x, y, z)})
    plot.save("analysis/output/something.png")
"""

from collections import deque

import matplotlib.pyplot as plt


class LivePlot:
    def __init__(self, rows, signals, window=2000, redraw_every=25,
                 title=None, xlabel="sim time (s)"):
        """
        rows: y-axis label per subplot, top to bottom.
        signals: {name: options} — one line per signal per row.
                 options: {"style": matplotlib format string} plus any
                 Line2D kwargs, e.g. {"color": "#0072B2"} (all optional).
        window: samples kept on screen.
        redraw_every: add() calls per redraw.

        Y-axis autoscale is expand-only: per-row min/max only ever grow,
        never shrink, even as old samples age out of the ring buffer.
        Trade-off: a one-off boot transient keeps the row's scale small
        for the rest of the run, at the benefit of a legend/axis that
        never jumps around while you're watching it live.
        """
        plt.ion()
        self._fig, axes = plt.subplots(len(rows), 1, sharex=True,
                                       figsize=(8, 8), squeeze=False)
        self._axes = axes[:, 0]
        self._time = deque(maxlen=window)
        self._buffers = {}  # (signal, row) -> deque
        self._lines = {}  # (signal, row) -> Line2D
        self._redraw_every = redraw_every
        self._count = 0
        self._ymin = [None] * len(rows)  # expand-only autoscale, per row
        self._ymax = [None] * len(rows)

        for row, (ax, label) in enumerate(zip(self._axes, rows)):
            for name, options in signals.items():
                style = options.get("style", "-")
                kwargs = {k: v for k, v in options.items() if k != "style"}
                self._lines[name, row] = ax.plot([], [], style, label=name,
                                                 **kwargs)[0]
                self._buffers[name, row] = deque(maxlen=window)
            ax.set_ylabel(label)
        # one legend on the top row — every row has the same signals
        self._axes[0].legend(loc="upper right")
        self._axes[-1].set_xlabel(xlabel)
        if title:
            self._fig.suptitle(title)

    def is_open(self):
        return plt.fignum_exists(self._fig.number)

    def add(self, t, values):
        """values: {signal name: sequence of one value per row}."""
        self._time.append(t)
        for name, row_values in values.items():
            for row, value in enumerate(row_values):
                self._buffers[name, row].append(value)

        self._count += 1
        if self._count % self._redraw_every == 0:
            self._redraw()

    def _redraw(self):
        for key, line in self._lines.items():
            line.set_data(self._time, self._buffers[key])
        for row, ax in enumerate(self._axes):
            ax.relim()
            ax.autoscale_view(scaley=False)  # x only; y is expand-only below

            data_lo, data_hi = ax.dataLim.y0, ax.dataLim.y1
            lo = data_lo if self._ymin[row] is None else min(self._ymin[row], data_lo)
            hi = data_hi if self._ymax[row] is None else max(self._ymax[row], data_hi)
            if lo < hi:
                self._ymin[row], self._ymax[row] = lo, hi
                ax.set_ylim(lo, hi)
        plt.pause(0.001)  # redraw and let the GUI breathe

    def save(self, path):
        self._fig.savefig(path, dpi=150)
