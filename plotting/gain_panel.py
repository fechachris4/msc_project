"""Live gain-tuning panel: six sliders that write straight into
controller.servo's module-level gains, read by the controller's next
control step -- no restart needed to feel a retune.

    from plotting.gain_panel import GainPanel
    panel = GainPanel()
    while running:
        ...
        panel.pump()   # only needed if nothing else pumps the GUI event
                       # loop (e.g. no LivePlot already calling plt.pause)

Closing the panel prints the six current values formatted ready to paste
back into servo.py's module constants. The panel never writes to disk --
committed values in servo.py stay the single source of truth.

Layering: this module is plotting-layer and imports controller.servo to
write gains. Nothing in controller/ or sim/ may import this module.
"""

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from controller import servo

# Slider ranges: (min, max), units per servo.py's own docstring.
_RANGES = {
    "KP_POS": (0.0, 10.0),    # 1/s
    "KP_ROT": (0.0, 10.0),    # 1/s
    "KD_POS": (0.0, 0.99),    # servo.py: KD < 1 is a stability constraint
    "KD_ROT": (0.0, 0.99),
    "K_NULL": (0.0, 5.0),     # 1/s
    "DAMPING": (0.001, 0.2),  # never 0 -- DLS must stay damped near singularities
}

_ORDER = ["KP_POS", "KP_ROT", "KD_POS", "KD_ROT", "K_NULL", "DAMPING"]


def _fmt(value):
    return str(round(float(value), 4))


class GainPanel:
    """Standalone figure, one slider per gain in _ORDER plus a Reset
    button. Sliders open at servo's current values; each callback writes
    the new value straight into servo (K_NULL via servo.set_k_null, so
    _K_NULL_VEC rescales with it -- the panel never touches servo
    privates itself)."""

    def __init__(self, on_change=None):
        """on_change, if given, is called after every slider write (after
        the value lands on servo) and after Reset restores the initial
        values -- Reset's set_val calls already re-trigger each slider's
        own on_changed callback, so this is wired once, in
        _make_callback, and covers both paths."""
        self._on_change = on_change
        self._initial = {name: getattr(servo, name) for name in _ORDER}

        self._fig, axes = plt.subplots(len(_ORDER) + 1, 1, figsize=(5, 6))
        self._fig.subplots_adjust(hspace=0.9, left=0.25, right=0.9)
        self._fig.suptitle("Live gains")

        self._sliders = {}
        for ax, name in zip(axes[:-1], _ORDER):
            lo, hi = _RANGES[name]
            slider = Slider(ax, name, lo, hi, valinit=self._initial[name])
            slider.on_changed(self._make_callback(name))
            self._sliders[name] = slider

        self._reset_button = Button(axes[-1], "Reset")
        self._reset_button.on_clicked(self._on_reset)

        self._fig.canvas.mpl_connect("close_event", self._on_close)

    def _make_callback(self, name):
        def callback(value):
            if name == "K_NULL":
                servo.set_k_null(value)
            else:
                setattr(servo, name, value)
            print(f"{name} = {_fmt(value)}")
            if self._on_change:
                self._on_change()
        return callback

    def _on_reset(self, event):
        for name in _ORDER:
            self._sliders[name].set_val(self._initial[name])

    def _on_close(self, event):
        print("# paste into controller/servo.py:")
        for name in _ORDER:
            print(f"{name} = {_fmt(self._sliders[name].val)}")

    def pump(self):
        """Let the panel's GUI event loop breathe -- call once per
        control-loop iteration in front ends that don't already run a
        LivePlot (LivePlot's own plt.pause pumps every open figure,
        this one included)."""
        plt.pause(0.001)
