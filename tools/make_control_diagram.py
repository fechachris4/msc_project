"""Block diagram of the reactive pose-hold loop with mount feedforward.

Labels are short on purpose: the equations live in the README.
Output: media/control_loop.png. Left of each "|": hardware; right: simulation.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

OUT = Path("media")
OUT.mkdir(parents=True, exist_ok=True)
BOX = dict(boxstyle="round,pad=0.25", linewidth=1.0, edgecolor="black")
CTRL, MEAS, PLANT = "#E8F1FA", "#F2F2F2", "#FBEFE3"
FF = "#D55E00"
FS_T, FS_B, FS_L = 10.5, 9.5, 9.0


def box(ax, x, y, w, h, title, body, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, facecolor=fc, **BOX))
    ax.text(x + w / 2, y + h - 0.3, title, ha="center", va="top",
            fontsize=FS_T, fontweight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.35, body, ha="center", va="center",
            fontsize=FS_B, linespacing=1.3)


def arrow(ax, p, q, label=None, dx=0.0, dy=0.18, ha="center"):
    ax.annotate("", xy=q, xytext=p,
                arrowprops=dict(arrowstyle="-|>", linewidth=1.0,
                                color="black", shrinkA=0, shrinkB=0))
    if label:
        ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, label,
                ha=ha, va="bottom", fontsize=FS_L)


fig, ax = plt.subplots(figsize=(8.2, 5.0), layout="constrained")
ax.set_xlim(0, 24.6)
ax.set_ylim(0.6, 12.2)
ax.axis("off")

# forward path, top row: six boxes
y0, h, w, gap = 8.4, 3.0, 3.5, 0.6
xs = [0.3 + i * (w + gap) for i in range(6)]
box(ax, xs[0], y0, w, h, "Reference", "world-fixed pose\n$(p^*,R^*)$, $\\xi^*=0$", MEAS)
box(ax, xs[1], y0, w, h, "Errors", "$e_p,\\ e_R,\\ e_\\xi$\nin $W$", CTRL)
box(ax, xs[2], y0, w, h, "PD law", "$\\dot x = Ke + K_d e_\\xi$\n$-\\ \\xi_{E,mount}$", CTRL)
box(ax, xs[3], y0, w, h, "DLS, null space", "$\\dot q = J^{\\#}\\dot x$\n$+ (I-J^{\\#}J)\\,u$", CTRL)
box(ax, xs[4], y0, w, h, "Safety filter", "arm spheres vs\nwearer cylinder,\njoint limits", CTRL)
box(ax, xs[5], y0, w, h, "Integrate", "$q_{cmd} \\leftarrow q_{cmd}+\\dot q\\Delta t$\nrunaway stop", CTRL)
labels = [None, None, None, None, None]
for i in range(5):
    arrow(ax, (xs[i] + w, y0 + h / 2), (xs[i + 1], y0 + h / 2), labels[i])

# plant and measurement, bottom row
y1, h1 = 1.4, 3.0
bx = {"mount": 2.6, "kin": 11.0, "arm": 19.2}
bw = {"mount": 5.0, "kin": 6.2, "arm": 4.4}
box(ax, bx["mount"], y1, bw["mount"], h1, "Mount state",
    "${}^W T_M$, $\\xi_M$\nVicon  |  scripted motion", MEAS)
box(ax, bx["kin"], y1, bw["kin"], h1, "Composed kinematics",
    "${}^W T_E$, ${}^W J$, $\\xi_E$\nthrough the mount", MEAS)
box(ax, bx["arm"], y1, bw["arm"], h1, "Arms",
    "joint position servo\nGen3  |  MuJoCo", PLANT)

# integrate -> arms
xi = xs[5] + w / 2
arrow(ax, (xi, y0), (xi, y1 + h1))
ax.text(xi + 0.25, (y0 + y1 + h1) / 2, "$q_{cmd}$", fontsize=FS_L, va="center")
# arms -> kinematics
arrow(ax, (bx["arm"], y1 + h1 / 2), (bx["kin"] + bw["kin"], y1 + h1 / 2), "$q,\\dot q$")
# mount -> kinematics
arrow(ax, (bx["mount"] + bw["mount"], y1 + h1 / 2), (bx["kin"], y1 + h1 / 2),
      "${}^W T_M,\\ \\xi_M$", dy=0.25)
# kinematics -> errors (feedback), and -> DLS (Jacobian)
xk = bx["kin"] + 1.2
xe = xs[1] + w / 2
ax.plot([xk, xk, xe, xe], [y1 + h1, y1 + h1 + 1.5, y1 + h1 + 1.5, y0 - 0.02],
        color="black", linewidth=1.0)
ax.annotate("", xy=(xe, y0), xytext=(xe, y0 - 0.3),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color="black"))
ax.text(xe + 0.25, y1 + h1 + 1.6, "measured pose, twist",
        ha="left", va="bottom", fontsize=FS_L - 0.5)
xj = xs[3] + w / 2
ax.plot([bx["kin"] + bw["kin"] - 1.2, bx["kin"] + bw["kin"] - 1.2, xj, xj],
        [y1 + h1, y1 + h1 + 0.9, y1 + h1 + 0.9, y0 - 0.02],
        color="black", linewidth=1.0, linestyle="--")
ax.annotate("", xy=(xj, y0), xytext=(xj, y0 - 0.3),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.0, color="black"))
ax.text(xj + 0.25, y1 + h1 + 1.8, "${}^W J$", fontsize=FS_L, va="center")

# feedforward: mount-induced EE velocity, straight into the PD block
xf = xs[2] + w / 2
xk2 = bx["kin"] + 2.6
ax.plot([xk2, xk2, xf, xf], [y1 + h1, y1 + h1 + 0.45, y1 + h1 + 0.45, y0 - 0.02],
        color=FF, linewidth=1.4, linestyle="--")
ax.annotate("", xy=(xf, y0), xytext=(xf, y0 - 0.3),
            arrowprops=dict(arrowstyle="-|>", linewidth=1.4, color=FF))
ax.text(xf - 0.2, y0 - 0.9, "feedforward", color=FF, fontsize=FS_L,
        ha="right", va="center")

fig.savefig(OUT / "control_loop.png", dpi=220)
print(OUT / "control_loop.png")
