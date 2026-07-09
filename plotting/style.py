"""Shared plotting constants: the Okabe-Ito colors and side conventions
used across plotting/ and analysis/, so every figure in the repo agrees
on what "right" and "left" look like.

Right arm: solid, C_RIGHT. Left arm: dashed, C_LEFT. C_BASE is the grey
used for base-motion traces. C_XYZ is the three-axis palette (x, y, z),
sharing its first two hues with C_RIGHT/C_LEFT.
"""

# Okabe-Ito
C_RIGHT = "#D55E00"
C_LEFT = "#0072B2"
C_BASE = "0.6"
C_XYZ = ("#D55E00", "#009E73", "#0072B2")

SIDE_COLOR = {"right": C_RIGHT, "left": C_LEFT}
SIDE_STYLE = {"right": "-", "left": "--"}
