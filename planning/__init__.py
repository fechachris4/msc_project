"""Collision-aware Cartesian path planning above the reactive controller.

This package is a planning layer, not a control layer.  It produces a
world-frame Cartesian path that is delivered to the existing
``ReactivePositionRunner`` through the existing ``DualArmTargetSource``
seam (``controller/trajectory.py``).  No file in ``controller/`` is
modified or imported into the control path by this package.

Design boundaries, all deliberate:

- Collision is evaluated in the TORSO frame, where the human envelope is
  static by construction.  The delivered reference stays WORLD-frame,
  because the task is a world-frame pose hold (see AGENTS.md).
- The path is represented by the project's existing minimum-jerk C2
  spline.  The optimiser only chooses where its knots go; smoothness,
  timing, and Cartesian rate limits remain the trajectory layer's job.
- The optimiser is plain numpy/scipy.  GTSAM/GPMP2 are deliberately not
  dependencies: the ideas are adapted, the library is not.
"""
