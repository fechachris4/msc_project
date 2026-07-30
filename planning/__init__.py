"""Collision-aware Cartesian path planning above the reactive controller.

This package is a planning layer, not a control layer.  It is a
feasibility stage inside ONE arm's reference pipeline: ``plan_arm``
optimises that arm's path and the composition root (``arm_flow.py``)
delivers it through the ordinary single-arm ``TargetSource`` seam
(``controller/trajectory.py``).  The planner never holds or replaces the
other arm's reference, and no file in ``controller/`` imports this
package into the control path.

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
