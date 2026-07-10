# Parameter Unit Comments Design

## Goal

Make the units of user-editable Python parameters immediately visible without
changing any values or runtime behaviour.

## Scope

Add short adjacent unit comments to research and configuration parameters in
`sim/`, `controller/`, `analysis/`, and `plotting/`. Use the established project
units: `m`, `rad`, `rad/s`, `Hz`, `s`, `1/s`, and `dimensionless`.

Preserve comments that already state the unit clearly. Do not annotate internal
calculation literals, tests, plotting cosmetics, generated outputs, or MuJoCo
XML values. Do not refactor code or rename parameters.

The `controller.desired_pos.POSES` position values are metres and its RPY
values, including `[0.0, 45.0, 190.0]`, are intentionally radians. Their values
and interpretation will remain unchanged.

## Comment Style

Prefer a short inline comment beside a scalar or compact assignment. For a
structured parameter group where inline comments would harm readability, use
one immediately adjacent comment that maps field names to units. Do not assign
a physical unit to numerical conventions such as the mixed-Jacobian DLS lambda;
label those by meaning instead.

## Verification

Review the diff to prove that only comments changed, compile every modified
Python file, and run the focused existing tests for any modules touched by the
comment edits. Preserve all pre-existing working-tree changes.
