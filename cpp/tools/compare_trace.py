#!/usr/bin/env python3
"""Compare the Python and C++ control traces column by column.

    python cpp/tools/compare_trace.py tests/golden/reactive_current.csv \
        cpp/build/cpp_trace.csv [--rtol 1e-12] [--atol 1e-12]

Reports, per column, the worst absolute and relative disagreement and where it
occurs, then the overall verdict at the requested tolerance. Discrete columns
(cycle, arm, and the saturation flags) are compared exactly, as the Python
harness does.

The point is to make residual differences visible rather than to hide them
behind a single pass/fail, so the summary always prints the worst offenders
even when the run passes.
"""

import argparse
import csv
import math
import sys
from pathlib import Path

DISCRETE_PREFIXES = ("speed_saturated_", "lead_clamped_", "range_clamped_")
DISCRETE_EXACT = (
    "cycle", "arm", "reason", "route_kind", "active_count", "waypoint_count",
    "human_adjusted", "limit_adjusted", "stopped", "target_adjusted",
)


def is_discrete(name):
    return name in DISCRETE_EXACT or name.startswith(DISCRETE_PREFIXES)


def load(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("expected")
    parser.add_argument("actual")
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    expected = load(args.expected)
    actual = load(args.actual)

    if len(expected) != len(actual):
        raise SystemExit(
            f"row count differs: expected {len(expected)}, actual {len(actual)}"
        )
    if tuple(expected[0]) != tuple(actual[0]):
        only_expected = sorted(set(expected[0]) - set(actual[0]))
        only_actual = sorted(set(actual[0]) - set(expected[0]))
        raise SystemExit(
            "trace columns or column ordering differ; "
            f"only_expected={only_expected[:8]} only_actual={only_actual[:8]}"
        )

    columns = list(expected[0])
    stats = {}
    discrete_mismatches = []
    failures = []

    for row_index, (expected_row, actual_row) in enumerate(zip(expected, actual)):
        for name in columns:
            expected_value = expected_row[name]
            actual_value = actual_row[name]
            if is_discrete(name):
                if expected_value != actual_value:
                    discrete_mismatches.append(
                        (row_index, name, expected_value, actual_value)
                    )
                continue

            expected_number = float(expected_value)
            actual_number = float(actual_value)
            absolute = abs(actual_number - expected_number)
            relative = (
                absolute / abs(expected_number)
                if expected_number != 0.0
                else (0.0 if absolute == 0.0 else math.inf)
            )
            record = stats.setdefault(
                name, {"abs": 0.0, "rel": 0.0, "row": -1, "exp": 0.0, "act": 0.0}
            )
            if absolute > record["abs"]:
                record.update(
                    abs=absolute,
                    rel=relative,
                    row=row_index,
                    exp=expected_number,
                    act=actual_number,
                )
            if not math.isclose(
                expected_number, actual_number,
                rel_tol=args.rtol, abs_tol=args.atol,
            ):
                failures.append(
                    (row_index, name, expected_number, actual_number,
                     absolute, relative)
                )

    ranked = sorted(stats.items(), key=lambda item: -item[1]["abs"])
    print(f"rows compared        : {len(expected)}")
    print(f"columns compared     : {len(columns)}")
    print(f"tolerance            : rtol={args.rtol:g} atol={args.atol:g}")
    print()
    print(f"worst {args.top} columns by absolute difference:")
    print(f"  {'column':<28} {'max_abs':>12} {'rel_at_max':>12} {'row':>6}")
    for name, record in ranked[: args.top]:
        print(
            f"  {name:<28} {record['abs']:>12.3e} {record['rel']:>12.3e} "
            f"{record['row']:>6}"
        )

    exact = sum(1 for _, record in ranked if record["abs"] == 0.0)
    print()
    print(f"columns bit-identical: {exact}/{len(ranked)}")
    if ranked:
        print(f"global max abs diff  : {ranked[0][1]['abs']:.3e} "
              f"({ranked[0][0]})")

    if discrete_mismatches:
        print()
        print(f"DISCRETE MISMATCHES  : {len(discrete_mismatches)}")
        for row_index, name, expected_value, actual_value in discrete_mismatches[:10]:
            print(f"  row {row_index} {name}: {expected_value} != {actual_value}")

    print()
    if failures or discrete_mismatches:
        print(f"FAIL: {len(failures)} numeric field(s) outside tolerance, "
              f"{len(discrete_mismatches)} discrete mismatch(es)")
        for item in failures[:10]:
            row_index, name, expected_number, actual_number, absolute, relative = item
            print(
                f"  row={row_index} field={name} expected={expected_number:.17g} "
                f"actual={actual_number:.17g} abs={absolute:.3e} rel={relative:.3e}"
            )
        return 1

    print(f"PASS: {len(expected) * len(columns)} fields match "
          f"(rtol={args.rtol:g}, atol={args.atol:g})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
