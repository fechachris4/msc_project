#!/usr/bin/env bash
# End-to-end parity verification: build, unit tests, then all three
# Python-vs-C++ comparisons. Run from the Python project root.
#
#   bash cpp/tools/verify_parity.sh
#
# Exits non-zero on the first failure so it is usable as a gate.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
BUILD="${BUILD:-$ROOT/cpp/build}"
OUT="${OUT:-$(mktemp -d)}"

echo "=============================================================="
echo "0. Build"
echo "=============================================================="
cmake -S cpp -B "$BUILD" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
cmake --build "$BUILD" -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)" >/dev/null
echo "build OK"

echo
echo "=============================================================="
echo "1. C++ unit tests"
echo "=============================================================="
ctest --test-dir "$BUILD" --output-on-failure

echo
echo "=============================================================="
echo "2. Effective configuration (expect byte-identical)"
echo "=============================================================="
"$PYTHON" -c 'from runtime_config import CONFIG, print_effective_config; print_effective_config(CONFIG)' \
    > "$OUT/py_config.txt"
"$BUILD/srl_print_config" > "$OUT/cpp_config.txt"
if diff -q "$OUT/py_config.txt" "$OUT/cpp_config.txt" >/dev/null; then
    echo "PASS: effective configuration is byte-identical"
else
    echo "FAIL: effective configuration differs"
    diff "$OUT/py_config.txt" "$OUT/cpp_config.txt" | head -20
    exit 1
fi

echo
echo "=============================================================="
echo "3. Golden trace  (250 cycles x 2 arms, human safety DISABLED)"
echo "=============================================================="
"$BUILD/srl_golden_trace" --out "$OUT/cpp_trace.csv"
"$PYTHON" cpp/tools/compare_trace.py tests/golden/reactive_current.csv \
    "$OUT/cpp_trace.csv"

echo
echo "=============================================================="
echo "4. Headless trace (2000 cycles x 2 arms, DEFAULT config:"
echo "   human safety + cylinder routing + trajectory all ENABLED)"
echo "=============================================================="
"$BUILD/srl_headless_trace" --out "$OUT/cpp_headless.csv" --steps 2000
"$PYTHON" cpp/tools/dump_headless.py --out "$OUT/py_headless.csv" --steps 2000
"$PYTHON" cpp/tools/compare_trace.py "$OUT/py_headless.csv" \
    "$OUT/cpp_headless.csv"

echo
echo "=============================================================="
echo "5. Trajectory generation (hold + line + C2 spline + circle)"
echo "=============================================================="
FIXTURE=cpp/tests/fixtures/trajectory_control.toml
"$BUILD/srl_trajectory_dump" "$FIXTURE" 400 > "$OUT/cpp_traj.txt"
"$PYTHON" cpp/tools/dump_trajectory.py "$FIXTURE" 400 > "$OUT/py_traj.txt"
"$PYTHON" - "$OUT/py_traj.txt" "$OUT/cpp_traj.txt" <<'PYEOF'
import math, sys
from collections import defaultdict

def load(path):
    rows = []
    for line in open(path):
        parts = line.strip().split(",")
        rows.append((parts[0], [float(x) for x in parts[1:]]))
    return rows

py, cpp = load(sys.argv[1]), load(sys.argv[2])
if len(py) != len(cpp):
    raise SystemExit(f"FAIL: line count differs ({len(py)} vs {len(cpp)})")

worst = defaultdict(float)
bad = 0
for index, ((kind_a, va), (kind_b, vb)) in enumerate(zip(py, cpp)):
    if kind_a != kind_b or len(va) != len(vb):
        raise SystemExit(f"FAIL: structure differs at line {index}")
    for a, b in zip(va, vb):
        worst[kind_a] = max(worst[kind_a], abs(a - b))
        if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
            bad += 1

for kind in sorted(worst, key=lambda k: -worst[k]):
    print(f"  {kind:<10} max_abs_diff={worst[kind]:.3e}")
if bad:
    raise SystemExit(f"FAIL: {bad} trajectory fields outside tolerance")
print("PASS: trajectory sampling matches (rtol=atol=1e-12)")
PYEOF

echo
echo "=============================================================="
echo "ALL PARITY CHECKS PASSED"
echo "artifacts in $OUT"
echo "=============================================================="
