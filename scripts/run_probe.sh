#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="${1:-$ROOT_DIR/out}"
mkdir -p "$OUT_DIR"

CXX=${CXX:-g++}
CXXFLAGS=${CXXFLAGS:-"-O2 -mavx2 -mfma -mbmi -mbmi2"}
BIN="$OUT_DIR/avx2_full_probe"
LOG="$OUT_DIR/probe.log"
SUMMARY="$OUT_DIR/summary.txt"

echo "[build] $CXX $CXXFLAGS"
$CXX $CXXFLAGS -o "$BIN" "$ROOT_DIR/src/avx2_full_probe.cpp" -lm

echo "[run] $BIN"
set +e
"$BIN" | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e

CPU=$(grep -m1 '^CPU:' "$LOG" | sed 's/^CPU:[[:space:]]*//')
PASS_LINE=$(grep -m1 '^ RESULTS:' "$LOG" || true)
VERDICT=$(grep -E '✅ ALL PASS|❌ [0-9]+ FAILURES' "$LOG" | tail -n1 || true)

{
  echo "cpu=$CPU"
  echo "rc=$RC"
  echo "results=${PASS_LINE# RESULTS: }"
  echo "verdict=$VERDICT"
} > "$SUMMARY"

echo "[summary]"
cat "$SUMMARY"

exit "$RC"
