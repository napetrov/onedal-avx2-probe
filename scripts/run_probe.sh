#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="${1:-$ROOT_DIR/out}"
mkdir -p "$OUT_DIR"

BIN="$OUT_DIR/avx2_full_probe"
LOG="$OUT_DIR/probe.log"
SUMMARY="$OUT_DIR/summary.txt"

# ─── Compiler / flags detection ────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)

if [ -z "${CXX:-}" ]; then
    if command -v g++ >/dev/null 2>&1; then CXX=g++
    elif command -v clang++ >/dev/null 2>&1; then CXX=clang++
    else echo "ERROR: no C++ compiler found"; exit 1; fi
fi

if [ -z "${CXXFLAGS:-}" ]; then
    if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
        # ARM: compile without AVX2 flags; probe will detect ARM and exit gracefully
        CXXFLAGS="-O2"
    else
        CXXFLAGS="-O2 -mavx2 -mfma -mbmi -mbmi2"
    fi
fi

LDFLAGS=""
if [ "$OS" != "Darwin" ]; then LDFLAGS="-lm"; fi

echo "[build] $CXX $CXXFLAGS (arch=$ARCH, os=$OS)"
$CXX $CXXFLAGS -o "$BIN" "$ROOT_DIR/src/avx2_full_probe.cpp" $LDFLAGS

echo "[run] $BIN"
set +e
"$BIN" | tee "$LOG"
RC=${PIPESTATUS[0]:-$?}
set -e

CPU=$(grep -m1 '^CPU:' "$LOG" | sed 's/^CPU:[[:space:]]*//' || echo "unknown")
PASS_LINE=$(grep -m1 'RESULTS:' "$LOG" || echo "N/A")
VERDICT=$(grep -E 'ALL PASS|FAILURES|ARM CPU' "$LOG" | tail -n1 || echo "N/A")

{
  echo "cpu=$CPU"
  echo "arch=$ARCH"
  echo "os=$OS"
  echo "rc=$RC"
  echo "results=${PASS_LINE# RESULTS: }"
  echo "verdict=$VERDICT"
} > "$SUMMARY"

echo "[summary]"
cat "$SUMMARY"

exit "$RC"
