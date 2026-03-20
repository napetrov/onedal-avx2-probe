#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="${1:-$ROOT_DIR/out}"
mkdir -p "$OUT_DIR"

BIN_AVX2="$OUT_DIR/avx2_full_probe"
BIN_ISA="$OUT_DIR/isa_dispatch_probe"
LOG_AVX2="$OUT_DIR/avx2_probe.log"
LOG_ISA="$OUT_DIR/isa_probe.log"
SUMMARY="$OUT_DIR/summary.txt"

# ─── Compiler / flags detection ────────────────────────────────────────────
ARCH=$(uname -m)
OS=$(uname -s)

if [ -z "${CXX:-}" ]; then
    if command -v g++ >/dev/null 2>&1; then CXX=g++
    elif command -v clang++ >/dev/null 2>&1; then CXX=clang++
    else echo "ERROR: no C++ compiler found"; exit 1; fi
fi

if [ "$ARCH" = "arm64" ] || [ "$ARCH" = "aarch64" ]; then
    CXXFLAGS_AVX2="-O2"
    CXXFLAGS_ISA="-O2"
else
    CXXFLAGS_AVX2="${CXXFLAGS:-"-O2 -mavx2 -mfma -mbmi -mbmi2"}"
    CXXFLAGS_ISA="-O2 -msse4.2 -mavx2 -mfma -mbmi -mbmi2"
fi

LDFLAGS=""
if [ "$OS" != "Darwin" ]; then LDFLAGS="-lm"; fi

echo "================================================"
echo " Probe 1: AVX2 Full Suite"
echo "================================================"
echo "[build] $CXX $CXXFLAGS_AVX2"
$CXX $CXXFLAGS_AVX2 -o "$BIN_AVX2" "$ROOT_DIR/src/avx2_full_probe.cpp" $LDFLAGS
set +e; "$BIN_AVX2" | tee "$LOG_AVX2"; RC_AVX2=${PIPESTATUS[0]:-$?}; set -e

echo ""
echo "================================================"
echo " Probe 2: ISA Dispatch (SSE2 / SSE4.2 / AVX2)"
echo "================================================"
echo "[build] $CXX $CXXFLAGS_ISA"
$CXX $CXXFLAGS_ISA -o "$BIN_ISA" "$ROOT_DIR/src/isa_dispatch_probe.cpp" $LDFLAGS
set +e; "$BIN_ISA" | tee "$LOG_ISA"; RC_ISA=${PIPESTATUS[0]:-$?}; set -e

# ─── Summary ────────────────────────────────────────────────────────────────
CPU=$(grep -m1 '^CPU:' "$LOG_ISA" | sed 's/^CPU:[[:space:]]*//' || echo "unknown")
DISPATCH=$(grep 'USE AVX2\|USE SSE4.2\|USE SSE2\|NO SIMD\|N/A' "$LOG_ISA" | tail -1 || echo "unknown")
AVX2_VERDICT=$(grep -E '✅ ALL PASS|❌.*FAIL' "$LOG_AVX2" | tail -1 || echo "N/A")

{
  echo "cpu=$CPU"
  echo "arch=$ARCH"
  echo "os=$OS"
  echo "rc_avx2=$RC_AVX2"
  echo "rc_isa=$RC_ISA"
  echo "dispatch=$DISPATCH"
  echo "avx2_full=$AVX2_VERDICT"
} > "$SUMMARY"

echo ""
echo "================================================"
echo " SUMMARY"
echo "================================================"
cat "$SUMMARY"

[ "$RC_AVX2" -eq 0 ] && [ "$RC_ISA" -eq 0 ]
