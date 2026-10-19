#!/usr/bin/env bash
# Run one SLAM-only office1 pass under ASan+UBSan using the staged instrumented
# libraries (/tmp/libORB_SLAM3_asan.so, /tmp/libg2o_asan.so). Swaps the working
# libraries in place for the run and restores the known-good ones afterwards.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office1.yaml"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office1"
OUT_ROOT="${1:-/tmp/asan_run}"
SLAM_LIB="${PROJECT_ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so"
G2O_LIB="${PROJECT_ROOT}/ORB-SLAM3/Thirdparty/g2o/lib/libg2o.so"

if [ ! -f /tmp/libORB_SLAM3_asan.so ]; then
    echo "Missing /tmp/libORB_SLAM3_asan.so — run scripts/build_asan.sh first" >&2
    exit 2
fi

mkdir -p "${OUT_ROOT}"

# Back up good libs.
cp -v "${SLAM_LIB}" /tmp/libORB_SLAM3_good_run.so
cp -v "${G2O_LIB}" /tmp/libg2o_good_run.so

# Swap in instrumented libs.
cp -v /tmp/libORB_SLAM3_asan.so "${SLAM_LIB}"
cp -v /tmp/libg2o_asan.so "${G2O_LIB}"

ASAN_PRELOAD="$(g++ -print-file-name=libasan.so)"
UBSAN_PRELOAD="$(g++ -print-file-name=libubsan.so)"

set +e
env \
    LD_PRELOAD="${ASAN_PRELOAD}:${UBSAN_PRELOAD}" \
    ASAN_OPTIONS="abort_on_error=1:detect_leaks=0:fast_unwind_on_malloc=0" \
    UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1" \
    PHOTO_SLAM_LINE_MODE=2 \
    PHOTO_SLAM_LINE_LOOP=0 \
    PHOTO_SLAM_SHADOW_LINE=0 \
    PHOTO_SLAM_SLAM_ONLY=1 \
    "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${DATASET}" "${OUT_ROOT}/" no_viewer \
    > "${OUT_ROOT}/run.log" 2>&1
rc=$?
set -e

# Restore good libs.
cp -v /tmp/libORB_SLAM3_good_run.so "${SLAM_LIB}"
cp -v /tmp/libg2o_good_run.so "${G2O_LIB}"

echo "EXIT=${rc}" >> "${OUT_ROOT}/run.log"
echo "lastframe=$(grep -oE 'Frame id: [0-9]+' "${OUT_ROOT}/run.log" | tail -1)" >> "${OUT_ROOT}/run.log"
echo "ASan run EXIT=${rc}; sanitizer findings:"
grep -E "ERROR: AddressSanitizer|runtime error:|UndefinedBehaviorSanitizer|SUMMARY:" "${OUT_ROOT}/run.log" | head -20 || true
