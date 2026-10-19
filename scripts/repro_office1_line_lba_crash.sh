#!/usr/bin/env bash
# Minimal repro runner for the office1 mono line-LBA premature termination.
# Captures exit code, signal, wall time, max RSS without swallowing the status.
#
# Usage:
#   bash scripts/repro_office1_line_lba_crash.sh <out_dir>
#
# Env (optional):
#   RUNS         number of sequential runs (default 1)
#   DEBUG_CSV    1 = enable PHOTO_SLAM_DEBUG_MONO_OFFICE1 CSV
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office1.yaml"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office1"
OUT_ROOT="${1:-/tmp/repro_lba}"
RUNS="${RUNS:-1}"
DEBUG_CSV="${DEBUG_CSV:-1}"

GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
BIN_MD5="$(md5sum "${BIN}" | awk '{print $1}')"
LIB_MD5="$(md5sum "${PROJECT_ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" | awk '{print $1}')"

mkdir -p "${OUT_ROOT}"

echo "=== repro runner ==="
echo "  commit     : ${GIT_COMMIT}"
echo "  bin md5    : ${BIN_MD5}"
echo "  lib md5    : ${LIB_MD5}"
echo "  runs       : ${RUNS}"
echo "  debug csv  : ${DEBUG_CSV}"
echo "  out        : ${OUT_ROOT}"
echo "==================="

# Record cgroup memory.events if available (before).
CG_DIR="$(cat /proc/self/cgroup 2>/dev/null | cut -d: -f3)"
if [ -n "${CG_DIR}" ] && [ "${CG_DIR}" != "/" ]; then
    for f in memory.events memory.current memory.max memory.peak; do
        p="/sys/fs/cgroup${CG_DIR}/${f}"
        if [ -r "${p}" ]; then echo "before ${f}: $(cat "${p}" 2>/dev/null | tr '\n' ' ')"; fi
    done
else
    echo "cgroup: no memory controller path (cgroup v2 root)"
fi

for ((i = 1; i <= RUNS; i++)); do
    RUN_DIR="${OUT_ROOT}/run_${i}"
    mkdir -p "${RUN_DIR}"
    RUN_LOG="${RUN_DIR}/run.log"
    CSV_PATH="${RUN_DIR}/track.csv"

    {
        echo "start_time=$(date '+%Y-%m-%dT%H:%M:%S')"
        echo "pid=$$"
        echo "pgid=$(ps -o pgid= -p $$ 2>/dev/null | tr -d ' ')"
        echo "git_commit=${GIT_COMMIT}"
        echo "bin_md5=${BIN_MD5}"
        echo "lib_md5=${LIB_MD5}"
        echo "cmd=${BIN} ${VOCAB} ${ORB_CFG} ${GAUSS_CFG} ${DATASET} ${RUN_DIR}/ no_viewer"
    } > "${RUN_DIR}/run_config.txt"

    echo
    echo "=== run ${i}/${RUNS} ==="

    T0=$(date +%s)
    set +e
    env \
        PHOTO_SLAM_LINE_MODE=2 \
        PHOTO_SLAM_LINE_LOOP=0 \
        PHOTO_SLAM_SHADOW_LINE=0 \
        PHOTO_SLAM_SLAM_ONLY=1 \
        PHOTO_SLAM_DEBUG_MONO_OFFICE1="${CSV_PATH}" \
        "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${DATASET}" "${RUN_DIR}/" no_viewer \
        > "${RUN_LOG}" 2>&1
    rc=$?
    set -e
    T1=$(date +%s)
    echo "wall_time_s=$((T1 - T0))" >> "${RUN_LOG}"

    echo "EXIT=${rc}" >> "${RUN_LOG}"
    case ${rc} in
        0)   SIG="clean" ;;
        134) SIG="SIGABRT" ;;
        137) SIG="SIGKILL(OOM or external kill)" ;;
        139) SIG="SIGSEGV" ;;
        143) SIG="SIGTERM" ;;
        124) SIG="timeout" ;;
        *)   SIG="exit_code_${rc}" ;;
    esac
    echo "SIGNAL=${SIG}" >> "${RUN_LOG}"

    LAST_FRAME="$(grep -oE 'Frame id: [0-9]+' "${RUN_LOG}" 2>/dev/null | tail -1)"
    LAST_LBA="$(grep -E 'LocalBundleAdjustment|LBA-SEC|Local bundle adjustment' "${RUN_LOG}" 2>/dev/null | tail -1)"
    echo "last_frame_line=${LAST_FRAME}" >> "${RUN_LOG}"
    echo "last_lba_line=${LAST_LBA}" >> "${RUN_LOG}"

    echo "run ${i}: EXIT=${rc} (${SIG}) | ${LAST_FRAME}"
    echo "         ${LAST_LBA}"
done

echo
echo "All runs completed."
