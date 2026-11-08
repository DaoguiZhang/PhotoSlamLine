#!/usr/bin/env bash
# 20x SLAM-only office1 run WITHOUT the debug CSV, to test the clean binary at
# natural timing (the CSV adds ~2.5x overhead and can mask a timing-sensitive
# termination). Captures exit code/signal per run without swallowing status.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office1.yaml"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office1"
OUT_ROOT="${1:-/tmp/repro_nocsv}"
RUNS="${RUNS:-20}"

mkdir -p "${OUT_ROOT}"

for ((i = 1; i <= RUNS; i++)); do
    RUN_DIR="${OUT_ROOT}/run_${i}"
    mkdir -p "${RUN_DIR}"
    RUN_LOG="${RUN_DIR}/run.log"

    echo "=== run ${i}/${RUNS} ==="
    T0=$(date +%s)
    set +e
    env \
        PHOTO_SLAM_LINE_MODE=2 \
        PHOTO_SLAM_LINE_LOOP=0 \
        PHOTO_SLAM_SHADOW_LINE=0 \
        PHOTO_SLAM_SLAM_ONLY=1 \
        "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${DATASET}" "${RUN_DIR}/" no_viewer \
        > "${RUN_LOG}" 2>&1
    rc=$?
    set -e
    T1=$(date +%s)

    case ${rc} in
        0)   SIG="clean" ;;
        134) SIG="SIGABRT" ;;
        137) SIG="SIGKILL(OOM or external kill)" ;;
        139) SIG="SIGSEGV" ;;
        143) SIG="SIGTERM" ;;
        *)   SIG="exit_code_${rc}" ;;
    esac
    LAST_FRAME="$(grep -oE 'Frame id: [0-9]+' "${RUN_LOG}" 2>/dev/null | tail -1)"
    echo "EXIT=${rc}" >> "${RUN_LOG}"
    echo "SIGNAL=${SIG}" >> "${RUN_LOG}"
    echo "wall_time_s=$((T1 - T0))" >> "${RUN_LOG}"
    echo "last_frame_line=${LAST_FRAME}" >> "${RUN_LOG}"
    echo "run ${i}: EXIT=${rc} (${SIG}) | ${LAST_FRAME}"
done
echo "All runs completed."
