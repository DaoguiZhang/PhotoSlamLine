#!/usr/bin/env bash
# office1 SLAM-only 30x acceptance with the chosen fix config (D=1 F=1).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office1.yaml"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office1"
GT="${DATASET}/pose_TUM.txt"
OUT_ROOT="${1:-/tmp/accept_30x}"
RUNS="${RUNS:-30}"
MONO_INIT_DEBUG="${MONO_INIT_DEBUG:-0}"

mkdir -p "${OUT_ROOT}"

for ((i = 1; i <= RUNS; i++)); do
    RUN_DIR="${OUT_ROOT}/run_${i}"
    mkdir -p "${RUN_DIR}"
    LOG="${RUN_DIR}/run.log"
    if grep -q '^EXIT=' "${LOG}" 2>/dev/null; then
        echo "  [${i}/${RUNS}]: skip (already done)"
        continue
    fi
    set +e
    env PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0 \
        PHOTO_SLAM_SLAM_ONLY=1 \
        PHOTO_SLAM_DEBUG_MONO_INIT="${MONO_INIT_DEBUG}" \
        PHOTO_SLAM_DETERMINISTIC_ORDER=1 \
        PHOTO_SLAM_ZERO_LINE_POINT_FALLBACK=1 \
        "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${DATASET}" "${RUN_DIR}/" no_viewer \
        > "${LOG}" 2>&1
    rc=$?
    set -e
    echo "EXIT=${rc}" >> "${LOG}"
    LF="$(grep -oE 'Frame id: [0-9]+' "${LOG}" 2>/dev/null | tail -1)"
    MAPS="$(grep -oE 'There are [0-9]+ maps in the atlas' "${LOG}" 2>/dev/null | tail -1)"
    RESETS="$(grep -c ResetActiveMap "${LOG}" 2>/dev/null || true)"
    echo "  [${i}/${RUNS}]: EXIT=${rc} | ${LF} | ${MAPS} | resets=${RESETS}"
done

echo "=== evaluating 30x acceptance ==="
python3 "${SCRIPT_DIR}/eval_batch.py" "${OUT_ROOT}" "${GT}" "${SCRIPT_DIR}/eval_mono_stability.py" 2>&1 | tail -45
echo "=== accept_30x done ==="
