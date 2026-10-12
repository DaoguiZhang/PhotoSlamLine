#!/usr/bin/env bash
# Cross-sequence regression with the chosen fix config (D=1 F=1).
# office0 / office2 / room0 x RUNS (default 5), full Gaussian (SLAM_ONLY=0).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
OUT_ROOT="${1:-/tmp/accept_xseq}"
RUNS="${RUNS:-5}"
MONO_INIT_DEBUG="${MONO_INIT_DEBUG:-0}"
SEQS="${SEQS:-office0 office2 room0}"

mkdir -p "${OUT_ROOT}"

for seq in ${SEQS}; do
    ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/${seq}.yaml"
    GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
    DATASET="/workspace/code/SEGS-SLAM/datasets/replica/${seq}"
    GT="${DATASET}/pose_TUM.txt"
    if [ ! -f "${ORB_CFG}" ]; then echo "MISSING ORB_CFG ${ORB_CFG}"; continue; fi
    if [ ! -f "${GT}" ]; then echo "MISSING GT ${GT}"; continue; fi
    for ((i = 1; i <= RUNS; i++)); do
        RUN_DIR="${OUT_ROOT}/${seq}/run_${i}"
        mkdir -p "${RUN_DIR}"
        LOG="${RUN_DIR}/run.log"
        if grep -q '^EXIT=' "${LOG}" 2>/dev/null; then
            echo "  [${seq} ${i}/${RUNS}]: skip (already done)"
            continue
        fi
        set +e
        env PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0 \
            PHOTO_SLAM_SLAM_ONLY=0 \
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
        echo "  [${seq} ${i}/${RUNS}]: EXIT=${rc} | ${LF} | ${MAPS} | resets=${RESETS}"
    done
done

echo "=== evaluating cross-sequence regression ==="
for seq in ${SEQS}; do
    GT="/workspace/code/SEGS-SLAM/datasets/replica/${seq}/pose_TUM.txt"
    echo "########## ${seq} ##########"
    python3 "${SCRIPT_DIR}/eval_batch.py" "${OUT_ROOT}/${seq}" "${GT}" "${SCRIPT_DIR}/eval_mono_stability.py" 2>&1 | tail -25
done
echo "=== accept_xseq done ==="
