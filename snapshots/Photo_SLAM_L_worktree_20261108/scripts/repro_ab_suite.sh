#!/usr/bin/env bash
# Interleaved A/B suite: A00 (D0 F0), D10 (D1 F0), F01 (D0 F1), DF11 (D1 F1).
# Runs round-robin to avoid time-of-day bias. Evaluates ATE/scale/reset per run.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office1.yaml"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office1"
GT="${DATASET}/pose_TUM.txt"
OUT_ROOT="${1:-/tmp/ab_suite}"
RUNS="${RUNS:-10}"

# group -> D:F
declare -A GRP
GRP[A00]="0:0"
GRP[D10]="1:0"
GRP[F01]="0:1"
GRP[DF11]="1:1"
ORDER=(A00 D10 F01 DF11)

mkdir -p "${OUT_ROOT}"

for ((i = 1; i <= RUNS; i++)); do
    for g in "${ORDER[@]}"; do
        D="${GRP[$g]%%:*}"
        F="${GRP[$g]##*:}"
        RUN_DIR="${OUT_ROOT}/${g}/run_${i}"
        mkdir -p "${RUN_DIR}"
        LOG="${RUN_DIR}/run.log"
        set +e
        env PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0 \
            PHOTO_SLAM_SLAM_ONLY=1 \
            PHOTO_SLAM_DETERMINISTIC_ORDER="${D}" \
            PHOTO_SLAM_ZERO_LINE_POINT_FALLBACK="${F}" \
            "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${DATASET}" "${RUN_DIR}/" no_viewer \
            > "${LOG}" 2>&1
        rc=$?
        set -e
        echo "EXIT=${rc}" >> "${LOG}"
        LF="$(grep -oE 'Frame id: [0-9]+' "${LOG}" 2>/dev/null | tail -1)"
        MAPS="$(grep -oE 'There are [0-9]+ maps in the atlas' "${LOG}" 2>/dev/null | tail -1)"
        echo "  [${i}/${RUNS}] ${g} (D=${D} F=${F}): EXIT=${rc} | ${LF} | ${MAPS}"
    done
done

echo "=== evaluating all groups ==="
for g in "${ORDER[@]}"; do
    echo "########## ${g} ##########"
    python3 "${SCRIPT_DIR}/eval_batch.py" "${OUT_ROOT}/${g}" "${GT}" "${SCRIPT_DIR}/eval_mono_stability.py" 2>&1 | tail -25
done
echo "=== A/B suite done ==="
