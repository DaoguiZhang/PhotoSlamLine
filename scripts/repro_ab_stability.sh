#!/usr/bin/env bash
# A/B stability runner: office1 SLAM-only with D (deterministic order) and
# F (zero-line point fallback) toggles. Runs N times, saves each trajectory and
# evaluates Sim(3) ATE/scale/reset per run.
#
# Usage:
#   GROUP=A00 D=0 F=1 RUNS=10 bash scripts/repro_ab_stability.sh <out_dir>
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office1.yaml"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office1"
GT="${DATASET}/pose_TUM.txt"
OUT_ROOT="${1:-/tmp/ab_runs}"
GROUP="${GROUP:-A00}"
D="${D:-0}"
F="${F:-1}"
RUNS="${RUNS:-10}"

mkdir -p "${OUT_ROOT}/${GROUP}"
echo "=== A/B group ${GROUP}: D=${D} F=${F} runs=${RUNS} ==="

for ((i = 1; i <= RUNS; i++)); do
    RUN_DIR="${OUT_ROOT}/${GROUP}/run_${i}"
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
    echo "  ${GROUP} run${i}: EXIT=${rc} | ${LF} | ${MAPS}"
done

echo "=== evaluating ${GROUP} ==="
python3 "${SCRIPT_DIR}/eval_batch.py" "${OUT_ROOT}/${GROUP}" "${GT}" "${SCRIPT_DIR}/eval_mono_stability.py"
