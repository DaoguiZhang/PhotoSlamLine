#!/usr/bin/env bash
# Isolation suite to locate the frame-675 tracking-failure mechanism.
# Configs (all SLAM_ONLY=1, D=0 F=0 to test the pipeline itself):
#   P0  : LINE_MODE=0 (pure point ORB-SLAM3, no line frontend/backend)
#   S5  : LINE_MODE=2 PO_LINE=0 LBA_LINE=1 (no PO line edges)
#   S4  : LINE_MODE=2 PO_LINE=1 LBA_LINE=0 (no LBA line edges)
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/office1.yaml"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
DATASET="/workspace/code/SEGS-SLAM/datasets/replica/office1"
GT="${DATASET}/pose_TUM.txt"
OUT_ROOT="${1:-/tmp/iso_suite}"
RUNS="${RUNS:-10}"

# group -> LINE_MODE:PO_LINE:LBA_LINE
declare -A GRP
GRP[P0]="0:1:1"
GRP[S5]="2:0:1"
GRP[S4]="2:1:0"
ORDER=(P0 S5 S4)

mkdir -p "${OUT_ROOT}"

for ((i = 1; i <= RUNS; i++)); do
    for g in "${ORDER[@]}"; do
        LM="${GRP[$g]%%:*}"
        REST="${GRP[$g]#*:}"
        PO="${REST%%:*}"
        LBA="${REST##*:}"
        RUN_DIR="${OUT_ROOT}/${g}/run_${i}"
        mkdir -p "${RUN_DIR}"
        LOG="${RUN_DIR}/run.log"
        if grep -q '^EXIT=' "${LOG}" 2>/dev/null; then
            echo "  [${i}/${RUNS}] ${g}: skip (already done)"
            continue
        fi
        set +e
        env PHOTO_SLAM_LINE_MODE="${LM}" PHOTO_SLAM_LINE_LOOP=0 \
            PHOTO_SLAM_PO_LINE="${PO}" PHOTO_SLAM_LBA_LINE="${LBA}" \
            PHOTO_SLAM_SHADOW_LINE=0 PHOTO_SLAM_SLAM_ONLY=1 \
            PHOTO_SLAM_DEBUG_MONO_INIT=0 \
            "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${DATASET}" "${RUN_DIR}/" no_viewer \
            > "${LOG}" 2>&1
        rc=$?
        set -e
        echo "EXIT=${rc}" >> "${LOG}"
        LF="$(grep -oE 'Frame id: [0-9]+' "${LOG}" 2>/dev/null | tail -1)"
        MAPS="$(grep -oE 'There are [0-9]+ maps in the atlas' "${LOG}" 2>/dev/null | tail -1)"
        echo "  [${i}/${RUNS}] ${g} (LM=${LM} PO=${PO} LBA=${LBA}): EXIT=${rc} | ${LF} | ${MAPS}"
    done
done

echo "=== evaluating isolation groups ==="
for g in "${ORDER[@]}"; do
    echo "########## ${g} ##########"
    python3 "${SCRIPT_DIR}/eval_batch.py" "${OUT_ROOT}/${g}" "${GT}" "${SCRIPT_DIR}/eval_mono_stability.py" 2>&1 | tail -25
done
echo "=== iso_suite done ==="
