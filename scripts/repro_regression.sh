#!/usr/bin/env bash
# Regression: run office0 / office2 / room0 (SLAM-only, line mode 2, loop 0)
# N times each and record exit codes. Fails the run if any exits non-zero.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BIN="${PROJECT_ROOT}/bin/replica_mono"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
DATASET_ROOT="/workspace/code/SEGS-SLAM/datasets/replica"
GAUSS_CFG="${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml"
OUT_ROOT="${1:-/tmp/repro_regression}"
RUNS="${RUNS:-3}"
SCENES="${SCENES:-office0 office2 room0}"

mkdir -p "${OUT_ROOT}"

overall=0
for scene in ${SCENES}; do
    ORB_CFG="${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/${scene}.yaml"
    DS="${DATASET_ROOT}/${scene}"
    if [ ! -f "${ORB_CFG}" ] || [ ! -d "${DS}" ]; then
        echo "SKIP ${scene}: missing config or dataset (${ORB_CFG}, ${DS})"
        continue
    fi
    for ((i = 1; i <= RUNS; i++)); do
        RUN_DIR="${OUT_ROOT}/${scene}_run${i}"
        mkdir -p "${RUN_DIR}"
        LOG="${RUN_DIR}/run.log"
        set +e
        env PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0 \
            PHOTO_SLAM_SLAM_ONLY=1 \
            "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${DS}" "${RUN_DIR}/" no_viewer \
            > "${LOG}" 2>&1
        rc=$?
        set -e
        echo "EXIT=${rc}" >> "${LOG}"
        LF="$(grep -oE 'Frame id: [0-9]+' "${LOG}" 2>/dev/null | tail -1)"
        echo "${scene} run${i}: EXIT=${rc} | ${LF}"
        if [ "${rc}" -ne 0 ]; then overall=1; fi
    done
done

echo "regression overall=${overall} (0=all clean)"
exit ${overall}
