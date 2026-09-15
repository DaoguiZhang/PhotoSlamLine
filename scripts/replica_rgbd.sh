#!/usr/bin/env bash
set -euo pipefail

# Resolve the project root from this script's location so it can be invoked
# from any working directory (also safe for paths containing spaces).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_DIR="${PROJECT_ROOT}/bin"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
RESULTS_ROOT="${PROJECT_ROOT}/results"

# Dataset root. Override with:
#   DATASET_ROOT=/path/to/dataset bash scripts/replica_rgbd.sh
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"

# Repeat count, sequence list and result-set name. Override with e.g.:
#   RUNS=1 SEQS="office0 office2" RESULT_SET=replica_rgbd_loopfix bash scripts/replica_rgbd.sh
RUNS="${RUNS:-5}"
SEQS="${SEQS:-office0 office1 office2 office3 office4 room0 room1 room2}"
RESULT_SET="${RESULT_SET:-replica_rgbd}"

# Runtime ablation switches. Standard configuration: full point-line SLAM with
# the already-accepted standard point loop closing.
#   PHOTO_SLAM_LINE_MODE  : 0=point-only, 1=line-frontend, 2=full point-line
#   PHOTO_SLAM_LINE_LOOP  : 0=standard point loop (accepted); 1/2 = stages B/C (not implemented yet)
#   PHOTO_SLAM_SHADOW_LINE: 1 = skip MapLine create/cull/fuse (S1 isolation)
#   PHOTO_SLAM_SLAM_ONLY  : 1 = skip Gaussian mapper (currently only honoured by examples/tum_rgbd.cpp)
PHOTO_SLAM_LINE_MODE="${PHOTO_SLAM_LINE_MODE:-2}"
PHOTO_SLAM_LINE_LOOP="${PHOTO_SLAM_LINE_LOOP:-0}"
PHOTO_SLAM_SHADOW_LINE="${PHOTO_SLAM_SHADOW_LINE:-0}"
PHOTO_SLAM_SLAM_ONLY="${PHOTO_SLAM_SLAM_ONLY:-0}"

require_dataset() {
    if [ ! -d "$1" ]; then
        echo "ERROR: dataset directory not found: $1" >&2
        echo "       Set DATASET_ROOT to the parent of the 'replica' folder." >&2
        exit 1
    fi
}

REPLICA_ROOT="${DATASET_ROOT}/replica"

# --- startup checks ----------------------------------------------------------
if [ ! -x "${BIN_DIR}/replica_rgbd" ]; then
    echo "ERROR: executable not found or not executable: ${BIN_DIR}/replica_rgbd" >&2
    exit 1
fi
if [ ! -f "${VOCAB}" ]; then
    echo "ERROR: vocabulary not found: ${VOCAB}" >&2
    exit 1
fi
case "${RUNS}" in
    ''|*[!0-9]*)
        echo "ERROR: RUNS must be a positive integer, got: '${RUNS}'" >&2
        exit 1
        ;;
esac
if [ "${RUNS}" -lt 1 ]; then
    echo "ERROR: RUNS must be >= 1, got: ${RUNS}" >&2
    exit 1
fi

GIT_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "${PROJECT_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
echo "=== Replica RGB-D runner ==="
echo "  git        : ${GIT_BRANCH} @ ${GIT_COMMIT}"
echo "  runs       : ${RUNS}"
echo "  seqs       : ${SEQS}"
echo "  result_set : ${RESULT_SET}"
echo "  mode       : LINE_MODE=${PHOTO_SLAM_LINE_MODE} LINE_LOOP=${PHOTO_SLAM_LINE_LOOP} SHADOW_LINE=${PHOTO_SLAM_SHADOW_LINE} SLAM_ONLY=${PHOTO_SLAM_SLAM_ONLY}"
echo "============================"

for i in $(seq 1 "${RUNS}")
do
    for seq in ${SEQS}
    do
        dataset_dir="${REPLICA_ROOT}/${seq}"
        require_dataset "${dataset_dir}"

        orb_cfg="${PROJECT_ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${seq}.yaml"
        gau_cfg="${PROJECT_ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
        if [ ! -f "${orb_cfg}" ]; then
            echo "ERROR: ORB-SLAM3 settings not found: ${orb_cfg}" >&2
            exit 1
        fi
        if [ ! -f "${gau_cfg}" ]; then
            echo "ERROR: gaussian mapper settings not found: ${gau_cfg}" >&2
            exit 1
        fi

        if [ "${RUNS}" -gt 1 ]; then
            out_dir="${RESULTS_ROOT}/${RESULT_SET}/run_${i}/${seq}"
        else
            out_dir="${RESULTS_ROOT}/${RESULT_SET}/${seq}"
        fi
        mkdir -p "${out_dir}"

        export PHOTO_SLAM_LINE_MODE PHOTO_SLAM_LINE_LOOP PHOTO_SLAM_SHADOW_LINE PHOTO_SLAM_SLAM_ONLY
        echo "=== [${i}/${RUNS}] seq=${seq} -> ${out_dir} ==="

        set +e
        "${BIN_DIR}/replica_rgbd" \
            "${VOCAB}" \
            "${orb_cfg}" \
            "${gau_cfg}" \
            "${dataset_dir}" \
            "${out_dir}" \
            no_viewer > "${out_dir}/run.log" 2>&1
        rc=$?
        set -e
        echo "EXIT=${rc}" >> "${out_dir}/run.log"
        if [ "${rc}" -ne 0 ]; then
            echo "ERROR: replica_rgbd exited with ${rc} for seq=${seq} (see ${out_dir}/run.log)" >&2
            exit "${rc}"
        fi
    done
done
