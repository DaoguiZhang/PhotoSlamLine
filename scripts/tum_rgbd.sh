#!/usr/bin/env bash
set -u

# Resolve the project root from this script's location so it can be invoked
# from any working directory (also safe for paths containing spaces).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN_DIR="${PROJECT_ROOT}/bin"
VOCAB="${PROJECT_ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
RESULTS_ROOT="${PROJECT_ROOT}/results"

# Dataset root. Override with:
#   DATASET_ROOT=/path/to/dataset bash scripts/tum_rgbd.sh
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"

require_dataset() {
    if [ ! -d "$1" ]; then
        echo "ERROR: dataset directory not found: $1" >&2
        echo "       Set DATASET_ROOT to the parent of the 'tum' folder." >&2
        exit 1
    fi
}

TUM_ROOT="${DATASET_ROOT}/tum"
SEQS="freiburg1_desk freiburg2_xyz freiburg3_long_office_household"

for i in 0 1 2 3 4
do
    for seq in ${SEQS}
    do
        dataset_dir="${TUM_ROOT}/rgbd_dataset_${seq}"
        require_dataset "${dataset_dir}"

        assoc_file="${PROJECT_ROOT}/cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_${seq}.txt"
        if [ ! -f "${assoc_file}" ]; then
            echo "ERROR: associations file not found: ${assoc_file}" >&2
            exit 1
        fi

        out_dir="${RESULTS_ROOT}/tum_rgbd_${i}/rgbd_dataset_${seq}"
        mkdir -p "${out_dir}"

        "${BIN_DIR}/tum_rgbd" \
            "${VOCAB}" \
            "${PROJECT_ROOT}/cfg/ORB_SLAM3/RGB-D/TUM/tum_${seq}.yaml" \
            "${PROJECT_ROOT}/cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml" \
            "${dataset_dir}" \
            "${assoc_file}" \
            "${out_dir}" \
            no_viewer
    done
done
