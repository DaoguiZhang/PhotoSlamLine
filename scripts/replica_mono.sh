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
#   DATASET_ROOT=/path/to/dataset bash scripts/replica_mono.sh
DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"

require_dataset() {
    if [ ! -d "$1" ]; then
        echo "ERROR: dataset directory not found: $1" >&2
        echo "       Set DATASET_ROOT to the parent of the 'replica' folder." >&2
        exit 1
    fi
}

REPLICA_ROOT="${DATASET_ROOT}/replica"
SEQS="office0 office1 office2 office3 office4 room0 room1 room2"

for i in 0 1 2 3 4
do
    for seq in ${SEQS}
    do
        dataset_dir="${REPLICA_ROOT}/${seq}"
        require_dataset "${dataset_dir}"

        out_dir="${RESULTS_ROOT}/replica_mono_${i}/${seq}"
        mkdir -p "${out_dir}"

        "${BIN_DIR}/replica_mono" \
            "${VOCAB}" \
            "${PROJECT_ROOT}/cfg/ORB_SLAM3/Monocular/Replica/${seq}.yaml" \
            "${PROJECT_ROOT}/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml" \
            "${dataset_dir}" \
            "${out_dir}" \
            no_viewer
    done
done
