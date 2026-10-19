#!/usr/bin/env bash
# ============================================================================
# Interleaved A/B/C repeats on fr3_structure_notexture_near.
#
# Runs are interleaved (r1:A,B,C  r2:A,B,C  ...) so that any slow machine drift
# is spread evenly across conditions instead of being confounded with them.
#
# IMPORTANT: PHOTO_SLAM_SEED only seeds the Gaussian mapper, which is skipped
# here (SLAM_ONLY=1). ORB-SLAM3's concurrent LocalMapping / LoopClosing threads
# make the SLAM run inherently non-repeatable, so these repeats measure the
# run-to-run SPREAD; they cannot drive it to zero.
#
# Usage:  RUNS=5 bash scripts/fr3_abc_interleave.sh /tmp/lr_abc_interleave
# ============================================================================
set -uo pipefail

RUNS="${RUNS:-5}"
OUTBASE="${1:-/tmp/lr_abc_interleave}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEQ="/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_structure_notexture_near"
GT="${SEQ}/groundtruth.txt"

mkdir -p "${OUTBASE}"

for r in $(seq 1 "${RUNS}"); do
    for c in A B C; do
        OUT="${OUTBASE}/r${r}_${c}"
        SLAM_ONLY=1 bash "${ROOT}/scripts/fr3_line_diag_abc.sh" "${c}" "${OUT}" "${r}"
        if [ -f "${OUT}/CameraTrajectory_TUM.txt" ]; then
            python3 "${ROOT}/scripts/eval_tum_se3_v2.py" "${GT}" \
                "${OUT}/CameraTrajectory_TUM.txt" "${OUT}/ate_se3" \
                > "${OUT}/ate_se3.txt" 2>&1
            echo "r=${r} c=${c} $(grep -m1 '^\[ATE v2\]' "${OUT}/ate_se3.txt")"
        else
            echo "r=${r} c=${c} NO TRAJECTORY"
        fi
    done
done

python3 "${ROOT}/scripts/aggregate_abc.py" "${OUTBASE}"
