#!/usr/bin/env bash
# ============================================================================
# Interleaved A/B/C repeats over scene x camera-mode combinations.
#
# Runs are interleaved (r1: combo1 A,B,C  combo2 A,B,C ...  r2: ...) so slow
# machine drift cannot be confounded with a condition.
#
# NOTE on repeatability: PHOTO_SLAM_SEED only seeds the Gaussian mapper, which
# is skipped here (SLAM-only). ORB-SLAM3's concurrent LocalMapping / LoopClosing
# threads make a run inherently non-repeatable, so these repeats measure the
# run-to-run SPREAD; they cannot drive it to zero.
#
# Layout produced: <out_base>/r<k>/<scene>_<mode>_<cond>/{log.txt,trajectory,...}
#
# Usage:
#   RUNS=5 bash scripts/fr3_ablation_repeats.sh /tmp/lr_ablation
#   RUNS=5 bash scripts/fr3_ablation_repeats.sh /tmp/lr_ablation "near rgbd" "far mono"
# ============================================================================
set -uo pipefail

RUNS="${RUNS:-5}"
OUTBASE="${1:?usage: fr3_ablation_repeats.sh <out_base> [\"scene mode\" ...]}"
shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMBOS=("$@")
if [ ${#COMBOS[@]} -eq 0 ]; then
    COMBOS=("near rgbd" "near mono" "far rgbd" "far mono")
fi

mkdir -p "${OUTBASE}"
for r in $(seq 1 "${RUNS}"); do
    for combo in "${COMBOS[@]}"; do
        # shellcheck disable=SC2086
        set -- ${combo}
        scene="$1"; mode="$2"
        for c in A B C; do
            d="${OUTBASE}/r${r}/${scene}_${mode}_${c}"
            bash "${ROOT}/scripts/fr3_ablation_run.sh" "${scene}" "${mode}" "${c}" "${d}" "${r}"
            rc=$?
            n=$(wc -l < "${d}/CameraTrajectory_TUM.txt" 2>/dev/null || echo NA)
            echo "r=${r} ${scene} ${mode} ${c} rc=${rc} traj_lines=${n}"
        done
    done
done

python3 "${ROOT}/scripts/ablation_aggregate.py" "${OUTBASE}" --out "${OUTBASE}"
