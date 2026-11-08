#!/usr/bin/env bash
# ============================================================================
# SEPARATELY-LABELLED STRESS TEST driver (NOT a natural-sequence experiment).
#
# The natural TUM/ETH3D sequences never make the point-only gate fail, so the
# rescue cannot be exercised there. This driver induces that failure with a
# FIXED point-observation reduction that is identical in every condition and
# never touches line observations:
#
#   PHOTO_SLAM_STRESS_DROP_POINT_RATIO  fraction of point observations dropped
#   PHOTO_SLAM_STRESS_START_FRAME       first frame id the drop applies to
#                                       (keeps initialization untouched)
#
# Conditions: P = A-prime (LINE_MODE=1, same entry, no line edges in optimizers),
#             B = LINE_MODE=2 rescue OFF, C = LINE_MODE=2 rescue ON.
#
# Usage: RUNS=3 STRESS_DROP=0.7 STRESS_START=30 \
#          bash scripts/stress_rescue_repeats.sh /tmp/lr_stress "near rgbd"
# ============================================================================
set -uo pipefail

RUNS="${RUNS:-3}"
OUTBASE="${1:?usage: stress_rescue_repeats.sh <out_base> [\"scene mode\" ...]}"
shift || true
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

COMBOS=("$@")
if [ ${#COMBOS[@]} -eq 0 ]; then
    COMBOS=("near rgbd")
fi

# Fixed, identical reduction rule for every condition.
export PHOTO_SLAM_STRESS_DROP_POINT_RATIO="${STRESS_DROP:-0.7}"
export PHOTO_SLAM_STRESS_START_FRAME="${STRESS_START:-30}"

echo "[stress] drop_ratio=${PHOTO_SLAM_STRESS_DROP_POINT_RATIO} start_frame=${PHOTO_SLAM_STRESS_START_FRAME} runs=${RUNS} combos=${COMBOS[*]}"
mkdir -p "${OUTBASE}"

for r in $(seq 1 "${RUNS}"); do
    for combo in "${COMBOS[@]}"; do
        # shellcheck disable=SC2086
        set -- ${combo}
        scene="$1"
        mode="$2"
        for c in P B C; do
            d="${OUTBASE}/r${r}/${scene}_${mode}_${c}"
            bash "${ROOT}/scripts/fr3_ablation_run.sh" "${scene}" "${mode}" "${c}" "${d}" "${r}"
            rc=$?
            n=$(wc -l < "${d}/CameraTrajectory_TUM.txt" 2>/dev/null || echo NA)
            resc=$(grep -c 'oldPtVerdict=FAIL new=OK' "${d}/log.txt" 2>/dev/null || echo 0)
            echo "r=${r} ${scene} ${mode} ${c} rc=${rc} traj=${n} rescued=${resc}"
        done
    done
done

python3 "${ROOT}/scripts/ablation_common_window.py" "${OUTBASE}" --out "${OUTBASE}"
