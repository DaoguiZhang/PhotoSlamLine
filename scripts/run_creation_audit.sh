#!/usr/bin/env bash
# ============================================================================
# perf-stage3 / 二: online Gaussian creation-source audit (one full online run)
#
# Runs the NORMAL online entry point (bin/replica_rgbd) on one Replica sequence
# with the line STRUCTURAL loss switched off (PHOTO_SLAM_LINE_LOSS_EVERY=0) so
# that the audit characterises the common pipeline rather than the structural
# ablation. Front end / keyframe policy / online densify schedule are untouched.
#
# The creation audit itself is host-only (no per-step device sync) and reports:
#   <out>/creation_audit_events.csv   one row per creation/removal event
#   <out>/creation_audit_summary.txt  running totals, rewritten per event
#
# Usage:
#   bash scripts/run_creation_audit.sh [SEQ] [OUT_DIR]
#
# NOTE: this run exists to AUDIT creation sources. Its wall-clock time is NOT a
# formal speed-up result (the audit itself is off in all timing comparisons).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SEQ="${1:-office0}"
OUT="${2:-${ROOT}/abl_out/online_audit_${SEQ}}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
DS="${DATASET_ROOT}/replica/${SEQ}"
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${SEQ}.yaml"
GAU_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"

for f in "${VOCAB}" "${ORB_CFG}" "${GAU_CFG}"; do
    [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ -d "${DS}/results" ]] || { echo "missing dataset: ${DS}/results" >&2; exit 1; }

mkdir -p "${OUT}"
cp -- "${ORB_CFG}" "${OUT}/orb_settings.used.yaml"
cp -- "${GAU_CFG}" "${OUT}/gaussian_settings.used.yaml"
git -C "${ROOT}" rev-parse HEAD > "${OUT}/git_commit.txt" 2>/dev/null || true

echo "[audit] sequence=${SEQ} frames=$(ls "${DS}/results" | grep -c '^frame')"
echo "[audit] out=${OUT}"

export PHOTO_SLAM_CREATION_AUDIT="${OUT}/creation_audit"
export PHOTO_SLAM_LINE_LOSS_EVERY=0   # line structural loss OFF (audit run only)
export PHOTO_SLAM_LINE_MODE="${PHOTO_SLAM_LINE_MODE:-2}"
export PHOTO_SLAM_LINE_LOOP="${PHOTO_SLAM_LINE_LOOP:-0}"
unset PHOTO_SLAM_GPU_PROF || true      # keep the hot loop unperturbed by design

cd "${ROOT}"
"${ROOT}/bin/replica_rgbd" \
    "${VOCAB}" "${ORB_CFG}" "${GAU_CFG}" "${DS}" "${OUT}" no_viewer \
    2>&1 | tee "${OUT}/run.log"

echo "[audit] done -> ${OUT}/creation_audit_summary.txt"
