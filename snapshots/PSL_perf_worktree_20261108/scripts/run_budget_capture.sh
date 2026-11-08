#!/usr/bin/env bash
# ============================================================================
# mapper-budget-v1 / 二: ONE full office0 online capture.
#
# Runs the normal online entry point and records, in a single pass:
#   * creation_audit_events.csv / _summary.txt : net creation accounting
#   * creation_log.bin                         : every accepted creation batch
#                                                (kind, iteration, xyz, rgb, dir)
#   * linepop_linepop.csv                      : line-Gaussian population per stage
#   * linepop_lineinherit.csv                  : is_line_/anchor/dir alignment
#                                                after every topology operation
#   * linesample_linesample.csv                : line-sampling candidates and
#                                                per-reason rejections
#   * snapshots/model_step<it>.pt              : full resumable tensors + state
#
# The line STRUCTURAL loss is off for this run (PHOTO_SLAM_LINE_LOSS_EVERY=0) so
# the capture characterises the common pipeline, not the structural ablation.
# Nothing about the front end, the keyframe policy or the densify schedule is
# changed. This run is a CAPTURE: its wall clock is not a speed result.
#
# usage: run_budget_capture.sh [SEQ] [OUT_DIR]
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEQ="${1:-office0}"
OUT="${2:-${ROOT}/abl_out/budget_capture_${SEQ}}"

DATASET_ROOT="${DATASET_ROOT:-/workspace/code/SEGS-SLAM/datasets}"
DS="${DATASET_ROOT}/replica/${SEQ}"
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/Replica/${SEQ}.yaml"
GAU_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"

[[ -d "${DS}/results" ]] || { echo "missing dataset ${DS}/results" >&2; exit 1; }
mkdir -p "${OUT}/snapshots"
cp -- "${ORB_CFG}" "${OUT}/orb_settings.used.yaml"
cp -- "${GAU_CFG}" "${OUT}/gaussian_settings.used.yaml"
git -C "${ROOT}" rev-parse HEAD > "${OUT}/git_commit.txt"
sha256sum "${ROOT}/bin/replica_rgbd" "${ROOT}/ORB-SLAM3/lib/libORB_SLAM3.so" \
    > "${OUT}/binaries.sha256"

# --- capture switches (all default off in a normal run) ---
export PHOTO_SLAM_CREATION_AUDIT="${OUT}/creation_audit"
export PHOTO_SLAM_CREATION_LOG="${OUT}/creation_log.bin"
export PHOTO_SLAM_LINE_POP_AUDIT="${OUT}/linepop"
export PHOTO_SLAM_LINE_SAMPLE_AUDIT="${OUT}/linesample"
export PHOTO_SLAM_SNAPSHOT_DIR="${OUT}/snapshots"
export PHOTO_SLAM_SNAPSHOT_AT="${SNAPSHOT_AT:-9000,20000}"
# line structural loss OFF for the capture
export PHOTO_SLAM_LINE_LOSS_EVERY=0
export PHOTO_SLAM_LINE_MODE="${PHOTO_SLAM_LINE_MODE:-2}"
export PHOTO_SLAM_LINE_LOOP="${PHOTO_SLAM_LINE_LOOP:-0}"
unset PHOTO_SLAM_GPU_PROF || true

echo "[capture] seq=${SEQ} frames=$(ls "${DS}/results" | grep -c '^frame') out=${OUT}"
cd "${ROOT}"
"${ROOT}/bin/replica_rgbd" \
    "${VOCAB}" "${ORB_CFG}" "${GAU_CFG}" "${DS}" "${OUT}" no_viewer \
    2>&1 | tee "${OUT}/run.log"

echo "[capture] done"
ls -la "${OUT}/snapshots"
