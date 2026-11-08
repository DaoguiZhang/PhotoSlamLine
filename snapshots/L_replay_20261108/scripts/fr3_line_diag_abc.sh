#!/usr/bin/env bash
# ============================================================================
# A/B/C line ablation on fr3_structure_notexture_near WITH full diagnostics.
#
#   A : PHOTO_SLAM_LINE_MODE=0                  point-only baseline
#   B : PHOTO_SLAM_LINE_MODE=2, RESCUE=0        line frontend + line edges in
#                                               pose optimisation, rescue OFF
#   C : PHOTO_SLAM_LINE_MODE=2, RESCUE=1        same as B but rescue ON
#
# B vs C isolates the RESCUE path only (identical line frontend and PO line
# edges).  Every run emits the three diagnostic streams that make the question
# answerable per frame:
#     [LineGate]      old control flow verdict BEFORE any bypass
#     [LineRescue]    post-optimisation joint verdict
#     [LineEdgeDiag]  per-PoseOptimizationWithLine point/line edge counts
#
# Usage:
#   bash scripts/fr3_line_diag_abc.sh <A|B|C> <out_dir> [run_index] [assoc_override]
#
# Environment:
#   FRAMES=<n>          truncate the association file to the first n pairs
#   SLAM_ONLY=1         skip Gaussian mapping (faster, tracking unchanged)
#   EXTRA_ENV=1         forward anything already exported
# ============================================================================
set -uo pipefail

COND="${1:?usage: fr3_line_diag_abc.sh <A|B|C> <out_dir> [run_index] [assoc]}"
OUTDIR="${2:?missing out_dir}"
RUN_IDX="${3:-0}"
ASSO_OVERRIDE="${4:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BIN="${ROOT}/bin/tum_rgbd"
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_structure_notexture_near_lowfast.yaml"
GAUSS_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_structure_notexture_near.yaml"
SEQ="/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_structure_notexture_near"
ASSO="${ROOT}/cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_structure_notexture_near.txt"

export LD_LIBRARY_PATH="${ROOT}/ORB-SLAM3/lib:${ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# --- condition wiring ------------------------------------------------------
case "${COND}" in
    A) export PHOTO_SLAM_LINE_MODE=0; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1 ;;
    *) echo "unknown condition '${COND}' (expected A|B|C)" >&2; exit 2 ;;
esac

# --- diagnostics (all three streams) ---------------------------------------
export PHOTO_SLAM_DEBUG_LINE_GATE=1
export PHOTO_SLAM_DEBUG_LINE_RESCUE=1
export PHOTO_SLAM_DEBUG_LINE_EDGES=1

# --- optional overrides ----------------------------------------------------
if [ -n "${ASSO_OVERRIDE}" ]; then ASSO="${ASSO_OVERRIDE}"; fi
if [ -n "${FRAMES:-}" ]; then
    ASSO_TMP="${OUTDIR}/_assoc_first${FRAMES}.txt"
    mkdir -p "${OUTDIR}"
    head -n "${FRAMES}" "${ASSO}" > "${ASSO_TMP}"
    ASSO="${ASSO_TMP}"
fi
if [ "${SLAM_ONLY:-0}" = "1" ]; then export PHOTO_SLAM_SLAM_ONLY=1; fi

mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/log.txt"

echo "[fr3-abc] cond=${COND} run=${RUN_IDX} mode=${PHOTO_SLAM_LINE_MODE} "\
"rescue=${PHOTO_SLAM_LINE_RESCUE} assoc=${ASSO} -> ${OUTDIR}" | tee "${LOG}.meta"

START=$(date +%s)
"${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}" "${ASSO}" "${OUTDIR}" \
    no_viewer > "${LOG}" 2>&1
RC=$?
END=$(date +%s)
echo "[fr3-abc] exit=${RC} wall=$((END-START))s" | tee -a "${LOG}.meta"
exit ${RC}
