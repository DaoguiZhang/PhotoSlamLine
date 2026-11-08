#!/usr/bin/env bash
# ============================================================================
# One A/P/B/C line-ablation run on an ETH3D RGB-D sequence, with all line
# diagnostics enabled and Gaussian mapping disabled (SLAM-only, so no
# image-quality metric is produced or reported).
#
#   cond A : LINE_MODE=0  point-only, original entry
#   cond P : LINE_MODE=1  A-prime: same entry as B/C, no line edges in the
#                         optimizers, lines still detected/matched
#   cond B : LINE_MODE=2, RESCUE=0
#   cond C : LINE_MODE=2, RESCUE=1
#
# Any PHOTO_SLAM_STRESS_* variables already exported by the caller are forwarded
# unchanged (that is how the separately-labelled stress test is driven).
#
# Usage: eth3d_ablation_run.sh <eth3d_seq> <A|P|B|C> <out_dir> [run_idx]
# ============================================================================
set -uo pipefail

SEQ_NAME="${1:?usage: eth3d_ablation_run.sh <eth3d_seq> <A|P|B|C> <out_dir> [run_idx]}"
COND="${2:?missing condition}"
OUTDIR="${3:?missing out_dir}"
RUN_IDX="${4:-0}"

case "${COND}" in A|P|B|C) ;; *) echo "bad condition '${COND}'" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
DATA="/workspace/code/Photo-SLAM/datasets/eth3d/${SEQ_NAME}"
ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_${SEQ_NAME}.yaml"
GAUSS_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_${SEQ_NAME}.yaml"
ASSO="${DATA}/associated.txt"

export LD_LIBRARY_PATH="${ROOT}/ORB-SLAM3/lib:${ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

for f in "${ROOT}/bin/eth3d_rgbd" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${ASSO}"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}" >&2; exit 3; }
done

case "${COND}" in
    A) export PHOTO_SLAM_LINE_MODE=0; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    P) export PHOTO_SLAM_LINE_MODE=1; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1 ;;
esac

export PHOTO_SLAM_DEBUG_LINE_GATE=1
export PHOTO_SLAM_DEBUG_LINE_RESCUE=1
export PHOTO_SLAM_DEBUG_LINE_SCAN=1
export PHOTO_SLAM_DEBUG_LINE_EDGES=1
export PHOTO_SLAM_SLAM_ONLY=1

mkdir -p "${OUTDIR}"
{
    echo "scene=${SEQ_NAME}"
    echo "mode=rgbd"
    echo "cond=${COND}"
    echo "run_idx=${RUN_IDX}"
    echo "line_mode=${PHOTO_SLAM_LINE_MODE}"
    echo "rescue=${PHOTO_SLAM_LINE_RESCUE}"
    echo "orb_cfg=${ORB_CFG}"
    echo "gauss_cfg=${GAUSS_CFG}"
    echo "sequence=${DATA}"
    echo "association=${ASSO}"
    if [ "${PHOTO_SLAM_STRESS_DROP_POINT_RATIO:-0}" != "0" ]; then
        echo "stress_drop_point_ratio=${PHOTO_SLAM_STRESS_DROP_POINT_RATIO}"
        echo "stress_start_frame=${PHOTO_SLAM_STRESS_START_FRAME:-0}"
    fi
} > "${OUTDIR}/run.meta"

START=$(date +%s)
# Bounded per-run wall clock. A stuck run (e.g. the line-aware map-merge path can
# livelock or segfault) is killed instead of burning hours, and is then marked
# INCOMPLETE so it can never be aggregated into metrics.
timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT:-1800}" \
    "${ROOT}/bin/eth3d_rgbd" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" \
    "${DATA}" "${ASSO}" "${OUTDIR}" no_viewer > "${OUTDIR}/log.txt" 2>&1
RC=$?
END=$(date +%s)
LAST_FRAME=$(grep -oE 'frame=[0-9]+' "${OUTDIR}/log.txt" 2>/dev/null | sed 's/frame=//' | sort -n | tail -1)
SAVED=$(grep -c "Saving keyframe trajectory" "${OUTDIR}/log.txt" 2>/dev/null)
# complete: clean exit AND the trajectory was actually written at shutdown.
if [ "${RC}" = "0" ] && [ "${SAVED:-0}" -ge 1 ]; then COMPLETE=1; else COMPLETE=0; fi
{
    echo "exit=${RC}"
    echo "wall_s=$((END-START))"
    echo "last_frame=${LAST_FRAME:-0}"
    echo "saved_keyframe_trajectory=${SAVED:-0}"
    echo "timeout_s=${RUN_TIMEOUT:-1800}"
    echo "complete=${COMPLETE}"
} >> "${OUTDIR}/run.meta"
exit ${RC}
