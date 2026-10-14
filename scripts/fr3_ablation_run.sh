#!/usr/bin/env bash
# ============================================================================
# One A/B/C line-ablation run for a given scene x camera mode, with all three
# line-diagnostic streams enabled and Gaussian mapping disabled (SLAM-only, so
# NO image-quality metric is produced or reported).
#
#   cond A : PHOTO_SLAM_LINE_MODE=0                  point-only baseline
#   cond B : PHOTO_SLAM_LINE_MODE=2, RESCUE=0         lines + line edges in PO
#   cond C : PHOTO_SLAM_LINE_MODE=2, RESCUE=1         identical to B, rescue armed
#
# Monocular does NOT receive depth: bin/tum_mono reads <seq>/rgb.txt and takes no
# association file. RGB-D uses the association file (rgb<->depth pairing).
#
# Usage: fr3_ablation_run.sh <near|far> <mono|rgbd> <A|B|C> <out_dir> [run_idx]
# ============================================================================
set -uo pipefail

SCENE="${1:?usage: fr3_ablation_run.sh <near|far> <mono|rgbd> <A|B|C> <out_dir> [run_idx]}"
MODE="${2:?missing mode}"
COND="${3:?missing condition}"
OUTDIR="${4:?missing out_dir}"
RUN_IDX="${5:-0}"

case "${SCENE}" in near|far) ;; *) echo "bad scene '${SCENE}'" >&2; exit 2 ;; esac
case "${MODE}"  in mono|rgbd) ;; *) echo "bad mode '${MODE}'" >&2; exit 2 ;; esac
case "${COND}"  in A|B|C|P) ;; *) echo "bad condition '${COND}'" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VOCAB="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
SEQ="/workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_structure_notexture_${SCENE}"
ASSO="${ROOT}/cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_structure_notexture_${SCENE}.txt"

export LD_LIBRARY_PATH="${ROOT}/ORB-SLAM3/lib:${ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${ROOT}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [ "${MODE}" = "mono" ]; then
    BIN="${ROOT}/bin/tum_mono"
    ORB_CFG="${ROOT}/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_structure_notexture_${SCENE}_lowfast.yaml"
    GAUSS_CFG="${ROOT}/cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml"
else
    BIN="${ROOT}/bin/tum_rgbd"
    ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_structure_notexture_${SCENE}_lowfast.yaml"
    GAUSS_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_structure_notexture_${SCENE}.yaml"
fi

for f in "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}" >&2; exit 3; }
done

case "${COND}" in
    A) export PHOTO_SLAM_LINE_MODE=0; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    P) # A-prime: SAME entry point and SAME state machine as B/C (the line
       # pipeline), lines detected/matched and MapLines built, but NO line edges
       # in PoseOptimization (Optimizer.cc:2620) nor in Local BA (5674).
       export PHOTO_SLAM_LINE_MODE=1; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0 ;;
    C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1 ;;
esac

# All diagnostics on; SLAM-only so no image-quality metric is produced.
export PHOTO_SLAM_DEBUG_LINE_GATE=1
export PHOTO_SLAM_DEBUG_LINE_RESCUE=1
export PHOTO_SLAM_DEBUG_LINE_SCAN=1
export PHOTO_SLAM_DEBUG_LINE_EDGES=1
export PHOTO_SLAM_SLAM_ONLY=1

mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/log.txt"
{
    echo "scene=${SCENE}"
    echo "mode=${MODE}"
    echo "cond=${COND}"
    echo "run_idx=${RUN_IDX}"
    echo "line_mode=${PHOTO_SLAM_LINE_MODE}"
    echo "rescue=${PHOTO_SLAM_LINE_RESCUE}"
    echo "orb_cfg=${ORB_CFG}"
    echo "gauss_cfg=${GAUSS_CFG}"
    echo "sequence=${SEQ}"
    echo "association=${ASSO}"
    if [ "${PHOTO_SLAM_STRESS_DROP_POINT_RATIO:-0}" != "0" ]; then
        echo "stress_drop_point_ratio=${PHOTO_SLAM_STRESS_DROP_POINT_RATIO}"
        echo "stress_start_frame=${PHOTO_SLAM_STRESS_START_FRAME:-0}"
    fi
} > "${OUTDIR}/run.meta"

START=$(date +%s)
# Bounded per-run wall clock + explicit completeness, so a stuck or crashed run
# can never be aggregated into metrics (see README/SUMMARY for the merge livelock).
if [ "${MODE}" = "mono" ]; then
    timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT:-1800}" \
        "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}" "${OUTDIR}" no_viewer \
        > "${LOG}" 2>&1
else
    timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT:-1800}" \
        "${BIN}" "${VOCAB}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}" "${ASSO}" "${OUTDIR}" no_viewer \
        > "${LOG}" 2>&1
fi
RC=$?
END=$(date +%s)
LAST_FRAME=$(grep -oE 'frame=[0-9]+' "${LOG}" 2>/dev/null | sed 's/frame=//' | sort -n | tail -1)
SAVED=$(grep -c "Saving keyframe trajectory" "${LOG}" 2>/dev/null)
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
