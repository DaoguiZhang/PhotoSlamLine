#!/usr/bin/env bash
# ============================================================================
# GATED acceptance group: run ONE group (one mode x one stress level x the three
# conditions) and run the export acceptance gate after EVERY round.
#
#   * fail fast: the first round whose exit code is non-zero, that is incomplete,
#     or that fails the export gate ABORTS the group - no further rounds are
#     started, so a broken basis cannot silently consume the whole budget.
#   * EXIT=0 IS NOT ACCEPTANCE: acceptance is decided only by
#     scripts/export_acceptance.py from the artefacts the round wrote.
#   * unique artefact paths: every path carries mode + level + condition + repeat
#     (+ tag), and a non-empty output directory is refused instead of reused, so
#     two groups can never overwrite each other.
#
# usage: acceptance_group.sh <rgbd|mono> <0|30|50> <tag> <outdir> [reps]
#        level 0 = stress-free (no point drop at all)
# ============================================================================
set -uo pipefail

MODE="${1:?usage: acceptance_group.sh <rgbd|mono> <0|30|50> <tag> <outdir> [reps]}"
LEVEL="${2:?missing level}"
TAG="${3:?missing tag}"
BASE="${4:?missing outdir}"
REPS="${5:-1}"

case "${MODE}"  in rgbd|mono) ;; *) echo "bad mode '${MODE}'" >&2; exit 2 ;; esac
case "${LEVEL}" in 0|30|50) ;;   *) echo "bad level '${LEVEL}'" >&2; exit 2 ;; esac

ROOT=/workspace/code/Photo-SLAM-L
SEQ=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3
VOC="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"
GATE="${ROOT}/scripts/export_acceptance.py"

if [ "${MODE}" = "rgbd" ]; then
    BIN="${ROOT}/bin/eth3d_rgbd"
    ORB_CFG="${ROOT}/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml"
    GAUSS_CFG="${ROOT}/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_desk_3.yaml"
    ASSO="${SEQ}/associated.txt"
else
    BIN="${ROOT}/bin/eth3d_mono"
    ORB_CFG="${ROOT}/cfg/ORB_SLAM3/Monocular/ETH3D/eth3d_desk_3.yaml"
    GAUSS_CFG="${ROOT}/cfg/gaussian_mapper/Monocular/ETH3D/eth3d_desk_3.yaml"
    ASSO=""
fi

export RUN_TIMEOUT=600
export PHOTO_SLAM_DEBUG_MERGE=1
export PHOTO_SLAM_SLAM_ONLY=1
export PHOTO_SLAM_DEBUG_LINE_GATE=1
export PHOTO_SLAM_DEBUG_LINE_RESCUE=1
export PHOTO_SLAM_DEBUG_LINE_SCAN=1
export PHOTO_SLAM_DEBUG_LINE_EDGES=1
export LD_LIBRARY_PATH="${ROOT}/ORB-SLAM3/lib:${ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${ROOT}/lib"

for f in "${BIN}" "${VOC}" "${ORB_CFG}" "${GAUSS_CFG}" "${GATE}" "${SEQ}/rgb.txt"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}" >&2; exit 3; }
done

# Unique output path: refuse to write into a directory that already holds runs.
if [ -d "${BASE}" ] && [ -n "$(ls -A "${BASE}" 2>/dev/null)" ]; then
    echo "REFUSING to reuse non-empty outdir: ${BASE}" >&2
    exit 2
fi
mkdir -p "${BASE}"
SUMMARY="${BASE}/GROUP_RESULT.txt"
: > "${SUMMARY}"
echo "group mode=${MODE} level=${LEVEL} tag=${TAG} reps=${REPS} base=${BASE} start=$(date -Is)" | tee -a "${SUMMARY}"

if [ "${LEVEL}" = "0" ]; then
    unset PHOTO_SLAM_STRESS_DROP_POINT_RATIO
    unset PHOTO_SLAM_STRESS_START_FRAME
    DROP_LABEL="none"
else
    export PHOTO_SLAM_STRESS_DROP_POINT_RATIO="0.${LEVEL}"
    export PHOTO_SLAM_STRESS_START_FRAME=30
    DROP_LABEL="${LEVEL}%"
fi
export PHOTO_SLAM_GATE_TAG="${TAG}"

GROUP_FAIL=0
for r in $(seq 1 "${REPS}"); do
    for cond in P B C; do
        case "${cond}" in
            P) export PHOTO_SLAM_LINE_MODE=1; export PHOTO_SLAM_LINE_RESCUE=0 ;;
            B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0 ;;
            C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1 ;;
        esac

        # Unique path: mode, level, tag, condition and repeat are all in it.
        d="${BASE}/${MODE}_L${LEVEL}_${TAG}_r${r}_${cond}"
        mkdir -p "${d}"
        {
            echo "scene=desk_3"
            echo "mode=${MODE}"
            echo "cond=${cond}"
            echo "run_idx=${r}"
            echo "level=${LEVEL}"
            echo "tag=${TAG}"
            echo "line_mode=${PHOTO_SLAM_LINE_MODE}"
            echo "rescue=${PHOTO_SLAM_LINE_RESCUE}"
            echo "orb_cfg=${ORB_CFG}"
            echo "gauss_cfg=${GAUSS_CFG}"
            echo "sequence=${SEQ}"
            echo "association=${ASSO}"
            echo "stress_drop_point_ratio=${PHOTO_SLAM_STRESS_DROP_POINT_RATIO:-0}"
            echo "stress_start_frame=${PHOTO_SLAM_STRESS_START_FRAME:-0}"
            echo "basis=v2-export-coherent+instrumented"
        } > "${d}/run.meta"

        START=$(date +%s)
        if [ "${MODE}" = "rgbd" ]; then
            timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT}" \
                "${BIN}" "${VOC}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}" "${ASSO}" "${d}" no_viewer \
                > "${d}/log.txt" 2>&1
        else
            timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT}" \
                "${BIN}" "${VOC}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}" "${d}" no_viewer \
                > "${d}/log.txt" 2>&1
        fi
        RC=$?
        END=$(date +%s)

        LAST=$(grep -oE 'frame=[0-9]+' "${d}/log.txt" 2>/dev/null | sed 's/frame=//' | sort -n | tail -1)
        SAVED=$(grep -c "Saving keyframe trajectory" "${d}/log.txt" 2>/dev/null)
        SAVED=${SAVED:-0}
        COMPLETE=0
        [ "${RC}" = "0" ] && [ "${SAVED}" -ge 1 ] && COMPLETE=1

        EXPORT_LINE=$(grep "SaveTrajectoryTUM: exported map=" "${d}/log.txt" 2>/dev/null | tail -1)
        EX_MAP=$(echo "${EXPORT_LINE}" | sed -n 's/.*exported map=\([-0-9]*\).*/\1/p')
        EX_WRITTEN=$(echo "${EXPORT_LINE}" | sed -n 's/.*written=\([0-9]*\).*/\1/p')
        EX_LOST=$(echo "${EXPORT_LINE}" | sed -n 's/.*skipped_lost=\([0-9]*\).*/\1/p')
        EX_OTHER=$(echo "${EXPORT_LINE}" | sed -n 's/.*skipped_other_map=\([0-9]*\).*/\1/p')
        {
            echo "exit=${RC}"
            echo "wall_s=$((END-START))"
            echo "last_frame=${LAST:-0}"
            echo "saved_keyframe_trajectory=${SAVED}"
            echo "timeout_s=${RUN_TIMEOUT}"
            echo "complete=${COMPLETE}"
            echo "export_map=${EX_MAP:-NA}"
            echo "export_written=${EX_WRITTEN:-NA}"
            echo "export_skipped_lost=${EX_LOST:-NA}"
            echo "export_skipped_other_map=${EX_OTHER:-NA}"
        } >> "${d}/run.meta"

        # ---- gates -------------------------------------------------------
        if [ "${RC}" != "0" ] || [ "${COMPLETE}" != "1" ]; then
            echo "round=${d} exit=${RC} complete=${COMPLETE} -> GROUP FAIL (run did not complete)" | tee -a "${SUMMARY}"
            GROUP_FAIL=1
            break 3
        fi

        python3 "${GATE}" "${d}" > "${d}/acceptance_run.txt" 2>&1
        GRC=$?
        cat "${d}/acceptance_run.txt" | tee -a "${SUMMARY}"
        if [ "${GRC}" != "0" ]; then
            echo "round=${d} EXPORT GATE FAILED (rc=${GRC}) -> aborting group, no further rounds" | tee -a "${SUMMARY}"
            GROUP_FAIL=1
            break 3
        fi
        echo "round=${d} drop=${DROP_LABEL} exit=${RC} complete=1 export_map=${EX_MAP:-NA} written=${EX_WRITTEN:-NA} other_map=${EX_OTHER:-NA} GATE=PASS" | tee -a "${SUMMARY}"
    done
done

if [ "${GROUP_FAIL}" = "0" ]; then
    echo "GROUP_RESULT=PASS mode=${MODE} level=${LEVEL} tag=${TAG}" | tee -a "${SUMMARY}"
else
    echo "GROUP_RESULT=FAIL mode=${MODE} level=${LEVEL} tag=${TAG}" | tee -a "${SUMMARY}"
fi
echo "end=$(date -Is)" | tee -a "${SUMMARY}"
exit ${GROUP_FAIL}
