#!/usr/bin/env bash
# ============================================================================
# SEPARATELY-LABELLED SYNTHETIC STRESS FAMILY, regenerated on the corrected
# trajectory-export basis (v2).
#
# Why v2 exists: System::SaveTrajectoryTUM pinned ONE origin taken from the
# atlas' CURRENT map while writing every frame in mlRelativeFramePoses, whose
# reference keyframes may live in other maps (each with its own world frame).
# With several populated maps that mixes world frames and the exported
# trajectory jumps metres between consecutive frames. Measured on the v1 runs:
# 30/30 trajectories had steps > 0.5 m (max 2.5 m), including the stress-free
# run. The exporters now select the map with the most keyframes and skip frames
# whose reference keyframe is elsewhere (the rule SaveTrajectoryEuRoC already
# used), and log the census + counters, which this driver records in run.meta.
#
#   P (A-prime) = PHOTO_SLAM_LINE_MODE=1, PHOTO_SLAM_LINE_RESCUE=0
#   B           = PHOTO_SLAM_LINE_MODE=2, PHOTO_SLAM_LINE_RESCUE=0
#   C           = PHOTO_SLAM_LINE_MODE=2, PHOTO_SLAM_LINE_RESCUE=1
#
# Stress levels are FIXED in advance and identical across conditions; runs are
# interleaved (level -> condition -> repeat) so machine drift cannot be
# confounded with the condition. No tracking-list or rescue-threshold change.
#
# usage: eth3d_desk3_repeats_v2.sh <rgbd|mono> <out_base> [reps]
# ============================================================================
set -uo pipefail

MODE="${1:?usage: eth3d_desk3_repeats_v2.sh <rgbd|mono> <out_base> [reps]}"
BASE="${2:?missing out_base}"
REPS="${3:-5}"

case "${MODE}" in rgbd|mono) ;; *) echo "bad mode '${MODE}'" >&2; exit 2 ;; esac

ROOT=/workspace/code/Photo-SLAM-L
SEQ=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3
VOC="${ROOT}/ORB-SLAM3/Vocabulary/ORBvoc.txt"

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
export PHOTO_SLAM_STRESS_START_FRAME=30
export LD_LIBRARY_PATH="${ROOT}/ORB-SLAM3/lib:${ROOT}/ORB-SLAM3/Thirdparty/g2o/lib:${ROOT}/ORB-SLAM3/Thirdparty/DBoW2/lib:${ROOT}/lib"

for f in "${BIN}" "${VOC}" "${ORB_CFG}" "${GAUSS_CFG}" "${SEQ}/rgb.txt"; do
    [ -e "${f}" ] || { echo "MISSING: ${f}" >&2; exit 3; }
done
if [ "${MODE}" = "rgbd" ]; then
    [ -e "${ASSO}" ] || { echo "MISSING: ${ASSO}" >&2; exit 3; }
fi

mkdir -p "${BASE}"
echo "start=$(date -Is) mode=${MODE} base=${BASE} reps=${REPS} basis=v2-export-coherent"

for r in $(seq 1 "${REPS}"); do
    for level in 30 50; do
        export PHOTO_SLAM_STRESS_DROP_POINT_RATIO="0.${level}"
        for cond in P B C; do
            case "${cond}" in
                P) export PHOTO_SLAM_LINE_MODE=1; export PHOTO_SLAM_LINE_RESCUE=0 ;;
                B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0 ;;
                C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1 ;;
            esac

            d="${BASE}/drop${level}/r${r}_desk_3_${MODE}_${cond}"
            mkdir -p "${d}"
            {
                echo "scene=desk_3"
                echo "mode=${MODE}"
                echo "cond=${cond}"
                echo "run_idx=${r}"
                echo "line_mode=${PHOTO_SLAM_LINE_MODE}"
                echo "rescue=${PHOTO_SLAM_LINE_RESCUE}"
                echo "orb_cfg=${ORB_CFG}"
                echo "gauss_cfg=${GAUSS_CFG}"
                echo "sequence=${SEQ}"
                echo "association=${ASSO}"
                echo "stress_drop_point_ratio=${PHOTO_SLAM_STRESS_DROP_POINT_RATIO}"
                echo "stress_start_frame=${PHOTO_SLAM_STRESS_START_FRAME}"
                echo "basis=v2-export-coherent"
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

            # Exporter accounting (new in v2): which map was exported and how
            # many frame entries were written / skipped, so the evaluation basis
            # of every run is auditable from run.meta alone.
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

            TL=0; UN=0
            if [ -s "${d}/CameraTrajectory_TUM.txt" ]; then
                TL=$(wc -l < "${d}/CameraTrajectory_TUM.txt")
                UN=$(awk '{print $1}' "${d}/CameraTrajectory_TUM.txt" | sort -u | wc -l)
            fi
            echo "level=${level}% r=${r} ${MODE} ${cond} exit=${RC} wall_s=$((END-START)) last_frame=${LAST:-0} complete=${COMPLETE} export_map=${EX_MAP:-NA} written=${EX_WRITTEN:-NA} other_map=${EX_OTHER:-NA} traj_lines=${TL} uniq_ts=${UN}"
        done
    done
done
echo "done=$(date -Is)"
