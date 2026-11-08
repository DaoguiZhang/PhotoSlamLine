#!/usr/bin/env bash
# Independent MONOCULAR experiment for desk_3 at the SAME entry point as the
# RGB-D A'/B/C comparison:
#   A' = PHOTO_SLAM_LINE_MODE=1, PHOTO_SLAM_LINE_RESCUE=0  (line pipeline, no
#        line edges in the optimizers)
#   B  = PHOTO_SLAM_LINE_MODE=2, PHOTO_SLAM_LINE_RESCUE=0  (full line)
#   C  = PHOTO_SLAM_LINE_MODE=2, PHOTO_SLAM_LINE_RESCUE=1  (full line + rescue)
# Stress levels are fixed in advance at 30% and 50% point-drop, 5 repeats,
# interleaved (level -> condition -> repeat) so that drift in the machine state
# cannot be confounded with the condition.
#
# Tracking lists and rescue thresholds are NOT modified here. Nothing in this
# script changes what is recorded; it only records whatever the binaries do.
#
# usage: bash scripts/mono_3050.sh [outdir] [reps]
set -uo pipefail

ROOT=/workspace/code/Photo-SLAM-L
BASE="${1:-/tmp/lr_mono_3050}"
REPS="${2:-5}"

export RUN_TIMEOUT=600
export PHOTO_SLAM_DEBUG_MERGE=1
export PHOTO_SLAM_SLAM_ONLY=1
export PHOTO_SLAM_DEBUG_LINE_GATE=1
export PHOTO_SLAM_DEBUG_LINE_RESCUE=1
export PHOTO_SLAM_DEBUG_LINE_SCAN=1
export PHOTO_SLAM_DEBUG_LINE_EDGES=1
export PHOTO_SLAM_STRESS_START_FRAME=30
export LD_LIBRARY_PATH="$ROOT/ORB-SLAM3/lib:$ROOT/ORB-SLAM3/Thirdparty/g2o/lib:$ROOT/ORB-SLAM3/Thirdparty/DBoW2/lib:$ROOT/lib"

SEQ=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3
VOC=$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt
ORBCFG=$ROOT/cfg/ORB_SLAM3/Monocular/ETH3D/eth3d_desk_3.yaml
GAUSS=$ROOT/cfg/gaussian_mapper/Monocular/ETH3D/eth3d_desk_3.yaml

# Fail fast if the entry point is not the one we intend to measure.
for f in "$VOC" "$ORBCFG" "$GAUSS" "$SEQ/rgb.txt" "$ROOT/bin/eth3d_mono"; do
    [ -e "$f" ] || { echo "FATAL: missing $f"; exit 2; }
done

mkdir -p "$BASE"
echo "start=$(date -Is) base=$BASE reps=$REPS scene=desk_3 mode=mono"

for r in $(seq 1 "$REPS"); do
    for level in 30 50; do
        export PHOTO_SLAM_STRESS_DROP_POINT_RATIO="0.${level}"
        for cond in A B C; do
            case "$cond" in
                A) export PHOTO_SLAM_LINE_MODE=1; export PHOTO_SLAM_LINE_RESCUE=0 ;;
                B) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=0 ;;
                C) export PHOTO_SLAM_LINE_MODE=2; export PHOTO_SLAM_LINE_RESCUE=1 ;;
            esac

            d="$BASE/drop${level}/r${r}_desk_3_mono_${cond}"
            mkdir -p "$d"
            {
                echo "scene=desk_3"
                echo "mode=mono"
                echo "cond=$cond"
                echo "run_idx=$r"
                echo "level=$level"
                echo "line_mode=$PHOTO_SLAM_LINE_MODE"
                echo "rescue=$PHOTO_SLAM_LINE_RESCUE"
                echo "sequence=$SEQ"
                echo "orb_cfg=$ORBCFG"
                echo "gaussian_cfg=$GAUSS"
                echo "voc=$VOC"
                echo "binary=$ROOT/bin/eth3d_mono"
                echo "stress_drop_point_ratio=$PHOTO_SLAM_STRESS_DROP_POINT_RATIO"
                echo "stress_start_frame=$PHOTO_SLAM_STRESS_START_FRAME"
            } > "$d/run.meta"

            START=$(date +%s)
            timeout --signal=TERM --kill-after=60 "${RUN_TIMEOUT}" \
                "$ROOT/bin/eth3d_mono" "$VOC" "$ORBCFG" "$GAUSS" "$SEQ" "$d" no_viewer \
                > "$d/log.txt" 2>&1
            RC=$?
            END=$(date +%s)

            LAST=$(grep -oE 'frame=[0-9]+' "$d/log.txt" 2>/dev/null | sed 's/frame=//' | sort -n | tail -1)
            SAVED=$(grep -c "Saving keyframe trajectory" "$d/log.txt" 2>/dev/null)
            SAVED=${SAVED:-0}
            COMPLETE=0
            [ "$RC" = "0" ] && [ "$SAVED" -ge 1 ] && COMPLETE=1

            {
                echo "exit=$RC"
                echo "wall_s=$((END-START))"
                echo "last_frame=${LAST:-0}"
                echo "saved_keyframe_trajectory=$SAVED"
                echo "complete=$COMPLETE"
            } >> "$d/run.meta"

            if [ -s "$d/CameraTrajectory_TUM.txt" ]; then
                TL=$(wc -l < "$d/CameraTrajectory_TUM.txt")
                UN=$(awk '{print $1}' "$d/CameraTrajectory_TUM.txt" | sort -u | wc -l)
            else
                TL=0; UN=0
            fi
            echo "drop=${level}% r=${r} cond=${cond} exit=${RC} wall_s=$((END-START)) last_frame=${LAST:-0} complete=${COMPLETE} traj_lines=${TL} uniq_ts=${UN}"
        done
    done
done
echo "done=$(date -Is)"
