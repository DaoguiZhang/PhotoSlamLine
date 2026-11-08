#!/usr/bin/env bash
# Common-input / event-stream replay: one process per (scene, seed).
#
# The process first runs the normal SLAM + Gaussian mapping pipeline with the
# exporter enabled (that is where the ONE common input per seed comes from),
# then replays the recorded geometry event stream five times from that same
# exported snapshot: A, A_repeat, V, N, VN.
#
# usage: run_seed.sh <scene> <seed> [steps]
#   scene  : office0_mono | desk3_rgbd
#   seed   : 0 | 1 | 2      (PHOTO_SLAM_SEED)
#   steps  : real optimizer.step() calls per condition (default 5000)
set -uo pipefail

SCENE="${1:?scene}"
SEED="${2:?seed}"
STEPS="${3:-5000}"

W=/workspace/code/PSL-replay
ROOT=/workspace/code/PSL-evtrepl
EXP=$ROOT/export/${SCENE}_s${SEED}
OUT=$ROOT/replay/${SCENE}_s${SEED}

case "$SCENE" in
  office0_mono)
    BIN=replica_mono
    ORB=$W/cfg/ORB_SLAM3/Monocular/Replica/office0.yaml
    GAU=$W/cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml
    SEQ=/workspace/code/SEGS-SLAM/datasets/replica/office0
    ASSOC="" ;;
  desk3_rgbd)
    BIN=eth3d_rgbd
    ORB=$W/cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml
    GAU=$W/cfg/gaussian_mapper/RGB-D/ETH3D/eth3d_desk_3.yaml
    SEQ=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3
    ASSOC=/workspace/code/Photo-SLAM/datasets/eth3d/desk_3/associated.txt ;;
  *) echo "unknown scene $SCENE" >&2; exit 2 ;;
esac

if [ -d "$OUT" ] && [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then
  echo "REFUSING to overwrite $OUT" >&2; exit 4
fi
mkdir -p "$OUT" "$EXP" "$ROOT/logs"

export LD_LIBRARY_PATH=$W/ORB-SLAM3/lib:$W/ORB-SLAM3/Thirdparty/g2o/lib:$W/ORB-SLAM3/Thirdparty/DBoW2/lib:$W/lib
export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_SLAM_ONLY=0 PHOTO_SLAM_NO_PACING=0 PHOTO_SLAM_SEED=$SEED
export PHOTO_SLAM_DEBUG_LINE_LOOP=1
# production defaults for the live leg; the replay sets its own values per
# condition through the in-binary override slot
export PHOTO_SLAM_DEDUP_VOXEL=1 PHOTO_SLAM_DEDUP_NOTIFY=1
export PHOTO_SLAM_EXPORT_DIR=$EXP
export PHOTO_SLAM_REPLAY_DIR=$EXP
export PHOTO_SLAM_REPLAY_STEPS=$STEPS
export PHOTO_SLAM_REPLAY_CONDS="A,A_repeat,V,N,VN"
export PHOTO_SLAM_REPLAY_ROTATE=$SEED

ARGS=("$W/ORB-SLAM3/Vocabulary/ORBvoc.txt" "$ORB" "$GAU" "$SEQ")
[ -n "$ASSOC" ] && ARGS+=("$ASSOC")
ARGS+=("$OUT/" "no_viewer")

LOG=$ROOT/logs/${SCENE}_s${SEED}.log
timeout 14400 "$W/bin/$BIN" "${ARGS[@]}" > "$LOG" 2>&1
rc=$?
echo "scene=$SCENE seed=$SEED steps=$STEPS exit=$rc out=$OUT log=$LOG"
exit $rc
