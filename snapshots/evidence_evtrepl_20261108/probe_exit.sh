#!/usr/bin/env bash
# Host-exit probe.
#
# The container forbids attaching to a running process (yama ptrace_scope=1 and
# /proc/sys is read-only), so the stacks must be taken by *starting* the program
# under gdb. gdb is allowed to trace its own child. To stop the program at the
# moment it should have exited, a watchdog sends SIGALRM after LIMIT seconds;
# gdb catches it, stops the inferior, and batch mode then runs
# "thread apply all bt" for every thread.
#
# usage: probe_exit.sh <scene> <steps> <limit_s> <attempts> [tag]
set -u
SCENE="${1:?scene}"
STEPS="${2:-50}"
LIMIT="${3:-240}"
ATTEMPTS="${4:-6}"
TAG="${5:-probe}"

W=/workspace/code/PSL-replay
ROOT=/workspace/code/PSL-evtrepl
EXP=$ROOT/export/${SCENE}
OUT=$ROOT/exitprobe
mkdir -p "$OUT"

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
  *) echo "unknown scene" >&2; exit 2 ;;
esac

export LD_LIBRARY_PATH=$W/ORB-SLAM3/lib:$W/ORB-SLAM3/Thirdparty/g2o/lib:$W/ORB-SLAM3/Thirdparty/DBoW2/lib:$W/lib
export PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 PHOTO_SLAM_SHADOW_LINE=0
export PHOTO_SLAM_SLAM_ONLY=0 PHOTO_SLAM_NO_PACING=0
export PHOTO_SLAM_DEDUP_VOXEL=1 PHOTO_SLAM_DEDUP_NOTIFY=1
export PHOTO_SLAM_REPLAY_DIR=$EXP
export PHOTO_SLAM_REPLAY_STEPS=$STEPS
export PHOTO_SLAM_REPLAY_CONDS="A,V,N,VN"

ARGS=("$W/ORB-SLAM3/Vocabulary/ORBvoc.txt" "$ORB" "$GAU" "$SEQ")
[ -n "$ASSOC" ] && ARGS+=("$ASSOC")
ARGS+=("$OUT/out_$TAG/" "no_viewer")

for i in $(seq 1 "$ATTEMPTS"); do
  LOG=$OUT/${TAG}_att${i}.log
  echo "=== attempt $i $(date -Is) log=$LOG"
  gdb -batch -ex "set pagination off" -ex "handle SIGALRM stop print nopass" \
      -ex run -ex "thread apply all bt" --args "$W/bin/$BIN" "${ARGS[@]}" \
      > "$LOG" 2>&1 &
  GPID=$!
  sleep "$LIMIT"
  IPID=$(pgrep -f "$W/bin/$BIN" | head -1)
  if [ -n "${IPID:-}" ]; then
    echo "    watchdog: still alive after ${LIMIT}s (pid $IPID) -> SIGALRM"
    kill -ALRM "$IPID" 2>/dev/null
    HUNG=1
  else
    HUNG=0
  fi
  wait "$GPID"
  if [ "$HUNG" = 1 ]; then
    echo "    HUNG attempt $i captured -> $LOG"
    echo "$LOG" > "$OUT/HUNG_POINTER"
    exit 0
  fi
  tail -3 "$LOG" | sed 's/^/    /'
done
echo "no hang in $ATTEMPTS attempts"
