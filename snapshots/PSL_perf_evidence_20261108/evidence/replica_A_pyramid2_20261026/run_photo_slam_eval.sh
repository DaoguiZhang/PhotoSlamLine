#!/usr/bin/env bash
# Run Photo-SLAM-eval after the eight training runs; serial, no training.
set -u

ROOT=/workspace/code/PSL-perf
SITE=$ROOT/evidence/replica_A_pyramid2_20261026/eval_site
EVAL=$ROOT/evidence/replica_A_pyramid2_20261026/eval_project
RES=$ROOT/results/replica_A_pyramid2_20261026_A_pyramid2/replica_rgbd_0
GT=/workspace/code/SEGS-SLAM/datasets/replica
OUTLOG=$ROOT/evidence/replica_A_pyramid2_20261026
printf 'scene,exit_code,eval_txt,metrics_traj,log\n' > "$OUTLOG/eval_summary.csv"

for scene in office0 office1 office2 office3 office4 room0 room1 room2; do
    LOG=$OUTLOG/eval_${scene}.log
    (
        cd "$EVAL" || exit 2
        PYTHONPATH="$SITE:$EVAL" python3 run.py \
            "$RES/$scene" "$GT/$scene" > "$LOG" 2>&1
    )
    rc=$?
    eval_ok=0
    traj_ok=0
    [ -s "$RES/$scene/eval.txt" ] && eval_ok=1
    [ -s "$RES/$scene/metrics_traj.txt" ] && traj_ok=1
    printf '%s,%d,%d,%d,%s\n' "$scene" "$rc" "$eval_ok" "$traj_ok" "$LOG" \
        >> "$OUTLOG/eval_summary.csv"
    echo "[eval] scene=$scene rc=$rc eval_txt=$eval_ok metrics_traj=$traj_ok log=$LOG"
done
cat "$OUTLOG/eval_summary.csv"
