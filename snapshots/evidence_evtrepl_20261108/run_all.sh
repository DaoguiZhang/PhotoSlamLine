#!/usr/bin/env bash
# Run all six (scene, seed) pairs sequentially. Each process produces ONE
# exported common input and then replays five conditions on it.
set -u
cd /workspace/code/PSL-evtrepl || exit 1
for SCENE in office0_mono desk3_rgbd; do
  for SEED in 0 1 2; do
    echo "=== $(date -Is) START $SCENE seed=$SEED"
    ./run_seed.sh "$SCENE" "$SEED" 5000 2>&1 | tail -2
    echo "=== $(date -Is) DONE  $SCENE seed=$SEED rc=$?"
    df -h /workspace | tail -1
  done
done
echo "=== $(date -Is) ALL DONE"
