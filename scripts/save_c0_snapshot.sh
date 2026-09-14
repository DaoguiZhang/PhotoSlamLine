#!/usr/bin/env bash
# =============================================================================
# Save the reproducible source/binary snapshot for the C0 wrap-up.
# - full git diff of modified tracked files
# - copies of NEW files (LineMode.h, new scripts)
# - binary checksums (libORB_SLAM3.so, bin/replica_mono)
# - config files used (SL + Gaussian)
# Writes to results/diagnosis/_version/c0_wrapup/ (never deletes anything).
# =============================================================================
set -euo pipefail
ROOT=/workspace/code/Photo-SLAM-L
DEST=$ROOT/results/diagnosis/_version/c0_wrapup
mkdir -p "$DEST"

{
    echo "# C0 wrap-up snapshot $(date -Is)"
    echo "# branch: $(git -C "$ROOT" branch --show-current)"
    echo "# HEAD:   $(git -C "$ROOT" rev-parse HEAD)"
    echo
    echo "## Binary checksums"
    echo "libORB_SLAM3.so = $(md5sum "$ROOT/ORB-SLAM3/lib/libORB_SLAM3.so" 2>/dev/null | awk '{print $1}')"
    echo "bin/replica_mono = $(md5sum "$ROOT/bin/replica_mono" 2>/dev/null | awk '{print $1}')"
    echo
    echo "## Snapshots dir"
    for d in c0_fix c0_prefix; do
        f="$ROOT/results/diagnosis/_snapshots/$d/libORB_SLAM3.so"
        [ -f "$f" ] && echo "$d = $(md5sum "$f" | awk '{print $1}')"
    done
} > "$DEST/README.txt"

git -C "$ROOT" diff > "$DEST/working_tree.diff"
git -C "$ROOT" diff ORB-SLAM3/src/Optimizer.cc > "$DEST/Optimizer.cc.diff"
git -C "$ROOT" diff ORB-SLAM3/src/System.cc    > "$DEST/System.cc.diff"
git -C "$ROOT" diff ORB-SLAM3/src/Tracking.cc  > "$DEST/Tracking.cc.diff"

for f in ORB-SLAM3/include/LineMode.h \
         scripts/run_c0_verify.sh \
         scripts/run_cross_seq_regression.sh \
         scripts/eval_ate_rpe.py \
         scripts/eval_ate_sim3.py \
         scripts/eval_crosscheck.py; do
    [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$DEST/$(basename "$f")"
done

for f in cfg/ORB_SLAM3/Monocular/Replica/office0.yaml \
         cfg/ORB_SLAM3/Monocular/Replica/office2.yaml \
         cfg/ORB_SLAM3/Monocular/Replica/room0.yaml \
         cfg/gaussian_mapper/Monocular/Replica/replica_mono.yaml \
         cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml; do
    [ -f "$ROOT/$f" ] && cp "$ROOT/$f" "$DEST/$(basename "$f")"
done

echo "snapshot saved to $DEST"
