#!/usr/bin/env bash
# Independent continuation for the evaluation campaign.
#
#   internal matrix drains  ->  build the pinned upstream  ->  6 upstream rows
#   ->  statistics  ->  acceptance report
#
# This is a SEPARATE process from `scripts/matrix.py run`. It never edits that
# script, never dispatches a cell itself (it only observes), never deletes a run
# directory and never overwrites one. Every stage writes its own log and is
# skipped if its output already exists, so re-running the wrapper resumes instead
# of repeating work.
#
# Ordering rule that this file exists to honour: the upstream build runs only
# after the internal matrix has drained. Compiling with -j on the same cores
# during timed runs would inflate the tracked wall-clock and the [EVAL-TIMING]
# buckets, which are themselves results.
#
# Usage:  ROOT=/workspace/eval_runs/<frozen root> nohup bash scripts/continue_after_matrix.sh &
set -uo pipefail
TREE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ROOT:?ROOT must point at the frozen results root of this stage}"
UP=/workspace/upstream/Photo-SLAM
UP_COMMIT=f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac
LOG="$ROOT/CONTINUATION.log"

say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

say "continuation start, root=$ROOT"

# ---------------------------------------------------------------- stage 1 ----
# Wait for the internal matrix to drain. A stopped scheduler that still has
# PENDING work is NOT "drained": if no scheduler is alive and cells remain
# pending, stop and let a human decide.
while :; do
    st=$(cd "$TREE" && python3 scripts/matrix.py status 2>/dev/null)
    pending=$(printf '%s\n' "$st" | awk '/^  PENDING/{print $2}')
    partial=$(printf '%s\n' "$st" | awk '/^  PARTIAL/{print $2}')
    running=$(pgrep -f 'scripts/matrix.py run' | head -1)
    say "internal: pending=${pending:-?} partial=${partial:-?} scheduler_pid=${running:-none}"
    if [ "${pending:-1}" = "0" ]; then
        say "internal matrix drained"
        break
    fi
    if [ -z "${running:-}" ]; then
        say "STOPPED_INCOMPLETE: no scheduler is running but ${pending} cells are pending; not proceeding"
        exit 3
    fi
    sleep 120
done

# ---------------------------------------------------------------- stage 2 ----
BUILDLOG="$ROOT/UPSTREAM_BUILD.log"
if [ ! -f "$ROOT/UPSTREAM_BUILD.ok" ]; then
    say "stage 2: building upstream at $UP_COMMIT"
    cd "$UP" || { say "upstream clone missing"; exit 4; }
    head_sha=$(git rev-parse HEAD)
    if [ "$head_sha" != "$UP_COMMIT" ]; then
        say "ABORT: upstream HEAD is $head_sha, expected $UP_COMMIT"
        exit 4
    fi
    # Recorded build adaptation: the rasteriser arch list predates this GPU.
    if ! grep -q '"75;86;120"' CMakeLists.txt; then
        cp CMakeLists.txt "$ROOT/UPSTREAM_CMakeLists.orig"
        sed -i 's/CUDA_ARCHITECTURES "75;86"/CUDA_ARCHITECTURES "75;86;120"/' CMakeLists.txt
        diff -u "$ROOT/UPSTREAM_CMakeLists.orig" CMakeLists.txt \
            > "$ROOT/UPSTREAM_arch_adaptation.diff"
        say "adapted CUDA_ARCHITECTURES (diff saved)"
    fi
    ( bash build.sh >> "$BUILDLOG" 2>&1 )
    rc=$?
    say "upstream build finished rc=$rc"
    if [ $rc -eq 0 ] && [ -x "$UP/build/photo-slam" ]; then
        sha256sum "$UP/build/photo-slam" > "$ROOT/UPSTREAM_BUILD.ok"
        say "upstream binary: $(cat "$ROOT/UPSTREAM_BUILD.ok")"
    else
        say "upstream build failed; 6 control rows stay BLOCKED (see $BUILDLOG)"
        exit 5
    fi
else
    say "stage 2: upstream binary already built: $(cat "$ROOT/UPSTREAM_BUILD.ok")"
fi

# ---------------------------------------------------------------- stage 3 ----
# The 6 control rows: cabinet and large_cabinet, monocular, 3 processes each.
VOCAB="$UP/ORB-SLAM3/Vocabulary/ORBvoc.txt"
DS=/workspace/code/SEGS-SLAM/datasets/tum
for scene in cabinet large_cabinet; do
    seq="$DS/rgbd_dataset_freiburg3_$scene"
    cfg="$UP/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_$scene.yaml"
    gcfg=$(ls "$UP"/cfg/gaussian_mapper/TUM/*"$scene"*.yaml 2>/dev/null | head -1)
    for rep in 1 2 3; do
        out="$ROOT/UPSTREAM/$scene/mono/UP/rep$rep"
        if [ -d "$out" ]; then
            say "stage 3: $scene rep$rep exists, not overwritten"
            continue
        fi
        mkdir -p "$out"
        say "stage 3: $scene rep$rep -> $out"
        {
            echo "id=UPSTREAM|TUM|$scene|mono|rep$rep"
            echo "upstream_commit=$UP_COMMIT"
            echo "upstream_bin=$(cat "$ROOT/UPSTREAM_BUILD.ok")"
            echo "cmd=$UP/build/photo-slam $VOCAB $cfg $gcfg $seq $out no_viewer"
            echo "start=$(date -Iseconds)"
        } > "$out/meta.txt"
        t0=$(date +%s)
        ( cd "$UP" && ./build/photo-slam "$VOCAB" "$cfg" "$gcfg" "$seq" "$out" \
            no_viewer > "$out/stdout.log" 2>&1 )
        rc=$?
        t1=$(date +%s)
        {
            echo "exit_code=$rc"
            echo "wall_s=$((t1-t0))"
            echo "end=$(date -Iseconds)"
            echo "note="
        } > "$out/status.txt"
        say "stage 3: $scene rep$rep exit=$rc wall=$((t1-t0))s"
    done
done

# ---------------------------------------------------------------- stage 4 ----
say "stage 4: statistics"
( cd "$TREE" && python3 eval/aggregate.py --root "$ROOT" >> "$LOG" 2>&1 )
say "aggregate rc=$?"

# ---------------------------------------------------------------- stage 5 ----
say "stage 5: acceptance"
( cd "$TREE" && python3 eval/acceptance.py --root "$ROOT" >> "$LOG" 2>&1 )
say "acceptance rc=$?"

say "continuation finished"
