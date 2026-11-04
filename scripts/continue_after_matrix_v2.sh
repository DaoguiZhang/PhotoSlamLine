#!/usr/bin/env bash
# Continuation chain, v2. Replaces continue_after_matrix.sh, which had three
# defects: it hardcoded a non-existent `build/photo-slam`, it checked only
# PENDING (so a paused scheduler could be mistaken for a drained one), and it
# printed "finished" regardless of the return codes of the stages.
#
# Stages (each one checks its own exit code and stops the chain with a specific
# blocker message on failure):
#   1. wait until the internal matrix is genuinely done: PENDING == 0, no
#      eval_frontend process running, and the scheduler's flock released
#   2. build upstream at the pinned commit (arch adaptation recorded)
#   3. run the 6 control rows with the OFFICIAL entry point and arguments
#   4. statistics (coverage + online/final + aggregate + map-history check)
#   5. acceptance report
#
# Directory reuse rule: a cell counts as done only if status.txt exists, its
# exit_code is 0, and the artefacts it promises exist. Anything else is kept and
# classified; nothing is overwritten.
set -uo pipefail
TREE="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ROOT:?ROOT must point at the frozen results root}"
UP=/workspace/upstream/Photo-SLAM
UP_COMMIT=f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac
LOG="$ROOT/CONTINUATION.log"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { say "BLOCKED: $*"; exit "$1"; }

# The upstream binary and configs as the repository itself builds/uses them.
UP_BIN="$UP/bin/tum_mono"                                   # CMAKE_RUNTIME_OUTPUT_DIRECTORY
UP_GAUSS="$UP/cfg/gaussian_mapper/Monocular/TUM/tum_mono.yaml"
UP_VOCAB="$UP/ORB-SLAM3/Vocabulary/ORBvoc.txt"

say "continuation v2 start, root=$ROOT"

# ---------------------------------------------------------------- stage 1 ----
while :; do
    st=$(cd "$TREE" && python3 scripts/matrix.py status 2>/dev/null)
    pending=$(printf '%s\n' "$st" | awk '/^  PENDING/{print $2}')
    sched=$(pgrep -f 'scripts/matrix.py run' | head -1)
    slam=$(pgrep -f 'entry_rel/eval_frontend' | head -1)
    # lock probe: a held lock means a scheduler still owns this run root
    lockbusy=0
    if [ -f "$ROOT/.matrix.lock" ]; then
        python3 - "$ROOT/.matrix.lock" <<'PY' || lockbusy=1
import fcntl, sys
f = open(sys.argv[1], "a")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(1)
PY
    fi
    say "internal: pending=${pending:-?} scheduler=${sched:-none} slam=${slam:-none} lock_busy=$lockbusy"
    if [ "${pending:-1}" = "0" ] && [ -z "$slam" ] && [ "$lockbusy" = "0" ]; then
        say "stage 1 done: no pending cells, no running SLAM, lock released"
        break
    fi
    if [ -z "$sched" ] && [ "${pending:-1}" != "0" ]; then
        die 3 "no scheduler is running but ${pending} cells are pending (STOPPED_INCOMPLETE)"
    fi
    sleep 120
done

# ---------------------------------------------------------------- stage 2 ----
if [ ! -f "$ROOT/UPSTREAM_BUILD.ok" ]; then
    say "stage 2: building upstream at $UP_COMMIT"
    [ -d "$UP/.git" ] || die 4 "upstream clone missing at $UP"
    head_sha=$(cd "$UP" && git rev-parse HEAD)
    [ "$head_sha" = "$UP_COMMIT" ] || die 4 "upstream HEAD=$head_sha, expected $UP_COMMIT"
    if ! grep -q '"75;86;120"' "$UP/CMakeLists.txt"; then
        cp "$UP/CMakeLists.txt" "$ROOT/UPSTREAM_CMakeLists.orig"
        sed -i 's/CUDA_ARCHITECTURES "75;86"/CUDA_ARCHITECTURES "75;86;120"/' \
            "$UP/CMakeLists.txt"
        diff -u "$ROOT/UPSTREAM_CMakeLists.orig" "$UP/CMakeLists.txt" \
            > "$ROOT/UPSTREAM_arch_adaptation.diff" || true
        say "adapted CUDA_ARCHITECTURES to include sm_120 (diff saved)"
    fi
    ( cd "$UP" && bash build.sh ) >> "$ROOT/UPSTREAM_BUILD.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ] || [ ! -x "$UP_BIN" ]; then
        say "stage 2 FAILED rc=$rc; binary $UP_BIN present=$([ -x "$UP_BIN" ] && echo yes || echo no)"
        die 5 "upstream build failed; see $ROOT/UPSTREAM_BUILD.log (6 control rows stay BLOCKED)"
    fi
    {
        echo "commit=$UP_COMMIT"
        echo "binary=$UP_BIN"
        sha256sum "$UP_BIN"
        echo "vocab_sha256=$(sha256sum "$UP_VOCAB" | cut -d' ' -f1)"
        echo "gauss_cfg_sha256=$(sha256sum "$UP_GAUSS" | cut -d' ' -f1)"
        echo "arch_adaptation=$( [ -f "$ROOT/UPSTREAM_arch_adaptation.diff" ] && echo yes || echo no)"
    } > "$ROOT/UPSTREAM_BUILD.ok"
    say "stage 2 done: $(grep -m1 sha256 "$ROOT/UPSTREAM_BUILD.ok")"
else
    say "stage 2 skipped: $(grep -m1 'binary=' "$ROOT/UPSTREAM_BUILD.ok")"
fi

# ---------------------------------------------------------------- stage 3 ----
# The 6 control rows: cabinet and large_cabinet, monocular, 3 processes each.
# Upstream ships ORB configs only for fr1_desk / fr2_xyz / fr3_long_office, so
# these two scenes use a config derived from the fork's official TUM config of
# the same sequence (same calibration constants), recorded as an adaptation.
DS=/workspace/code/SEGS-SLAM/datasets/tum
CFGDIR="$ROOT/UPSTREAM_cfg"
mkdir -p "$CFGDIR"
for scene in cabinet large_cabinet; do
    seq="$DS/rgbd_dataset_freiburg3_$scene"
    [ -d "$seq" ] || die 6 "input missing: $seq"
    [ -f "$seq/rgb.txt" ] || die 6 "upstream tum_mono needs $seq/rgb.txt"
    src="$TREE/cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_$scene.yaml"
    [ -f "$src" ] || die 6 "no source config to derive from: $src"
    dst="$CFGDIR/tum_freiburg3_$scene.yaml"
    if [ ! -f "$dst" ]; then
        cp "$src" "$dst"
        { echo "# Derived from $src (identical content) because upstream"
          echo "# Photo-SLAM at $UP_COMMIT ships no ORB config for this scene."
          echo "# Derivation: byte-identical copy; source sha256:"
          sha256sum "$src"; } > "$CFGDIR/DERIVATION_$scene.txt"
    fi
    for rep in 1 2 3; do
        out="$ROOT/UPSTREAM/$scene/mono/UP/rep$rep"
        if [ -f "$out/status.txt" ] && grep -q '^exit_code=0$' "$out/status.txt" \
           && [ -f "$out/CameraTrajectory_TUM.txt" ]; then
            say "stage 3: $scene rep$rep already complete, not overwritten"
            continue
        fi
        [ -e "$out" ] && say "stage 3: $scene rep$rep exists but is incomplete; keeping it and running into rep${rep}_r1"
        [ -e "$out" ] && out="${out}_r1"
        mkdir -p "$out"
        {
            echo "id=UPSTREAM|TUM|$scene|mono|rep$rep"
            echo "upstream_commit=$UP_COMMIT"
            echo "orb_cfg=$dst"
            echo "gauss_cfg=$UP_GAUSS"
            echo "start=$(date -Iseconds)"
        } > "$out/meta.txt"
        t0=$(date +%s)
        ( cd "$UP" && "$UP_BIN" "$UP_VOCAB" "$dst" "$UP_GAUSS" "$seq" "$out" \
            no_viewer ) > "$out/stdout.log" 2>&1
        rc=$?
        t1=$(date +%s)
        {
            echo "exit_code=$rc"
            echo "wall_s=$((t1-t0))"
            echo "end=$(date -Iseconds)"
            echo "note="
        } > "$out/status.txt"
        say "stage 3: $scene rep$rep exit=$rc wall=$((t1-t0))s dir=$out"
        if [ $rc -ne 0 ]; then say "stage 3: non-zero exit kept as a result (no retry, no deletion)"; fi
    done
done

# ---------------------------------------------------------------- stage 4 ----
say "stage 4: statistics"
( cd "$TREE" && python3 scripts/matrix.py ledger ) >> "$LOG" 2>&1 || die 7 "ledger failed"
( cd "$TREE" && python3 eval/coverage.py --root "$ROOT" --out "$ROOT/metrics" ) >> "$LOG" 2>&1 \
    || die 7 "coverage.py failed"
( cd "$TREE" && python3 eval/online_eval.py --root "$ROOT" --out "$ROOT/metrics" ) >> "$LOG" 2>&1 \
    || die 7 "online_eval.py failed"
( cd "$TREE" && python3 eval/map_history_check.py --root "$ROOT" \
    --out "$ROOT/metrics/MAP_HISTORY.json" ) >> "$LOG" 2>&1 || die 7 "map_history_check failed"
( cd "$TREE" && python3 eval/aggregate.py --root "$ROOT" ) >> "$LOG" 2>&1 \
    || die 7 "aggregate.py failed"
say "stage 4 done"

# ---------------------------------------------------------------- stage 5 ----
( cd "$TREE" && python3 eval/acceptance.py --root "$ROOT" ) >> "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    say "stage 5: acceptance reported failures (rc=$rc); see $ROOT/ACCEPTANCE_ARTEFACTS.csv"
    die 8 "acceptance did not pass; not declaring completion"
fi
say "stage 5 done: acceptance passed"
say "continuation v2 finished"
