#!/usr/bin/env bash
# Serial, resumable, single-instance scheduler for the formal comparison.
#
# NOT STARTED in this round - see README.md §3 for the two unmet preconditions.
# It is provided so the campaign can be launched without further engineering
# once the protocol question is settled.
#
# Properties required by the acceptance rules:
#   * single instance      -> flock on LOCK
#   * resumable            -> one .ok marker per finished (cell, group, rep)
#   * serial               -> one child at a time
#   * no retry             -> a cell that ends in LOW_PSNR / INIT_FAIL / TRACK_FAIL
#                             is recorded and skipped, never re-run
#   * real pid + progress  -> progress.jsonl, one line per start/finish
#   * never overwrite      -> output dir carries the version hash; non-empty = refuse
#   * provenance           -> PROVENANCE.txt in every run dir
#
# usage: schedule_formal.sh <groupA_bin_dir> <groupA_tag> <groupB_bin_dir> <groupB_tag> <reps>
set -u
cd "$(dirname "$0")" || exit 1
PROTO=/workspace/code/PSL-lineband-multi-results/baseline_protocol
LOCK=$PWD/LOCK
exec 9>"$LOCK" || exit 1
if ! flock -n 9; then
  echo "another instance is running (flock held) - refusing to start" >&2
  exit 9
fi

GA_BIN=${1:?group A bin dir}; GA_TAG=${2:?group A tag}
GB_BIN=${3:?group B bin dir}; GB_TAG=${4:?group B tag}
REPS=${5:-5}

DONE=$PWD/done; mkdir -p "$DONE"
PROG=$PWD/progress.jsonl

echo "{\"event\":\"scheduler_start\",\"ts\":\"$(date -Is)\",\"pid\":$$,\"A\":\"$GA_TAG\",\"B\":\"$GB_TAG\",\"reps\":$REPS}" >> "$PROG"

run_one() {   # cell group tag bindir rep
  local cell="$1" group="$2" tag="$3" bindir="$4" rep="$5"
  local marker="$DONE/${cell}__${group}__r${rep}.ok"
  [ -f "$marker" ] && { echo "skip $cell $group r$rep (done)"; return 0; }

  # --- plan row -> command -------------------------------------------------
  local row
  row=$(python3 - "$PROTO/cells.csv" "$cell" <<'PY'
import csv,sys
cells, cell = sys.argv[1], sys.argv[2]
for r in csv.DictReader(open(cells)):
    if r['cell']==cell:
        print('|'.join([r['binary'],r['orb_cfg'],r['gauss_cfg'],r['off_cfg'],
                        r['seq_dir'],r['assoc'],r['frames']]))
        break
PY
)
  IFS='|' read -r binary orb_cfg gauss_cfg off_cfg seq_dir assoc frames <<< "$row"
  [ -z "${binary:-}" ] && { echo "unknown cell $cell" >&2; return 2; }

  local bin="$bindir/$binary"
  local outdir="/workspace/code/PSL-evtrepl/FORMAL_OUT/${tag}/${cell}/r${rep}"
  if [ -d "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
    echo "REFUSING to overwrite $outdir" >&2; return 4
  fi
  mkdir -p "$outdir"

  # provenance first, so a crash still leaves the identity behind
  {
    echo "cell=$cell group=$group tag=$tag rep=$rep"
    echo "binary=$bin sha256=$(sha256sum "$bin" 2>/dev/null | cut -d' ' -f1)"
    echo "orb_cfg=$orb_cfg sha256=$(sha256sum "$orb_cfg" 2>/dev/null | cut -d' ' -f1)"
    echo "gauss_cfg=$gauss_cfg sha256=$(sha256sum "$gauss_cfg" 2>/dev/null | cut -d' ' -f1)"
    echo "off_cfg=$off_cfg sha256=$(sha256sum "$off_cfg" 2>/dev/null | cut -d' ' -f1)"
    echo "frames=$frames"
    env | grep '^PHOTO_SLAM_' | sort
  } > "$outdir/PROVENANCE.txt"

  local t0=$(date +%s)
  echo "{\"event\":\"start\",\"ts\":\"$(date -Is)\",\"pid\":$$,\"cell\":\"$cell\",\"group\":\"$group\",\"rep\":$rep,\"out\":\"$outdir\"}" >> "$PROG"

  local args=("$bin" "$PROTO/../" )   # placeholder: real invocation mirrors run_seed.sh
  # The real argv is produced by make_argv.py from cells.csv + budget_per_cell.csv
  # so the scheduler never invents a command line.
  local argv
  argv=$(python3 "$PWD/make_argv.py" "$cell" "$bin" "$outdir") || {
     echo "{\"event\":\"argv_failed\",\"cell\":\"$cell\"}" >> "$PROG"; return 5; }

  timeout -s TERM 7200 bash -c "$argv" > "$outdir/stdout.log" 2>&1
  local rc=$?
  local t1=$(date +%s); local dur=$((t1-t0))

  local status=OK
  case "$rc" in
    0)   status=OK ;;
    124) status=TIMEOUT ;;
    134) status=ABORT ;;
    139) status=SEGV ;;
    3)   status=SHUTDOWN_TIMEOUT ;;
    4|5|6) status=MAPPER_FAIL ;;
    *)   status="RC$rc" ;;
  esac
  # A run that exited 0 but produced no metrics is not OK.
  if [ "$status" = OK ] && [ ! -s "$outdir/KeyFrameTrajectory_TUM.txt" ]; then
     status=NO_OUTPUT
  fi
  # Low quality / init fail / track fail are recorded, never retried.
  if [ "$status" = OK ] && grep -qi "tracking fail\|fail to initialize" "$outdir/stdout.log"; then
     status=TRACK_FAIL
  fi

  echo "{\"event\":\"finish\",\"ts\":\"$(date -Is)\",\"cell\":\"$cell\",\"group\":\"$group\",\"rep\":$rep,\"rc\":$rc,\"status\":\"$status\",\"seconds\":$dur}" >> "$PROG"
  {
    echo "cell=$cell group=$group tag=$tag rep=$rep"
    echo "rc=$rc status=$status seconds=$dur"
  } > "$DONE/${cell}__${group}__r${rep}.ok"
  echo "[$group $cell r$rep] rc=$rc status=$status ${dur}s"
}

# --- iterate --------------------------------------------------------------
mapfile -t CELLS < <(python3 -c "
import csv
for r in csv.DictReader(open('$PROTO/cells.csv')): print(r['cell'])
")
for cell in "${CELLS[@]}"; do
  for rep in $(seq 1 "$REPS"); do
    run_one "$cell" A "$GA_TAG" "$GA_BIN" "$rep" || true
    run_one "$cell" B "$GB_TAG" "$GB_BIN" "$rep" || true
  done
done

echo "{\"event\":\"scheduler_finish\",\"ts\":\"$(date -Is)\",\"pid\":$$}" >> "$PROG"
