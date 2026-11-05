#!/usr/bin/env bash
# Finalisation chain for the supplement campaign.
#
# Waits for the supplementary queue to drain, then performs the offline re-eval
# and the per-ID settlement. It is a separate process from the supplement
# scheduler and never touches it; it only observes.
#
# Stages (each checks its own exit code and stops with a specific blocker):
#   1. wait: supplement PENDING == 0, no eval_frontend running, lock released
#   2. offline re-evaluation with ONE protocol:
#        - supplement_v4 (the 339 supplementary cells)
#        - final_runs_v2 (the 147 reusable T0 cells + the 6 upstream rows)
#   3. merge the two records into one per-ID view (supplement attempt wins for a
#      given id; otherwise the v_final_2 attempt is used), keeping BOTH
#   4. settlement over the original 492 plan ids
#   5. statistics: per scene x arm with every repeat, median, range, valid
#      repeat count; ONLINE and FINAL separately
#   6. acceptance
#
# Usage: ROOT=/workspace/eval_runs/supplement_v4 nohup bash scripts/finalize_after_supplement.sh &
set -uo pipefail
TREE="$(cd "$(dirname "$0")/.." && pwd)"
SUPP="${ROOT:-/workspace/eval_runs/supplement_v4}"
BASE=/workspace/eval_runs/final_runs_v2
FINAL=/workspace/eval_runs/final_v4
LOG="$FINAL/FINALIZE.log"
mkdir -p "$FINAL"
say() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }
die() { say "BLOCKED: $*"; exit "$1"; }

say "finalize start, supplement=$SUPP final=$FINAL"
strays=0

# ---------------------------------------------------------------- stage 1 ----
while :; do
    st=$(cd "$TREE" && python3 scripts/supplement.py status 2>/dev/null) || {
        say "supplement status failed; retrying"; sleep 120; continue; }
    pending=$(printf '%s\n' "$st" | awk '/^  PENDING/{print $2}'); pending=${pending:-0}
    slam=$(pgrep -f 'entry_rel/eval_frontend' | head -1)
    sched=$(pgrep -f 'scripts/supplement.py run' | head -1)
    lockbusy=0
    if [ -f "$SUPP/.supplement.lock" ]; then
        python3 - "$SUPP/.supplement.lock" <<'PY' || lockbusy=1
import fcntl, sys
f = open(sys.argv[1], "a")
try:
    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(1)
PY
    fi
    say "supplement: pending=$pending scheduler=${sched:-none} slam=${slam:-none} lock_busy=$lockbusy"
    if [ "$pending" = "0" ] && [ -z "$slam" ] && [ "$lockbusy" = "0" ]; then
        say "stage 1 done: queue drained, no running SLAM, lock released"
        break
    fi
    if [ -z "$sched" ] && [ "$pending" != "0" ]; then
        strays=$((strays + 1))
        say "no scheduler but pending=$pending (observation $strays/3)"
        if [ $strays -ge 3 ]; then
            die 3 "no supplement scheduler is running but $pending cells are pending (STOPPED_INCOMPLETE)"
        fi
    else
        strays=0
    fi
    sleep 120
done

# ---------------------------------------------------------------- stage 2 ----
say "stage 2: offline re-evaluation (one protocol, both records)"
mkdir -p "$FINAL/supp_metrics" "$FINAL/base_metrics"
( cd "$TREE" && python3 eval/online_eval.py --root "$SUPP" --out "$FINAL/supp_metrics" ) \
    >> "$LOG" 2>&1 || die 4 "online_eval on the supplement failed"
( cd "$TREE" && python3 eval/online_eval.py --root "$BASE" --out "$FINAL/base_metrics" ) \
    >> "$LOG" 2>&1 || die 4 "online_eval on the v_final_2 record failed"
( cd "$TREE" && python3 eval/upstream_eval.py --root "$BASE" ) >> "$LOG" 2>&1 \
    || die 4 "upstream_eval failed"
say "stage 2 done"

# ---------------------------------------------------------------- stage 3 ----
say "stage 3: one record per plan id (symlinks, nothing copied)"
( cd "$TREE" && python3 eval/build_merged_root.py --base "$BASE" --supp "$SUPP" \
    --out "$FINAL/root" --metrics-supp "$FINAL/supp_metrics" \
    --metrics-base "$FINAL/base_metrics" ) | tee -a "$LOG"
[ "${PIPESTATUS[0]}" = "0" ] || die 5 "build_merged_root failed"

# ---------------------------------------------------------------- stage 4 ----
say "stage 4: settlement over the 492 plan ids"
( cd "$TREE" && python3 eval/settle.py --root "$FINAL/root" \
    --out "$FINAL/SETTLEMENT_492.csv" ) >> "$LOG" 2>&1 || die 6 "settle failed"

# ---------------------------------------------------------------- stage 5 ----
say "stage 5: statistics (ONLINE and FINAL kept apart)"
( cd "$TREE" && python3 eval/aggregate.py --root "$FINAL/root" --out "$FINAL" ) >> "$LOG" 2>&1 \
    || die 7 "aggregate failed"

# ---------------------------------------------------------------- stage 6 ----
say "stage 6: acceptance"
( cd "$TREE" && python3 eval/acceptance.py --root "$SUPP" ) >> "$LOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    say "acceptance reported failures (rc=$rc); see $SUPP/ACCEPTANCE_ARTEFACTS.csv"
    die 8 "acceptance did not pass; not declaring completion"
fi
say "finalize finished"
