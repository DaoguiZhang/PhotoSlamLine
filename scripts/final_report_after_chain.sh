#!/usr/bin/env bash
# Wait for the finalisation chain to finish, then write the consolidated report.
#
# Runs as its own process, started once. It never touches the supplement
# scheduler, the binaries, the configs or the manifests; it only reads the
# chain's log and, at the very end, the products the chain wrote.
#
# Output: /workspace/eval_runs/final_v4/FINAL_REPORT.md
set -uo pipefail
TREE="$(cd "$(dirname "$0")/.." && pwd)"
FINAL=/workspace/eval_runs/final_v4
LOG="$FINAL/FINALIZE.log"
OUTLOG="$FINAL/FINAL_REPORT.log"
say() { echo "[$(date -Iseconds)] $*" >> "$OUTLOG"; }

say "report watcher start"
while :; do
    if [ ! -f "$LOG" ]; then
        sleep 120; continue
    fi
    if grep -q "finalize finished" "$LOG"; then
        say "chain finished; building the consolidated report"
        break
    fi
    if grep -q "BLOCKED" "$LOG"; then
        say "chain reported BLOCKED; not building a report over a failed chain"
        grep "BLOCKED" "$LOG" | tail -1 >> "$OUTLOG"
        exit 3
    fi
    sleep 120
done

cd "$TREE" || exit 1
nice -n 19 python3 eval/final_report.py --root "$FINAL/root" \
    --out "$FINAL/FINAL_REPORT.md" >> "$OUTLOG" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
    say "final_report.py failed rc=$rc"
    exit 4
fi
say "wrote $FINAL/FINAL_REPORT.md"
