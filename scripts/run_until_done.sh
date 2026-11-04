#!/usr/bin/env bash
# Keep the serial scheduler fed until nothing is left to do.
#
# The scheduler itself is single-instance (flock) and resumable: it derives every
# cell's state from the artefacts on disk, so a restart never redoes a completed
# cell and never reuses a directory. This wrapper only decides whether another
# pass is justified:
#
#   * rescan, print the counts
#   * stop when PENDING == 0 (COMPLETED/FAILED/INVALID/TIMEOUT/PARTIAL/BLOCKED are
#     all terminal for scheduling purposes; FAILED and PARTIAL are retried once
#     inside the scheduler, into a sibling _r1 directory)
#   * stop after a pass that made no progress, so a hard blocker cannot spin
#   * otherwise run another pass
#
# It never edits results, never deletes a run directory and never declares the
# matrix finished: that is decided from the artefacts afterwards.
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=/workspace/eval_runs/final_runs
pass=0
while :; do
    pass=$((pass+1))
    before=$(grep -c 'exit=' "$ROOT/matrix_progress.log" 2>/dev/null || echo 0)
    echo "=== pass $pass start $(date -Iseconds) attempts_so_far=$before"
    python3 scripts/matrix.py run
    python3 scripts/matrix.py status | sed 's/^/[status] /'
    pending=$(python3 scripts/matrix.py status | awk '/^  PENDING/{print $2}')
    after=$(grep -c 'exit=' "$ROOT/matrix_progress.log" 2>/dev/null || echo 0)
    echo "=== pass $pass end $(date -Iseconds) pending=$pending new_attempts=$((after-before))"
    if [ "${pending:-0}" -eq 0 ]; then echo "no pending cells left"; break; fi
    if [ "$((after-before))" -le 0 ]; then
        echo "no progress in this pass and $pending cells still pending -- stopping, needs a human decision"
        break
    fi
done
