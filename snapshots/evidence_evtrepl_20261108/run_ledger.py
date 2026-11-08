#!/usr/bin/env python3
"""Per-host-process ledger for the 30 mapping-condition runs.

Three different questions must be answered separately and must not be merged:

  1. did the 30 mapping conditions run?          -> `conditions_complete`
  2. is every product of each condition there?   -> `products_complete`
  3. did the host process that carried them exit normally? -> `host_exit`

A run whose products are complete but whose host process had to be killed is
recorded as such: `host_exit = KILLED_SIGKILL`, never as 0.

Sources of truth (nothing is retyped from memory except the operator action on
desk3 seed 1, which is quoted verbatim):
  logs/batch.log                     start/end wall clock per (scene, seed)
  replay/<scene>_s<seed>/<cond>/*    the products themselves
  logs/<scene>_s<seed>.log           the run's own stdout/stderr
"""
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPLAY = os.path.join(ROOT, "replay")
LOGS = os.path.join(ROOT, "logs")
CONDS = ["A", "A_repeat", "V", "N", "VN"]

# Operator action, verbatim: desk3 seed 1 finished its replay, printed the final
# line, and then no longer made progress; after 18 minutes the operator sent
# SIGKILL to the slurm-less child (pid 294640).  It must not be recorded as 0.
OPERATOR_ACTIONS = {
    ("desk3_rgbd", 1): dict(
        host_exit="KILLED_SIGKILL",
        signal="SIGKILL",
        killer="operator",
        note="replay finished and all products were flushed; process did not "
             "leave main()'s join and was killed after 18 min",
    ),
}

# start/end timestamps as printed by run_all.sh
times = {}
starts = {}
with open(os.path.join(LOGS, "batch.log")) as f:
    for line in f:
        m = re.search(r"=== (\S+) START (\S+) seed=(\d+)", line)
        if m:
            starts[(m.group(2), int(m.group(3)))] = m.group(1)
        m = re.search(r"=== (\S+) DONE  (\S+) seed=(\d+) rc=(\d+)", line)
        if m:
            times[(m.group(2), int(m.group(3)))] = (m.group(1), int(m.group(4)))

rows = []
for scene in ["office0_mono", "desk3_rgbd"]:
    for seed in [0, 1, 2]:
        start = starts.get((scene, seed), "")
        end, masked_rc = times.get((scene, seed), ("", -1))
        cond_ok, missing = [], []
        for c in CONDS:
            d = os.path.join(REPLAY, f"{scene}_s{seed}", c)
            need = ["summary.txt", "steps.csv", "event_log.csv"]
            ok = all(os.path.isfile(os.path.join(d, n)) and
                     os.path.getsize(os.path.join(d, n)) > 0 for n in need)
            (cond_ok if ok else missing).append(c)
        # step rows: 6 checkpoints (+ extra at replay_steps_) per condition
        step_rows = 0
        for c in cond_ok:
            p = os.path.join(REPLAY, f"{scene}_s{seed}", c, "steps.csv")
            with open(p) as f:
                step_rows += max(0, sum(1 for _ in f) - 1)
        op = OPERATOR_ACTIONS.get((scene, seed))
        if op:
            host_exit = op["host_exit"]
            note = op["note"]
        elif masked_rc == 0 and not missing:
            host_exit = "EXIT_0"
            note = ""
        else:
            host_exit = "UNKNOWN"
            note = "masked pipeline rc, see per-run log"
        rows.append(dict(
            scene=scene, seed=seed,
            host_pid=("294640" if op else "not-captured"),
            start_utc=start, end_utc=end,
            wall_s=("" if not (start and end) else
                    str(int((__import__("datetime").datetime.fromisoformat(end) -
                             __import__("datetime").datetime.fromisoformat(start))
                            .total_seconds()))),
            host_exit=host_exit,
            masked_rc=masked_rc,
            conditions_complete=len(cond_ok),
            conditions_expected=len(CONDS),
            missing_conditions="|".join(missing),
            products_complete=("yes" if not missing else "no"),
            step_checkpoint_rows=step_rows,
            mapping_complete=("yes" if not missing else "no"),
            note=note,
        ))

fields = ["scene", "seed", "host_pid", "start_utc", "end_utc", "wall_s",
          "host_exit", "masked_rc", "conditions_complete", "conditions_expected",
          "missing_conditions", "products_complete", "step_checkpoint_rows",
          "mapping_complete", "note"]
out = os.path.join(ROOT, "RUN_LEDGER.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(open(out).read())
