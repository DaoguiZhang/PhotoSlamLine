# Ledger reconciliation (single snapshot)

Snapshot: 2026-11-04T17:31Z, root `/workspace/eval_runs/final_runs_v2`.

## The two numbers I reported before were wrong, and why

I reported "218 + 7 + 1 + 258 + 6 = 490 against 492" and "attempts = 228".
Both were artefacts of how I read the state, not missing rows:

1. **490 vs 492.** I added the counts from *two different parts* of
   `matrix.py status` output: the per-dataset rows (`TUM/Replica/ETH3D`, which
   do not include the 6 upstream rows) plus the `BLOCKED` line, while the
   dataset rows had already been truncated by `head -8`. The same snapshot
   recomputed from disk gives **492/492** rows with no row unclassified:

   | state (from the run directories) | count |
   |---|---|
   | COMPLETED (`status.txt` exit_code=0) | 451 |
   | FAILED (-11) | 7 |
   | PARTIAL (directory without `status.txt`) | 1 |
   | PENDING (no directory) | 27 |
   | BLOCKED by `blocked_reason` (upstream rows) | 6 |
   | **total** | **492** |

   The 33 rows with no directory are 27 internal + 6 upstream; only the
   `blocked_reason` field distinguishes them, which is why a naive walk shows
   33 PENDING.

2. **attempts = 228.** That was the size of `ATTEMPT_LEDGER.csv`, which is
   written on demand and was last regenerated at 12:10. The progress log — the
   append-only record of every attempt — had 227 attempts then and **458 now**:
   458 unique ids, `exit=0` x 451, `exit=-11` x 7, no id attempted twice. So the
   ledger was stale, not the log wrong.

## Rules this produced

* The ledger is now regenerated at every evaluation pass
  (`aggregate.py` calls `matrix.py ledger`), and the reconciliation above is
  reproduced mechanically in `eval/reconcile.py`.
* Counts are only ever quoted from one snapshot with the derivation shown, never
  from a truncated `status` printout.

## The five stages are not the same thing

Kept apart everywhere below:

| stage | meaning | where it is recorded |
|---|---|---|
| process exited normally | `exit_code=0` | `status.txt` |
| complete input processed | frames processed == frames in the frozen input list | `final_state.csv` row count vs manifest `n_input` |
| artefacts valid | the files exist and parse (A-E in `eval/acceptance.py`) | `ACCEPTANCE_ARTEFACTS.csv` |
| metrics evaluable | GT reachable for the frames and the map history is unambiguous | per-run `status`/verdict in `ONLINE_PER_RUN.csv`, `MAP_HISTORY.json` |
| finally accepted | all of the above and the statistical tables regenerated | final acceptance report |

A run that crashed after writing a partial trajectory is **not** a successful
experiment: the 7 `-11` cells stay in the failure count and their partial
trajectories are reported separately, never mixed into the completed-run table.
