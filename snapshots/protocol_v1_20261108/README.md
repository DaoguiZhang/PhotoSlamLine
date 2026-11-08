# Full GPU mapping experiment (locked v1)

Read `VERSION_RECORD.md` first. It fixes the source tree, build, hashes,
YAMLs, switches, manifest and the Full/OFF definition.

## Layout

```
PSL-fullmap-lock-v1/          locked source + Release build
  _lock/                     base metadata, included diff, final artefact hashes
PSL-fullmap-exp-v1/
  manifest.csv               30 formal cells (full sequences)
  config_hashes.csv          per-cell ORB/Gaussian/Full/OFF hashes
  configs/off/               OFF YAMLs (only the three weights differ)
  inputs/                    fixed truncated views for the small samples
  small/                     small-sample runs + acceptance evidence
  runs/formal-v1/            formal batch: <cell>/r<rep>_<cond>/
  report/                    aggregation output
  logs/                      batch log and PID
  scripts/                   make_manifest, formal_batch, check_run, formal_report
```

## Monitor

```bash
tail -f /workspace/code/PSL-fullmap-exp-v1/logs/formal_batch.log
tail -f /workspace/code/PSL-fullmap-exp-v1/runs/formal-v1/summary.csv
```

## Resume

The batch never overwrites a non-empty run directory and skips runs whose
`status.json` has `accept=true`:

```bash
cd /workspace/code/PSL-fullmap-exp-v1
nohup env REPEATS=5 RUN_TIMEOUT=7200 python3 scripts/formal_batch.py \
  >> logs/formal_batch.log 2>&1 &
```

## Check one run

```bash
python3 scripts/check_run.py runs/formal-v1/<cell>/r<rep>_<cond>
```

Accepted means: exit 0, `Total Iterations > 0`, non-empty `psnr.txt`,
`point_cloud.ply` with `element vertex > 0`. A cell that never reaches
`Mapper.min_num_initial_map_kfs` is reported as `Gaussian model is empty` and
is **not** accepted.

## Report

```bash
python3 scripts/formal_report.py
```

Outputs `report/formal_summary_cell.csv`, `formal_summary_group.csv` and
`formal_paired_delta.csv`. Group averages are computed per repeat over cells
(equal weight) and then mean±std over repeats, not by pooling scenes across
repeats.

## Hard rules

* Do not replace the binary or libraries while `formal-v1` runs.
* Any code fix needs a new source tree, build and output root (e.g. `v2`).
* The official upstream baseline is pinned separately in
  `/workspace/code/PSL-upstream-official`; it is not part of this batch.
