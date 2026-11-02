# MASTER MANIFEST — cross-dataset front-end evaluation (authoritative scope)

This file SUPERSEDES `INVENTORY.md` section 1 (which listed Replica/ETH3D as
"not run, blocked on loader"). Those were implementable, not blocked; the loader
and the configs now exist. `INVENTORY.md` is kept only as history.

Machine-readable manifest: `/workspace/eval_runs/MASTER_MANIFEST.csv`
(regenerate with `python3 scripts/matrix.py build`, inspect with
`python3 scripts/matrix.py status`).

## Scope actually enumerated

| dataset | modality | scenes | arms | reps | rows |
|---|---|---|---|---|---|
| TUM | mono | 8 | T0,P1,P2 | 3 | 72 |
| TUM | RGB-D | 7 (+1 blocked) | T0,P1,P2 | 3 | 63 (+9 blocked) |
| Replica | mono | 8 | T0,P1,P2 | 3 | 72 |
| Replica | RGB-D | 8 | T0,P1,P2 | 3 | 72 |
| ETH3D | mono | 11 | T0,P1,P2 | 3 | 99 |
| ETH3D | RGB-D | 11 | T0,P1,P2 | 3 | 99 |
| upstream Photo-SLAM | mono | 2 (cabinet) | UPSTREAM | 3 | 6 (all BLOCKED) |
| **total** | | | | | **492** |

Status at manifest build: **477 PENDING, 15 BLOCKED**.
Blocked rows are only:
* `TUM freiburg1_room RGB-D` ×9 — the repository has no `freiburg1_room` RGB-D
  config; generating one is not a copy of a same-sensor template I could verify,
  so it is declared rather than guessed.
* 6 further TUM rows with a missing association file (reason recorded per row).
* upstream Photo-SLAM — binary not built (see below).

Every other row is runnable: configs exist, inputs exist, associations exist.

## Prerequisites completed in this session

| item | status | evidence |
|---|---|---|
| evaluator unified | DONE | `eval_tum_se3_v2.py` is the single source of ATE/RPE; per-estimate-frame dump extended with rotation error and alignment `R,t,s` |
| correct coverage + continuous tracking | DONE, tests green | `eval/coverage.py`, `eval/test_coverage.py` — 9 required cases pass, incl. "rms(per-frame err) == ATE" |
| RGB-D pure-point dispatch | DONE | `EVAL_ARM=T0` → `TrackRGBD` (no line API at all); P1/P2 → `TrackRGBDWithLine`; arm/mode mismatch aborts; `[EVAL-DISPATCH]` witness in every run |
| RGB-D abnormal cost located | DONE, root cause found | staged timing: 99.4 % of wall clock was the pacing sleep; caused by a non-time-ordered association file; fixed and guarded (entry rejects non-monotonic assoc) |
| Replica loader | DONE | `replica-mono` / `replica-rgbd` modes (numbered frames, no timestamp file), synthetic 1/30 s time base recorded |
| Replica depth config | DONE | `RGBD.DepthMapFactor: 6553.5`, cross-checked: max depth 23770 → 3.63 m |
| ETH3D configs | DONE | 11 sequences × 2 modalities, intrinsics read from each `calibration.txt`, depth factor 1000 (mm), max 18721 → 18.7 m |
| upstream build + read-only measurement | **BLOCKED** | upstream Photo-SLAM requires its Gaussian/torch/CUDA backend, which this worktree deliberately does not build; no upstream binary exists, so no upstream number is reported |

## Persistent execution

The matrix is executing **serially** under a resumable scheduler:

```
PID       : see /workspace/eval_runs/matrix.pid (scheduler process)
log       : /workspace/eval_runs/matrix_run.log
progress  : /workspace/eval_runs/matrix_progress.log   (one line per finished run)
per-run   : /workspace/eval_runs/v2/<dataset>/<scene>/<modality>/<arm>/rep<k>/
            meta.txt (cmd, env, start) · stdout.log · status.txt (exit_code, wall)
resume    : python3 scripts/matrix.py run     # skips COMPLETED, never overwrites
status    : python3 scripts/matrix.py status
```

A run directory is created before the process starts and never reused; a failed
run keeps its `status.txt` with the real exit code and is not retried.

## Versioning rule (no silent table mixing)

* `v1` = first cabinet batch. **INVALID_WRONG_BUILD** — built from sources that
  did not match `d3214194`. Retained, excluded from results.
* `v2a` = 36 cabinet cells on build `lib=87f72874… , entry=2f4a2ab5…`. The mono
  half used the `WithLine` API for all arms (dispatch was env-derived, not
  API-explicit) and has no `[EVAL-DISPATCH]` witness; the RGB-D half was fed a
  non-time-ordered association file and is **INVALID**.
* `v2b` (current) = build `lib=87f72874… , entry=ff97a6d2…`: explicit dispatch,
  witness, stage timing, monotonic-assoc guard, Replica loader. Everything the
  scheduler produces under `eval_runs/v2/` is `v2b`.

Rows are only merged into one table if they share the same build hashes, which
are stamped into every manifest row (`build_entry_sha256`, `build_lib_sha256`).

## Known metric limitation carried forward

On these near-planar monocular sequences the position-only Sim(3) alignment
leaves a systematic rotation residual (median 20.5° on
`large_cabinet/mono/T0/rep1`), so a **joint** 0.20 m + 5° gate yields 0 for every
cabinet mono run. `eval/coverage.py` therefore reports, side by side:
joint-gated coverage, translation-only coverage, the lower bound
(correct / all input frames) and the undetermined fraction. None of these is
N/A — they are measured — but the joint-gated column must not be read as
"the system never tracked".
