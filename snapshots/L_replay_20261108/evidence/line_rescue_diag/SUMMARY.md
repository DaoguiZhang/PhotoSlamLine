# Line-rescue tracking: evaluation + diagnostic audit (2026-10-13)

Branch: `fix/line-rescue-tracking`. Dataset: `fr3_structure_notexture_near`
(TUM RGB-D, 1054 association pairs), dedicated relaxed ORB config
`tum_freiburg3_structure_notexture_near_lowfast.yaml` (1054/1054 frames tracked).

Conditions:

| cond | `PHOTO_SLAM_LINE_MODE` | `PHOTO_SLAM_LINE_RESCUE` | meaning |
|---|---|---|---|
| A | 0 | 0 | point-only baseline (no line extraction) |
| B | 2 | 0 | line frontend + line edges in pose optimisation, rescue OFF |
| C | 2 | 1 | identical to B but the rescue path is armed |

## 1. Offline evaluation was wrong: `n=3551` is not a frame count

| file | lines | uniq ts | first ts | last ts | order |
|---|---|---|---|---|---|
| `groundtruth.txt` | 3645 | 3645 | 1341836644.5361 | 1341836680.9757 | asc |
| A/B/C trajectory | 1054 | 1054 | 1341836644.4643 | 1341836680.9605 | asc |
| `rgb.txt` | 1082 | 1082 | 1341836644.4643 | 1341836680.9605 | asc |

GT and the estimates share the **same absolute bag time base** with near-total
overlap, so **no time shift is legitimate or needed**. An earlier claim that the
GT window barely overlapped the trajectory was a misread and is retracted.

`scripts/eval_tum_se3.py::associate(ts_a, ts_b)` iterates over `ts_a` with a
monotone pointer and never de-duplicates `ts_b`. Invoked as
`eval_tum_se3.py <gt> <est>` it iterates the **dense GT (~100 Hz)** and maps many
GT samples onto the same estimate pose, so the printed `n` counts *GT* samples,
not evaluated estimate frames:

```
gt-first : rmse=0.022837 ... n=3551     <- 3645 GT samples, estimate poses reused
est-first: rmse=0.022824 ... n=1052     <- one error per estimate frame
```

The **pair count/labelling** was the defect; the RMSE bias is negligible because
duplicated GT samples resolve to the same nearest estimate pose. The earlier
table's *numbers* therefore survive, but its `n` and its interpretation do not.

Fix: `scripts/eval_tum_se3_v2.py` — estimate-centric, at most one error per
estimate frame, GT interpolation (linear translation + slerp rotation), reports
the true pair count, the time-residual distribution, SE(3) ATE RMSE (Umeyama,
scale fixed to 1), per-frame sigma and RPE. It aborts **without ATE** if the time
windows do not overlap.

### Corrected single runs (preserved `/tmp/lr_abc`, `/tmp/lr_rescue`)

| cond | pairs | ATE RMSE (m) | mean | median | per-frame σ | max | RPE Δ=1 (m/s) |
|---|---|---|---|---|---|---|---|
| A1 | 1052/1054 | 0.022824 | 0.021576 | 0.020784 | 0.007448 | 0.054364 | 0.2507 |
| B1 | 1052/1054 | 0.030187 | 0.028259 | 0.026969 | 0.010619 | 0.056302 | 0.2461 |
| C1 | 1052/1054 | 0.020544 | 0.018730 | 0.019131 | 0.008446 | 0.050777 | 0.2475 |

2 estimate frames are unmatched (the first two precede the GT start at
6644.5361). Median time residual is 0.0000 s, max 0.0047 s.

## 2. Online diagnostics: why `lnEdges=0` was meaningless

The old `[LineRescue]` line printed `lnEdges` / `lnUniqIn` from
`mLineRescueEdges` / `mLineRescueInliers`, which are assigned **only inside**
`HasReliableLineSupport()`, which is called **only when `old_ok == false`**. On a
sequence where the rescue never triggers they are *structurally always 0*. They
are **not** evidence about line participation. That claim is withdrawn.

Two further口径 defects were fixed:

* `mnMatchesInliers` / `mnMatchesInliersLine` are assigned only in
  `TrackLocalMap*`, so at the MotionModel / TrackReferenceKF stages they hold the
  **previous frame's** values. The rescue point floor now recomputes the count
  from the current frame; `[LineRescue]` no longer prints those stale members.
* `mLineRescueInliers/Edges` are now reset at the top of every `ApplyLineRescue`
  call, so they can never leak across calls.

### New diagnostics (all default OFF, no behaviour change when unset)

* `PHOTO_SLAM_DEBUG_LINE_GATE=1` → `[LineGate]`, emitted at the **original
  early-return sites BEFORE any bypass** (`TrackReferenceKeyFrame` `<15`,
  `MotionModel` `<20`), recording frame id, point matches and
  `old_would_fail=`. The field naming was changed from `bypass=` to
  `rescueEnabled=` so it cannot be misread as "a bypass happened"; a real bypass
  requires `old_would_fail=1 AND rescueEnabled=1`.
* `PHOTO_SLAM_DEBUG_LINE_RESCUE=1` → `[LineRescue]` with
  `ptBeforePO lnBeforePO ptAfterPO lnAfterPO oldPtVerdict new reason
  rescueLnUniq rescueLnEdges gateMmOldFail gateTrkRefOldFail`. The pre-bypass
  result (`gate*OldFail`) and the post-optimisation verdict (`oldPtVerdict`) are
  **separate fields**.
* `PHOTO_SLAM_DEBUG_LINE_EDGES=1` → `[LineEdgeDiag]` from
  `Optimizer::PoseOptimizationWithLine`, one line per call, counting the
  optimizer-local edge vectors of *that* call: `NL lnTrackedNonNull ptEdges ptIn
  ptOut ptChi2 lnEdges lnEdgesMono lnEdgesStereo lnIn lnOut lnUniq lnSpread
  lnChi2`. `lnSpread` is the measured directional coverage (deg, wrap-around) of
  the line edges actually used, so degeneracy is measured, not assumed.

## 3. Results — 5 interleaved repeats, diagnostics on, SLAM-only

Interleaved order `r1:A,B,C r2:A,B,C … r5:A,B,C`, identical data/config/eval.

| cond | n | ATE mean (m) | ATE std | min | max | spread |
|---|---|---|---|---|---|---|
| A | 5 | 0.020575 | 0.001459 | 0.018230 | 0.022056 | 0.003826 |
| B | 5 | 0.020646 | 0.001405 | 0.018911 | 0.022264 | 0.003354 |
| C | 5 | 0.022053 | 0.002276 | 0.020297 | 0.025988 | 0.005690 |

All 15 runs: 1054/1054 frames tracked, 1052 evaluable pairs, RPE Δ=1 ≈ 0.24-0.25 m/s.

### Decisive findings

1. **The rescue path is never exercised.** Across all 15 runs:
   `[LineGate]` fired **once per run** (frame 1, TrackReferenceKeyFrame) with
   `old_would_fail=0`; the MotionModel `<20` gate **never fired**; `[LineRescue]`
   produced 1053 decisions **all** with `oldPtVerdict=OK`; `rescued=0`; **actual
   gate bypasses = 0**. See `gate_summary_all_runs.txt`.
   → **There are no frames where the old point-only flow fails and C bypasses it.**
   → No missed-accounting existed either: `old=` was never recording a hidden
   old-path failure, because the old path never failed.

2. **B and C are logically identical.** `GetLineRescueEnabled()` is read in only
   three places, and in all three the condition is gated on a point-test failure
   that never occurs. So B vs C is a **pure null comparison** — and it directly
   measures the noise floor: the observed B−C difference is 0.001407 m, while the
   A−B difference is **0.000071 m (0.3 %)**.
   → The line contribution to ATE on this sequence is **not separable from
   run-to-run noise**.

3. **Lines DO participate in pose optimisation** (the earlier "never" was wrong).
   Per-PO calls with line edges: A = **0** (sanity check), B = 449-638,
   C = 468-536, i.e. ~21-29 % of all `PoseOptimizationWithLine` calls, up to 8
   line edges each, ~1200-1450 line edges per run.

4. **Erroneous lines exist but are contained by the robust kernel.** In run r1_B:
   445 calls carried line edges (21.1 %); 188 (42.2 %) had ≥1 line edge rejected;
   **56 calls had every line edge rejected**; 283/1258 (22.5 %) of all line edges
   were rejected as outliers. Directional coverage: median 58.8°, and only
   27/269 calls with ≥2 line edges had spread < 25° (the bulk of the low-spread
   calls are single-line calls, where spread is 0 by construction).

5. **No measurable degradation ⇒ no suppression added.** Since A and B are
   indistinguishable (0.000071 m vs a ~0.0014 m null-comparison noise floor), the
   condition for adding online-residual/geometry-based line suppression is **not
   met**. No threshold was tuned, no scene-name or GT-based switching was added.

## 4. Limitations (stated plainly)

* n = 5 repeats. The B-vs-C null comparison bounds the noise at ~0.0014 m; the
  A-vs-B effect (0.00007 m) is ~20× smaller, so it is not resolvable here — but
  neither is any modest true effect of the same order.
* SLAM-side nondeterminism is **not seedable**: `PHOTO_SLAM_SEED` only seeds the
  Gaussian mapper (skipped under `PHOTO_SLAM_SLAM_ONLY=1`), and ORB-SLAM3's
  LocalMapping/LoopClosing threads interleave with tracking by wall-clock.
  Repeats therefore quantify the spread; they cannot eliminate it.
* `fr3_structure_notexture_near` cannot evaluate the rescue: with the relaxed
  config the point test never fails. A sequence (or a separately-labelled,
  clearly artificial stress test) that naturally trips the point gate is
  required before the rescue can be assessed at all.
* Single-sequence, single-sensor (RGB-D). No mono/stereo claims.
* No claim of line benefit or harm is made.

## 5. Artifacts

Scripts (new):

* `scripts/eval_tum_se3_v2.py` — corrected estimate-centric SE(3) ATE / σ / RPE.
* `scripts/line_diag_report.py` — ①②③ summary of the three streams.
* `scripts/line_bc_compare.py` — B-vs-C per-frame join; decisive set.
* `scripts/line_quality_report.py` — spread / outlier / chi2 line-quality stats.
* `scripts/fr3_line_diag_abc.sh` — one A/B/C condition with all diagnostics.
* `scripts/fr3_abc_interleave.sh` + `scripts/aggregate_abc.py` — repeats + table.

Evidence (this folder): `ate_preserved_A1B1C1.txt`, `interleave_aggregate.txt`,
`bc_compare_r1.txt`, `bc_perframe_r1.csv`, `gate_summary_all_runs.txt`,
`line_quality_B_C_r1.txt`.
