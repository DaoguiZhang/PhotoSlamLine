# mapper-budget-v2-hash — result

Branch `feat/mapper-budget-v1`. The measurement definition was frozen before any R50 number was seen
(`reports/mapper_budget_v2_manifest.md`).

**Completion criterion met: there is a runnable replay and a real paired result.** The result is
negative, and it is reported as such.

```bash
# one command: paired replay + summary
W=/workspace/code/PSL-perf CONDS=R100,R50 FIXED_COUNT=0 EVAL_STEPS=0,1000,5000 EVAL_VIEWS=12 \
  PHOTO_SLAM_REPLAY_INACTIVE=1 PHOTO_SLAM_REPLAY_INACTIVE_PERIOD=45 \
  PHOTO_SLAM_INACTIVE_SEED=20260101 bash scripts/run_frozen_abl.sh bv2_formal_r0 5000 office0
python3 scripts/summarize_budget_v2.py abl_out/bv2_formal_r0
```

## 1. What was built

| piece | where |
|---|---|
| unified per-step record + snapshot trigger for **both** training loops | `GaussianMapperLine::increaseIteration` → `recordTrainStep()` → `maybeSnapshotAndAudit()` |
| snapshots at 0 / 100 / 9000 / 20000 / final, one per target step, output directory recorded | `maybeSnapshotAndAudit()` |
| the ONE algorithmic variable (acceptance fraction of the inactive-geo pool) | `budgetSetCondition()` / `applyInactiveBudget()`; condition tags `R100` / `R50` |
| replay driver that feeds the inactive-geo entry **without the SLAM keyframe thread** | `replayInitInactiveDriver()` / `replayDriveInactiveGeo()`, called from the replay loop before each step |
| variable topology (fixed-count gating released) | `PHOTO_SLAM_FROZEN_FIXED_COUNT=0`; clone / split / prune / opacity run normally in both arms |
| kept-index + hash audit | `<arm>/inactive_kept.csv`, `<arm>/inactive_batches.csv` |

Why no scene persistence was needed: the inactive-geo entry only requires keyframe fields
(`kps_pixel_`, `kps_point_local_`, `img_undist_`, `intr_`, `image_width_`) that the **frozen scene
already carries**, so the driver can supply the stream directly. It is called at a fixed
optimizer-step period in ascending keyframe-id order, so both arms get the identical stream with no
SLAM thread and no wall-clock dependence.

## 2. Stream symmetry — the confound that was found and fixed

The first attempt (100-step check) showed **11 batches in R100 but 1 in R50**, and R50's single kept
batch contained keyframe ids left over from R100. Cause: `depth_cached_` and the `depth_cache_*`
accumulators are per-run state and were **not reset between arms**, so the second arm inherited the
first arm's leftover candidates. That would have made the comparison invalid. Fixed in
`replayInitInactiveDriver()` (reset the cursor and both caches).

Verified on the formal run — the two arms received **exactly the same pool**:

| arm | batches | candidates | kept | keep ratio |
|---|---|---|---|---|
| R100 | 11 | 148 345 | 148 345 | 1.000 |
| R50 | 11 | 148 345 | **74 176** | **0.500** |

Same `init_fingerprint` `029c3e7bd9f04f7e_n2054_nline186`, same `input_hash`
`78970620a5242603`, same `kf_seq_hash` `c2c47d1b2d9cf55d`, `init_fp_match=1`,
`count_violation=0`, `nan=0` in both arms, and the step-0 evaluation is identical
(PSNR 12.7098 / SSIM 0.2340 in both before any optimizer update).

## 3. Paired result (5000 steps, one pair, same recorded update sequence)

| | R100 | R50 | Δ = R50 − R100 |
|---|---|---|---|
| PSNR | 40.8008 | 40.3833 | **−0.4175 dB** |
| SSIM | 0.976854 | 0.975297 | **−0.001557** |
| final Gaussians | 144 734 | **123 684** | **−21 050 (−14.5 %)** |
| line Gaussians | 26 | 26 | 0 |
| mapping seconds | 23.0015 | 23.5907 | **+0.589 s (+2.6 %)** |
| refine seconds | 24.108 | 24.711 | +0.603 s |
| NaN | 0 | 0 | — |

Count trajectory (from the arms' own `*_metrics.csv`): R100 2054 → 144 734; R50 2054 → 123 684.

## 4. Verdict against the pre-registered gate

| gate (one-sided, Δ = R50 − R100) | required | measured | verdict |
|---|---|---|---|
| PSNR | ≥ −0.05 dB | **−0.4175 dB** | **FAIL** (8× outside) |
| SSIM | ≥ −0.001 | **−0.001557** | **FAIL** |
| LPIPS | ≤ +0.002 | **not measured** | — |
| mapping time | faster (3 % engineering target) | **+2.6 % slower** | **not faster** |

Applying the decision rules fixed in advance:

* **Not a candidate.** It is not "faster with acceptable quality" — it is **slower and worse**, and it
  fails the quality gate on both metrics that were measured.
* **Do not call it an equal-quality speed-up**, and do not call the +2.6 % a regression either: that
  magnitude is inside the previously measured same-scene wall-clock drift (0.7–6.1 %), so the correct
  statement is **"no measurable reduction in mapping time"**, not "slower".
* **The budget was not compensated in count**: R50 ends with 21 050 fewer Gaussians, so the reduction
  was *not* refilled by later clone/split densification — and yet the time did not fall. That is the
  "budget offset by other entries" case in the shape that most clearly indicts the target: removing
  14.5 % of the final Gaussians bought no time at all.
* This is consistent with section 三 of the previous round: per-step cost is largely
  count-insensitive at these sizes, so trimming the点 budget is not where the mapping time lives.

Consequently, per the pre-registered stopping rule: **stop ratio scanning** (no 0.25 / 0.75 scan), and
let the next round be decided by the real-scale kernel profile, not by another budget ratio.

## 5. What is NOT established by this round

* **LPIPS was not measured** (no LPIPS pass was run for this pair), so one of the three pre-registered
  quality gates is unevaluated. The PSNR/SSIM failures are sufficient to reject the candidate, but the
  gate is not fully discharged.
* **The arms' stream is generated from the frozen keyframes, not replayed from the office0 capture.**
  The per-step external-input recording asked for in item 二 (ordinary points, line points, inactive
  candidates with global order relative to the actual update) is **not** what fed this run. What the
  arms demonstrably share is the *frozen scene, the init checkpoint, the schedule and the generated
  stream* (148 345 candidates each) — not the captured stream.
* **Candidate-generation cost is omitted** from both arms (the driver calls the entry directly), so the
  measured mapping seconds are *not* full online-SLAM time and must not be compared with the earlier
  capture's wall clock.
* **Snapshot completeness is still partial.** The unified hook now covers both loops, but the run above
  did not exercise the online capture path, so the 0 / 100 / 9000 / 20000 / final snapshot set has not
  been demonstrated end-to-end in a single online run.
* One pair only; no repeat with swapped order, no multi-seed, no multi-scene. The earlier limitations
  (initialisation colour provenance not fully traceable, front end not reproducible across processes)
  still stand.
* Absolute PSNR here (40.4–40.8) is far above the earlier office0 frozen tables (~21 dB) because this
  is a **different frozen snapshot with 12 evaluation views**; only the within-run R100-vs-R50
  difference is interpretable.

## 6. Deliverables

* modified files: `include/gaussian_mapper_line.h`, `src/gaussian_mapper_line.cpp`,
  `reports/mapper_budget_v2_manifest.md`, `scripts/summarize_budget_v2.py`, this report.
* input identifiers: `abl_out/bv2_formal_r0/run.meta` (commit, exe/library sha256, configs, scene,
  seeds, `input_hash`, `image_hash`, `kf_seq_hash`).
* raw CSVs: `abl_out/bv2_formal_r0/frozen/summary_R100R50.csv`, `<arm>_metrics.csv`,
  `<arm>/inactive_batches.csv`, `<arm>/inactive_kept.csv`.
* count and quality-vs-time curves: `<arm>_metrics.csv` (per-step `cum_mapping_seconds`,
  `total_gaussians`, `eval_psnr`, `eval_ssim`).
