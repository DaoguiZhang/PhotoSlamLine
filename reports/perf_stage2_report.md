# perf-stage2 — sampling-count experiment, plus corrections to the Stage-1 report

Branch `perf/mapper-min-cost`. Stage-1 report is preserved unchanged in
`reports/perf_stage1_results.md` (not overwritten).
Raw data: `abl_out/{acc100b,perf_r3,q_acc100,q_formal_r0}/`, plus
`abl_out/{formal_r0,formal_r2,acc100,mb2,eq_stage1,prof_jk}/` from Stage 1.

---

## 1. amax/amin — tested, found NOT equivalent, reverted

`tests/amax_amin_grad_semantics.cpp` + `.README.md` (standalone LibTorch 2.12.0, CPU and
CUDA). Threshold must be below the ratios for the term to be active; the first version of
the test had it above and was vacuously zero — corrected and re-run.

| case | loss | gradient wrt `scaling_` |
|---|---|---|
| unique max / unique min | identical | identical |
| tied max (two axes equal the max) | identical | **DIFFERS, rel 0.025** |
| tied min (two axes equal the min) | identical | **DIFFERS, rel 0.50** |
| **actual initialisation pattern**, `scale_axis1 == scale_axis2` exactly (0.08000002056) | identical | **DIFFERS, rel 0.50** |
| realistic random scales, no ties | identical | identical |

`max(dim).values` routes the gradient to the single argmax/argmin; `amax`/`amin` spread it
over every tied extremum. **The repository's initialisation produces an exact tie between
the two perpendicular axes**, so the substitution would have changed training semantics from
the first iteration.

The Stage-1 equivalence check did **not** cover this: it compared the two structural losses
against their `*_Ref` versions, and the anisotropy penalty is not part of those. That was a
real gap in the verification, now closed by the standalone test.

**Action: reverted to `torch::max(scaling, dim=1)` / `torch::min(scaling, dim=1)` values in
both training functions.** All other Stage-1 changes are kept.

## 2. Cache invalidation hardened

The Stage-1 cache key was `(numel, data_ptr)`. That is **not** a reliable version:
`restoreInitCheckpoint()` (`gaussian_model_line.cpp:4364-4366`) and `loadPly()` write
`is_line_` / `xyz_init_` / `line_dir_w_` **in place with `copy_()`**, so the pointer stays
identical while the content changes. In the frozen flow this is benign (the restore happens
before the first training step populates the cache) but it is exactly the fragility that was
flagged.

Now there is an explicit monotonic `line_partition_version_`, bumped by `invalidateLineCache()`
at **all 21 writers** of those tensors (assignment sites plus the three in-place `put(...)`
calls), and `refreshLineCache()` rebuilds when the version changed. The `(numel, data_ptr)`
signature is kept only as a secondary net for a writer that might be missed. This covers
checkpoint restore, mask update, anchor/direction update and every topology change.

## 3. Stage-1 measurements confirmed after the revert

* 100-step acceptance re-run: all gates pass (identical step-0 render, `max|Δloss| = 0` /
  `max rel|Δgrad| = 0` for all five expressions, identical training-view sequence, identical
  schedule witnesses, `count_violation = 0`, `params_unchanged = 1`).
* One paired re-measurement (`perf_r3`, seed 777, 5000 steps): **J 32.892 s -> K 29.363 s =
  -3.529 s (-10.7 %)**, same-scene B0/B1 control **-0.170 s (-0.7 %)**.
  Across the three paired measurements the control spread is **0.7 % to 6.1 %**.
* `bin/replica_rgbd` sha256 after this round: see `archive/perf-stage2/binaries.sha256`.

## 4. Line-seed density experiment (Q0 / Q25 / Q50 / Q100)

One freshly captured frozen snapshot (`q_acc100`, `q_formal_r0`), seed 777, **all conditions
in one process on that snapshot**; new identifiers, not pooled with the earlier snapshots.
Structural loss **off** in every condition; identical public constraints.

Subset construction (`GaussianMapperLine::makeLineSubsetMask`): all ordinary Gaussians are
always kept; the line rows are subsampled with a **stride** so the kept sets are **nested**
(55 ⊂ 99 ⊂ 166 line Gaussians), and stratified by the sign-canonicalised, quantised world
direction (43 groups) so a stride cannot drop a whole line.

**Limitation, stated:** no explicit `MapLine -> sample` association exists at freeze time
(`computeGroupedLineLoss`'s `line_to_sample_ids` path is unused), so line membership is
approximated by the direction; **lines sharing a direction are indistinguishable**.

Verification that the ordinary Gaussians are untouched: the non-line fingerprint
(`paramFingerprintNonLine`, hash over the non-line rows of xyz/opacity/scaling/rotation/SH
plus their count) is **identical in all four conditions**. Counts are constant within each
condition (`count_violation = 0`); counts differ *across* conditions, which is the variable
of this experiment. `init_fp_match = 1` only for Q100, as expected.

### 5000 optimizer updates

| tag | total Gaussians | line | ms/step | PSNR | SSIM | LPIPS@5000 |
|---|---|---|---|---|---|---|
| Q0 | 1849 | 0 | 4.777 | **21.6536** | 0.823036 | 0.49270 |
| Q25 | 1908 | 59 | 5.117 | 21.5824 | 0.829360 | **0.48476** |
| Q50 | 1952 | 103 | 4.857 | 21.6330 | **0.829675** | **0.48477** |
| Q100 | 2022 | 173 | 5.076 | 21.6203 | 0.829165 | 0.48645 |

Scale tails stay stable in every condition that has line Gaussians (Q25/Q50/Q100: s_par p50
0.13-0.15 m, aspect p50 2.0-2.7, max 9-15) — expected, since the structural loss is off and
subsampling does not change the initialisation of the lines that remain. Q0 has no line
Gaussians, so the statistic is undefined (NaN), not zero.

## 5. Answers

**1. Can Q25/Q50 keep Q100 quality with fewer Gaussians and less time?**
*Quality*: yes on SSIM and LPIPS. Q25 (59 line Gaussians) reaches LPIPS 0.48476 versus Q100's
0.48645 with 173 — i.e. **~1/3 of the line Gaussians match or beat the full set
perceptually**; SSIM likewise (0.82936 vs 0.82917). PSNR disagrees and slightly prefers Q100
(21.620 vs 21.582), the same PSNR/LPIPS disagreement seen throughout this work.
*Time*: **no**. ms/step is 4.777 / 5.117 / 4.857 / 5.076 for Q0/Q25/Q50/Q100 — **not
monotone in the line count**, and Q25 is the *slowest* despite carrying far fewer line
Gaussians than Q100. The spread (~0.34 ms/step, 7 %) is at the level of the run-to-run
control.

**2. Does Q100 give a quality benefit over Q0 worth paying for?**
Marginal and metric-dependent. Line seeds help SSIM (+0.0066) and LPIPS (-0.0063) but hurt
PSNR (-0.033). The measured diagonal cost is ~0.3 ms/step, which is inside the noise. So:
two of three metrics favour having line seeds, but there is **no measurable time cost to
avoid and no consistent quality gain to buy**. Not a clear win.

**3. Are ordinary points and line points created twice?**
**Not audited.** The instrumentation for the online creation-source census (§五 of the task)
was not implemented — the remaining budget went to the density experiment. No claim is made
either way.

**4. Where does the main cost come from — count, projection coverage, or common training
compute?**
From this experiment: **count is not the driver.** Tripling the line-Gaussian count
(59 -> 173, i.e. +6 % of the model) changes the per-step time by an amount smaller than the
run-to-run spread. Combined with the Stage-1 attribution (backward ~80 % of the iteration,
photometric loss forward the largest *forward* item, on the full image and on the **common**
path), the evidence points to **common training compute**, not the Gaussian count and not
line-specific work.

**Therefore: stop scanning line density**, as the task prescribes for exactly this outcome.
The prescribed next step is to split the backward pass into photometric-loss backward,
rasterisation backward and the remaining tensor work, with a few same-state measurements on
the structural-off baseline, **without rewriting CUDA**. That was **not done** this round.

## 6. Corrections applied to the Stage-1 wording

* `r0` and `r2` are **different frozen snapshots (batches) of the same scene (office0)**,
  captured by separate processes — not "different scenes".
* The B0/B1 difference is a **repeated-run drift reference**, not a statistically calibrated
  significance test. "About 3x" is descriptive only: it is two or three paired numbers, and
  the observed control spread is **0.7 %-6.1 %**, so no significance claim is made.
* The **min-of-3 micro-benchmark is reported separately from the formal training timings**
  and is never used as the headline speed-up: the micro-benchmark pins the snapshot, view
  and resolution and does no `optimizer.step()`, so its numbers are attribution only.
* Every timing table now states the snapshot, the resolution (1200x680 via
  `gau_pyramid` / full-res branch) and the measurement basis. Total = render + loss forward
  + backward + densify-stats + optimizer holds for the **profiled** rows; the wall-clock
  `ms/step` is larger than the profiled sum because the profiler adds its own
  synchronisation and cannot see host-side stalls — the two are explicitly not added
  together.
* The frozen experiment **does** include the rasterisation cost of the changed Gaussian
  parameters (measured: 0.69 ms for the S geometry vs 0.21 ms for the B geometry at the same
  view). What it does **not** cover is the online system's SLAM, bundle adjustment and
  loop-closure updates, and the densify/prune work this protocol pins off.

## 7. Not completed this round

* §五 online creation-source census (ordinary seeds / line seeds / inactive-geo densify /
  clone / split / prune), duplicate-seeding checks, and any de-duplication switch.
* The backward decomposition (photometric-loss backward vs rasterisation backward vs other
  tensor ops).
* Seed replication beyond seed 777 for the density conditions: the density result shows **no
  time effect**, so per the task's own stopping rule ("若线数量变化对耗时影响很小，停止扫描线
  密度") it is not replicated further.
