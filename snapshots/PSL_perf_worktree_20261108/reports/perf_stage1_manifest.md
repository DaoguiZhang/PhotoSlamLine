# perf-stage1 manifest — fixed before the candidate results were seen

**protocol_version: `perf-stage1`** (branch `perf/mapper-min-cost`, worktree `/workspace/code/PSL-perf`)
Baseline of this work: commit `be26301108` ("baseline: frozen-geometry ablation harness").

## Scope of this round

* Stage 1 only: remove duplicated computation and unnecessary overhead in the existing
  training path. **Training objective and schedule unchanged.**
* Stage 2 (reducing the structural-loss update frequency) is **not** run in this round.
  Reason recorded in section 6 of the report: the structural configuration in the repo
  (`C`/`K`: position + eccentricity + orientation) does **not** pass the scale-stability
  check — the eccentricity term drives the median longitudinal scale from 0.14 m to 2.38 m
  in a 3.42 m scene and the line Gaussians' mean opacity from 0.26 to 0.029 — and the
  quality benefit is not stable across metrics (PSNR favours it, LPIPS does not, sign flips
  between scenes). Lowering the frequency of a term whose own configuration is unstable
  would mask that instability, which the task explicitly forbids.

## Frozen input (identical for every condition of a run)

* One frozen scene per process, produced once by SLAM and then replayed; **SLAM is not
  re-run per condition**.
* Shared initialisation checkpoint restored by all conditions; identity proven by
  `init_fingerprint` equality and by the bit-identical step-0 evaluation.
* Training schedule: pre-generated keyframe sequence, identical for all conditions
  (verified against the views *actually visited*, including order).
* Scene: Replica `office0`, RGB-D. Seed 777.
* Fixed geometry / fixed Gaussian count: densify, prune, append and opacity reset are off
  (`PHOTO_SLAM_FROZEN_FIXED_COUNT=1`).

## Conditions and naming

| tag | meaning | structural implementation |
|---|---|---|
| `B0`, `B1` | **B-original / B-fast**: structural loss **off** | irrelevant (no structural term is evaluated); the pair bounds the mapper-side repeatability |
| `J` | **S-original**: structural loss **on**, pre-optimisation implementation | `computeVectorizedLineLossRef` + `computeLineShapeConstraintRef` |
| `K` | **S-fast**: structural loss **on**, optimised implementation, same weights | `computeVectorizedLineLoss` + `computeLineShapeConstraint` |

`S` = the configuration the repo already had, i.e. the earlier **`C`** condition: all three
structural weights at their configured values
(`weight_line_coherence = 0.001`, `weight_line_shape_ecc = 0.002`,
`weight_line_shape_ori = 0.005`). No other configuration is used, and **no shape-parameter
experiment is started**.

`PHOTO_SLAM_FGM_USE_REF` selects the implementation globally; the per-condition tags `J`/`K`
force it, which lets S-original and S-fast run as two conditions of ONE process on ONE
frozen scene. That is the only way to compare them on identical input (the SLAM front-end is
not reproducible across processes: measured `kfs=109` vs `117` for the same seed).

## Evaluation grid (identical for all conditions)

Completed optimizer steps: **0, 100, 250, 500, 1000, 2000, 3000, 5000**.
Views: 24 fixed, evenly spaced keyframes, never sampled by the training sequence.
Metrics: PSNR, SSIM, LPIPS (offline, on the saved renders), per condition.
Scale tails: line-Gaussian 3-axis scale and aspect-ratio quantiles.
Gaussian counts at every eval point.

## Quality target — set in advance, from the pre-existing baseline

Taken from the already-recorded `fgm-conv-v2` baseline (`v2_seed*`, `abl2_*`), where the
structural-off condition reaches roughly PSNR 20.6-21.7 and SSIM 0.82-0.83 across scenes.

**Pre-registered common target: PSNR >= 20.5 AND SSIM >= 0.815.**

* The same threshold is applied to every condition.
* LPIPS is reported alongside but is **not** part of the gate (the CUDA LPIPS build is
  broken here, so it runs on CPU and its absolute scale is not comparable to published
  numbers).
* The eval grid is coarse on a log-like spacing, so a "steps to target" figure is reported
  as an **interpolated estimate and as a bracket (the two adjacent eval points)**, never as
  an exact step count.
* If a condition does not reach the target, that is reported as "not reached" — no
  extrapolation.

## Timing conventions

* `cum_mapping_seconds` covers ONLY the optimizer-step window: the timer wraps
  `trainForOneIterationErrorGuided()` and is bounded by device synchronisation on both sides.
* Data loading and host-to-device transfer are **not** included; the convention is identical
  for every condition.
* Evaluation, checkpoint saving and plotting are timed separately (`cum_eval_seconds`) and
  excluded from `cum_mapping_seconds`.
* The profiler/micro-benchmark are **not** enabled during the formal timed runs.
* Conditions are run **sequentially** in one process (no GPU contention) and the run order
  is alternated across repeats to reduce order bias.
* Bitwise determinism is not claimed: the pipeline's CUDA reductions are non-deterministic.
  The same-scene `B0` vs `B1` pair is the measured tolerance.

## Distinction the report must keep

* **implementation speed-up** — Stage 1 (this round);
* **training-strategy change** — Stage 2 (not run);
* **full online-system speed-up** — not claimed; the measurement is on a frozen-geometry
  replay with the Gaussian count pinned, so it does not include the rasterisation cost that
  a changed geometry would add in the online system.
