# perf-stage1 results — minimal cost reduction of Gaussian mapping

Branch `perf/mapper-min-cost`, worktree `/workspace/code/PSL-perf`.
Baseline commit `be26301108`. Manifest (inputs, naming, pre-registered target, timing
conventions): `reports/perf_stage1_manifest.md`.
Archive: `archive/perf-stage1/` (patch, configs, binary ids, raw CSVs, checksums; 20 MB).

`B` = **LineSeed + Photo** (structural loss off). `S` = **LineSeed + Photo + Struct** =
the existing `C` configuration, all three weights at their configured values
(0.001 / 0.002 / 0.005). No other structural configuration is used.

---

## 2. Where the time actually goes (measured, same parameter snapshot)

Controlled micro-benchmark: one fixed parameter snapshot, one fixed view and resolution,
identical warmup, **no `optimizer.step()`**, graph rebuilt and gradients zeroed per
iteration, min of 3 blocks of 200 iterations, wall clock with device sync at every loop
boundary. Measured on the *pre-optimisation* implementation.

Structural-on path at 5000 steps (ms per iteration):

| section | cost | share |
|---|---|---|
| **backward** (`loss.backward()`) | **9.2 - 10.2** | ~80 % |
| **loss forward** (L1+SSIM+aniso+structural) | **0.84 - 1.12** | ~10 % |
| -> of which photometric L1+SSIM | 0.78 - 0.87 | |
| -> of which structural terms | 0.46 - 0.68 | ~6 % |
| render forward | 0.21 (B geometry) / 0.69 (S geometry) | 2-8 % |
| optimizer.step | 0.11 | 1 % |
| zero_grad | 3.0e-05 | ~0 % |

**Top three, with code locations:**

1. **Backward pass** — `loss.backward()` in
   `GaussianMapperLine::trainForOneIterationErrorGuided` (`src/gaussian_mapper_line.cpp`,
   `loss.backward();` right after the structural-loss block).
2. **Loss forward** — `loss_utils::l1_loss(...)` and `loss_utils::ssim(...)` in the same
   function, on the full image; this is the largest single *forward* item and it is on the
   **common** path, not the line path.
3. **Structural loss forward** — `GaussianModelLine::computeVectorizedLineLoss`
   (`src/gaussian_model_line.cpp:230`) and
   `GaussianModelLine::computeLineShapeConstraint` (`src/gaussian_model_line.cpp:4111`).
   For only ~170 line Gaussians these cost 0.46-0.68 ms, i.e. ~30-40 us per tiny op — far
   above kernel-launch cost.

Geometry is a separate effect: the S geometry costs **0.69 ms** to render versus **0.21 ms**
for the B geometry at the same view and resolution. Attributing that to the loss code would
be wrong; it is the inflated Gaussians touching more rasterisation tiles.

The macro `GpuSectionProfiler` section "loss_fwd" confirms only **0.24 ms** of the
structural-path reduction, while the wall-clock loops show ~1.0 ms. The missing part is
**host-side**: boolean-mask indexing on CUDA reads the true-count back from the device, so
each `index({mask})` stalls the host, and a CUDA-event profiler cannot see a host stall.

---

## 3. Stage-1 changes (all mathematically equivalent)

| # | change | evidence it was worth doing |
|---|---|---|
| 1 | **Cached line index**: one `int64` index tensor + `index_select` replaces 6 boolean-mask gathers per iteration | the structural forward cost is 0.46-0.68 ms for ~170 elements; each `index({bool_mask})` forces a device count read |
| 2 | **Constant tensors gathered once**: `xyz_init_` and `line_dir_w_` on the line subset are cached, not re-gathered per call | same |
| 3 | **One shared `exp(scaling_)`**: the anisotropy penalty and the eccentricity term now share one activation instead of building it twice | duplicate intermediate tensor, exactly the item the task lists |
| 4 | **Zero-weight terms skipped**: a zero weight returns before building any graph node; the eccentricity and orientation components are only built when their own weight is non-zero | the task's first checklist item |
| 5 | **`amax`/`amin` instead of `max(dim)`/`min(dim)` values**: the argmax/argmin tensors and their backward nodes are no longer built | measured small (aniso = 0.018 ms/iter) but free |

Cache soundness: the key is `(numel, data_ptr)` of `is_line_` / `xyz_init_` / `line_dir_w_`.
Verified by inspection that **every** writer of those three tensors *replaces* the tensor
(15 assignment sites, no `index_put_`/`copy_`/in-place op anywhere), so a pointer change is a
reliable invalidation signal and a topology change cannot be missed. The cache is only
populated on the first call and after a topology change; steady state does no device work.

Not done, deliberately: no formula change, no precision change, no new sampling, no new CUDA
kernel, no `detach`, no dropped public constraint, no caching of anything that depends on
updated parameters.

### Numerical equivalence — measured, not asserted

`PHOTO_SLAM_FGM_EQCHECK=1` evaluates the optimised and the pre-optimisation implementation
on the **same** parameter snapshot, each from a fresh graph with zeroed gradients, and
compares the loss value and the gradient with respect to each target parameter.

Result at steps 0 and 100, in all four conditions (`B0`, `B1`, `J`, `K`):

```
expression,max_abs_delta_loss,max_rel_delta_grad,verdict
position,0,0,PASS
ecc,0,0,PASS
ori,0,0,PASS
all_struct,0,0,PASS
exp_subset_substitution,0,0,PASS
```

**Bitwise identical (max|Δloss| = 0, max relative |Δgrad| = 0).**

Tolerance and its basis: every change is an algebraic identity — the same rows gathered in
the same (`nonzero` is row-ordered) order, the same `exp` result reused, `term + 0.0f`
dropped, `amax`/`amin` returning the same values. IEEE-754 therefore gives bitwise-identical
results, so the acceptance criterion is **exact equality**; `1e-6` relative is kept only as
an allowance for a different ATen reduction schedule, and anything larger would mean the
"equivalent" claim is false. Nothing was accepted on the basis of "close enough".

---

## 4. Acceptance check (about 100 steps, four conditions in one process)

| gate | result |
|---|---|
| initial model identical | `init_fingerprint` equal for `B0`,`B1`,`J`,`K` |
| step-0 render identical | PSNR `12.84914124`, SSIM `0.2553931912` in all four, bit-identical |
| loss + gradient equivalence | 0 / 0 for all five expressions at step 0 and 100 |
| training views identical | identical sequence **including order** in all four |
| schedule identical | `default_sh` 0/0/0 and `pyr_remaining` 3300/3250/3200 in all four |
| learning rate / SH / pyramid | driven by the same reset state, witnesses equal |
| evaluation views not trained | `eval ∩ trained = 0` at every eval point |
| Gaussian count unchanged | 2006 total / 168 line in all four, `count_violation = 0` |
| evaluation does not change training state | `params_unchanged = 1`, `iteration_unchanged = 1` at every eval point |
| S-original vs S-fast | PSNR 14.90148330 vs 14.90048325 (Δ 0.001 dB); SSIM Δ 0.00006 — below the B0/B1 control (Δ 0.0064 dB / 0.00046), as expected for identical math |

---

## 5. Formal result — seed 777, 5000 optimizer updates, two independent scenes

Order alternated (`PHOTO_SLAM_FROZEN_ROTATE`), sequential, no profiler during timing.

| run | order | tag | mapping s | ms/step | PSNR | SSIM |
|---|---|---|---|---|---|---|
| r0 | B0,B1,J,K | B0 | 24.926 | 4.985 | 21.5942 | 0.827969 |
| r0 | | B1 | 23.825 | 4.765 | 21.6358 | 0.829140 |
| r0 | | **J (S-original)** | **33.667** | 6.733 | 21.6033 | 0.82706 |
| r0 | | **K (S-fast)** | **30.408** | 6.082 | 21.6852 | 0.82889 |
| r2 | J,K,B0,B1 | **J** | **33.998** | 6.800 | 20.8866 | 0.816400 |
| r2 | | **K** | **29.287** | 5.857 | 20.8640 | 0.815749 |
| r2 | | B0 | 23.589 | 4.718 | 20.5691 | 0.811265 |
| r2 | | B1 | 25.016 | 5.003 | 20.5388 | 0.810753 |

Same-snapshot micro-benchmark, structural path only (fast vs reference):

| step | fast fwd | fast bwd | fast total | ref total | saving |
|---|---|---|---|---|---|
| 0 | 1.120 | 0.762 | 1.882 | 2.737 | **0.855 ms (-31 %)** |
| 5000 | 1.692 | 0.635 | 2.327 | 3.309 | **0.982 ms (-30 %)** |

### Q1 — same number of steps, how much time is saved?

| scene | J | K | saving | same-scene control (B0 vs B1) | saving / control |
|---|---|---|---|---|---|
| r0 | 33.667 s | 30.408 s | **-3.259 s (-9.7 %), -0.652 ms/step** | 1.102 s (4.4 %) | 3.0x |
| r2 | 33.998 s | 29.287 s | **-4.711 s (-13.9 %), -0.942 ms/step** | 1.427 s (6.1 %) | 3.3x |

The saving is consistent in sign and size across two independent scenes and is about **3x the
same-scene repeatability**. It is real, but it is not large relative to the noise floor, and
the control pair — two runs of *identical* code — already differs by 4-6 %.

### Q2 — at the same mapping time, does quality drop?

Linear interpolation on the common overlap; nothing extrapolated.

* **K vs J**: K is better at every matched time (+0.03 to +0.43 dB PSNR, r0; +0.04 to
  +0.35 dB, r2), because it fits ~5-7 % more optimizer steps into the same wall time. This
  is purely the speed-up, since J and K do identical math.
* **K vs B at equal time**: K is **worse at every matched time**, by -0.07 dB at the common
  end and up to -1.7 dB early (r0), and -0.19 to -0.70 dB (r2).
  So after Stage 1 the structural run still does not match the structural-off run at equal
  mapping time.

### Q3 — time to the pre-registered target (PSNR >= 20.5 and SSIM >= 0.815)

| run | B0 | B1 | J | K |
|---|---|---|---|---|
| r0 | ~7.45 s | ~7.43 s | ~10.64 s | **~9.63 s** |
| r2 | not reached | not reached | not reached | not reached |

* Stage 1 reduces the time-to-target by **1.01 s (-9.5 %)**: 10.64 s -> 9.63 s.
* **S-fast is still ~2.2 s (+30 %) slower than B to reach the same quality.**
* r2 never reaches the target within 5000 steps for *any* condition, so no comparison is
  reported there — recorded as "not reached", not extrapolated.
* The eval grid is coarse (0/100/250/500/1000/2000/3000/5000) and the time axis was
  interpolated, so these are **interpolated estimates with a bracket**, e.g. K sits between
  6.66 s and 12.66 s.

### LPIPS (offline, on the saved common-view renders, `r0`, at 5000 steps)

| B0 | B1 | J | K |
|---|---|---|---|
| 0.49382 | 0.49362 | 0.49671 | 0.49533 |

K beats J by 0.0014 but is still ~0.0016 worse than B. **PSNR favours the structural term
(K 21.6852 > B0 21.5942), LPIPS does not** — the same sign disagreement already found in
`fgm-conv-v2`.

### Scale tails of the line Gaussians (required check)

| tag | step | s_parallel p50 | s_parallel p90 | s_parallel max | s_perp p50 | aspect p50 | aspect p90 |
|---|---|---|---|---|---|---|---|
| B0 | 5000 | 0.1433 | 0.2649 | 2.146 | 0.06028 | 2.23 | 5.90 |
| B1 | 5000 | 0.1390 | 0.2789 | 2.125 | 0.06123 | 2.20 | 5.02 |
| J | 5000 | 2.3534 | 2.5017 | 2.650 | 0.00436 | 544.79 | 576.85 |
| K | 5000 | 2.3898 | 2.5043 | 2.692 | 0.00435 | 547.06 | 579.93 |

J and K produce the **same geometry** (0.015 % apart at 1000 steps, 1.5 % at 5000 — run-to-run
nondeterminism), confirming Stage 1 changed no behaviour.

---

## 6. Stage 2 (lower-frequency structural updates) — **not run**, and why

The task allows Stage 2 only if the structural configuration passes the scale-stability check
*and* has a quality benefit worth keeping. Neither holds:

* **Scale stability: fails.** The eccentricity term drives the median longitudinal scale from
  0.140 m to **2.39 m** in a scene whose bounding-box diagonal is 3.42 m, collapses the
  cross-section to 0.0044 m and the median aspect ratio to **547**, and takes the line
  Gaussians' mean opacity from 0.260 to 0.029. Stage 1 is mathematically equivalent, so it
  changed none of this.
* **Quality benefit: not stable.** PSNR improves (r0 K 21.6852 vs B0 21.5942; r2 K 20.8640 vs
  B0 20.5691) but LPIPS degrades, and in the `fgm-conv-v2` runs the sign flipped between
  scenes. At equal mapping time the structural run is worse in every measurement.
* Therefore a `S-fast-K5` condition would spend its budget computing a term whose own
  configuration is unstable, which the task explicitly forbids ("不要用低频计算掩盖旧偏心项
  的尺度失稳"). **Skipped, as instructed, with the reason recorded.**

Note also that the S condition already costs ~30 % more than B at equal quality even after
Stage 1, so Stage 2 would have to buy a real quality gain to be worth running at all.

---

## 7. What this round does and does not claim

* **Implementation speed-up: yes.** The structural path is ~30 % cheaper (0.86-0.98 ms/iter
  measured in isolation on one snapshot), and 5000 optimizer updates take **9.7 % / 13.9 %**
  less wall clock in two independent scenes, ≈3x the same-scene repeatability.
* **Training-strategy change: none.** Stage 1 is bitwise equivalent; Stage 2 was not run.
* **Full online-system speed-up: NOT claimed.** The measurement is a frozen-geometry replay
  with the Gaussian count pinned. In the online system the changed geometry would change
  rasterisation cost, and densification/pruning would add work that this protocol removes.
* **S-fast being faster than S-original does not mean the structural method is a net win.**
  K is still slower than B (~30 % at equal quality on the scene where the target is reached),
  and worse than B at equal mapping time on both scenes.

**Next bottleneck with evidence:** the backward pass (≈80 % of the iteration) and the
photometric loss forward (0.78-0.87 ms, and it is on the common path). The structural path is
now a small share. The remaining structural overhead is dominated by the extra backward pass,
which cannot be removed without changing the loss.

**Recommended default:** with no stable equal-quality time benefit, keep the structural terms
**off by default**, and treat the eccentricity term's unbounded ratio as a bug to fix if the
structural regulariser is to be revisited (the fix is a bounded/hinged ratio; not implemented
here because it changes the loss and the task forbids changing the formula this round).
