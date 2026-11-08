# raster-perf-v1 — render bottleneck at ~192 k Gaussians

Branch `feat/mapper-budget-v1`. R100 is retained as the **acceleration baseline**; ratio scanning is
stopped. Loss definition, initialisation, training schedule and sampling policy are unchanged; no
structural loss was added and no further point budget was cut.

**Status.** Item 一 is complete (`reports/mapper_budget_v2_corrections.md`). Item 三 produced **real
kernel-level evidence at 191 918 Gaussians**. Item 二 is **partial** and item 四/五 are **not
implemented** — see the honest boundary in §2 and §4.

## 1. What was measured, and how

Instrumented **inside** the rasteriser (`cuda_rasterizer/rasterizer_impl.cu`), at its own stage
boundaries, with CUDA events: `PHOTO_SLAM_RASTER_PROF=1` plus `PHOTO_SLAM_RASTER_PROF_MIN_P=<P>` so
that only cycles at/above a Gaussian count are profiled (warm-up is therefore implicit, and the
profiling cost is confined to the real-scale segment). Events are **disjoint**, so the stages are not
double counted, and the forward total is reported separately from its parts as a consistency check.
Every 500 profiled cycles the **incremental (independent block) means** are printed, so the series is a
distribution across blocks rather than one growing average. No `optimizer.step` changes, no parameter
changes, no schedule changes. Off by default: with the switch unset no events are created and the
arithmetic is identical.

Run (one capture, office0, 2000 frames): `abl_out/raster_prof_r0`, 19 765 iterations, final
**191 918** Gaussians, 136.7 s billed mapping (~6.9 ms/step at the end).
33 profiled blocks.

## 2. Honest boundary on item 二 (the no-SLAM entry)

The profile above runs **inside the capture process**, so it does *not* satisfy "an entry that does not
start SLAM". What exists for a standalone entry:

* `GaussianModelLine::saveFullSnapshot()` already persists **all** Gaussian tensors (xyz, SH dc/rest,
  scaling, opacity, rotation, `is_line_`, `line_dir_w_`, `xyz_init_`, `max_radii2D_`,
  `xyz_gradient_accum_`, `denom_`, `exist_since_iter_`, `point_ids_`) — not a PLY — and the earlier
  capture wrote `snapshots/model_step20000.pt` (55 MB, 193 354 Gaussians, `default_sh=3`) plus
  `state_step20000.txt`.
* What is **missing** for a standalone load–benchmark: the **views** (per-keyframe
  `R_quaternion_`, `t_`, `intr_`, `image_width_/height_`, image path), the fixed camera
  (`Camera(camera_id,width,height,params,model_id=1)` is sufficient — `setPose` +
  `setCameraParams` + `computeTransformTensors` rebuild a usable `GaussianKeyframeLine`), and the
  undistort mask. These are small and were **not** added this round.

So a no-SLAM entry is a small, well-defined addition (~1 saved `views.pt` in the snapshot, plus a
loader binary) — not a rebuild of the online transaction system — but it was not delivered here, and
the evidence in §3 was obtained in-process instead. That is a real shortfall against the round's
specification.

## 3. Real-scale kernel evidence (191 918 Gaussians, 728 k tile instances/frame)

Tile size is `BLOCK_X=BLOCK_Y=16`, `NUM_CHANNELS=3`. Per-cycle ms, one block (cycles 16 000–16 500,
representative — see the spread below):

| rasteriser stage | ms / cycle | share of rasteriser |
|---|---|---|
| `FORWARD::preprocess` (projection + SH + covariance + bounding) | 0.0173 | 0.3 % |
| `cub` inclusive scan + **host read of `num_rendered`** | 0.0171 | 0.3 % |
| `duplicateWithKeys` (**tile-instance generation**) | 0.1069 | 2.0 % |
| `cub::DeviceRadixSort::SortPairs` (**sort**) | 0.0727 | 1.4 % |
| `identifyTileRanges` (+ `cudaMemset`) | 0.0054 | 0.1 % |
| `FORWARD::render` | 0.1930 | 3.6 % |
| **forward total** | **0.4123** | 7.7 % |
| **`BACKWARD::render`** | **4.9119** | **91.6 %** |
| `BACKWARD::preprocess` | 0.0426 | 0.8 % |

Consistency: forward parts sum to 0.4124 vs the measured forward total 0.4123 — no double counting.

Distribution across the 33 blocks: `BACKWARD::render` **min 3.156 / median 4.920 / max 5.383 ms**;
`preprocess` 0.017, `duplicateWithKeys` 0.099–0.116, `sort` 0.073, `identifyRanges` 0.005,
`FORWARD::render` 0.193–0.199 (all very stable across blocks).

Other recorded quantities: total Gaussians **191 918**; tile instances mean **727 351**, max
**1 101 463** per frame; sorting and instance generation together are **< 4 %** of the rasteriser.
Visible-count and projection-scale tails are available from the existing diagnostics
(`R100_diag_step*.csv`) but were not re-derived for this run; memory was not captured.

**Answers to the specific questions in item 三:**

* **projection/preprocess** — negligible (0.017 ms, 0.3 %). SH→RGB and covariance are *not* the cost.
* **tile-instance generation** — 0.107 ms (2 %); **sort** 0.073 ms (1.4 %). Option B in item 四 is
  therefore **not** where the time is, and cutting redundant Gaussian–tile allocations would target a
  ~4 % slice.
* **render forward** — 0.193 ms (3.6 %).
* **render backward** — **4.91 ms (91.6 %), the dominant kernel by ~25× over the forward render.**
* CPU/GPU: the only host interaction inside the rasteriser is the `cudaMemcpy` of `num_rendered`
  (0.017 ms together with the scan, i.e. ~0.3 %) — host waiting is **not** a rasteriser-level
  bottleneck. **No atomic/bandwidth claim is made:** `backward.cu` contains 7 `atomicAdd` sites
  (per-pixel-per-Gaussian accumulation of `dL_dcolor` per channel, `dL_dmean2D.x/.y`,
  `dL_dconic2D.x/.y/.w`), which makes atomic contention a *candidate* explanation, but this run has no
  counter-level evidence (no Nsight counters were used), so it remains a hypothesis.

**A result worth stating explicitly:** at 1 954 Gaussians the earlier decomposition put the renderer
backward at 5.92 ms/iter; here, at **191 918 Gaussians (98× more)**, the rasteriser backward is
**4.92 ms/iter — essentially the same**. The backward's cost is therefore driven by the
tile/instance workload (~728 k instances for a 16×16-tile image), not by the Gaussian count, which is
the concrete confirmation of why the earlier frozen-pair numbers could not be scaled by count.

## 4. Item 四 — candidate selection and why it is not implemented

Evidence selects **option A**: `BACKWARD::render` dominates. What a "fast splat backward" port would
have to deal with, from the code as it stands:

* the kernel contract is fixed: inputs `tile_grid, block, imgState.ranges, binningState.point_list,
  width, height, background, geomState.means2D, geomState.conic_opacity, colour_ptr,
  imgState.accum_alpha, imgState.n_contrib, dL_dpix`; outputs `dL_dmean2D, dL_dconic, dL_dopacity,
  dL_dcolor`, all accumulated with `atomicAdd` on the **global** arrays;
* the layout is the **standard 3DGS tile layout** (16×16 tiles, sorted per-tile instance ranges,
  `NUM_CHANNELS=3`, SH evaluated in the shared-memory forward preprocess and only its RGB result
  passed down). A compact-representation backward assumes **its own** forward output layout, so porting
  it means rewriting the forward too, i.e. an **algorithmic change to the renderer**, not a
  drop-in equivalent speed-up;
* the mathematical conventions that must be matched exactly: SH degree/eval order, the `exp` scaling
  activation, opacity handling in the alpha compositing, the 0.3σ / bounding-radius culling that
  produces `radii` and hence the instance set, and the gradient of the same culling.

Therefore the correct deliverable this round is the interface difference and the minimal port list,
which is the case item 四 explicitly anticipates ("if the integration cost clearly exceeds this
round, deliver the interface differences, the minimal port list and the evidence"). Not done: no
candidate kernel, no switch, no numeric or speed validation.

**Minimal port list if it is continued:** (1) keep `rasterize_points.h`'s signature and the
`GeometryState/BinningState/ImageState` chunk allocators; (2) replace only `BACKWARD::render` with a
tiled/shared-memory variant that reduces the per-pixel-per-Gaussian atomic traffic, keeping the same
input set (no forward change) — this is the only variant that can be validated as *numerically
equivalent*; (3) validate on one parameter snapshot and view: render output, weighted training loss
and per-parameter-group gradients, pre-registered tolerance (not just the total gradient norm);
(4) only then a ~100-step state check and one paired 5000-step run on the R100 baseline.

## 5. Item 五 — not run

No candidate was implemented, so no numerical equivalence test, no 100-step check and no paired 5000-step
comparison were run. LPIPS remains unmeasured. The pre-registered tolerances that would apply are the
ones already written for the previous round (global relative L2 ≤ 1e-3 on each parameter group,
quality gate PSNR ≥ −0.05 dB, SSIM ≥ −0.001, LPIPS ≤ +0.002).

## 6. Delivery

* modified file: `cuda_rasterizer/rasterizer_impl.cu` (stage profiler + stage marks; off by default).
* command: `PHOTO_SLAM_RASTER_PROF=1 PHOTO_SLAM_RASTER_PROF_MIN_P=100000 bash
  scripts/run_budget_capture.sh office0 abl_out/raster_prof_r0`
* identifiers: `abl_out/raster_prof_r0/git_commit.txt`, `binaries.sha256`,
  `gaussian_settings.used.yaml`, `orb_settings.used.yaml`, `run.log`.
* raw data: `abl_out/raster_prof_r0/run.log` (33 blocks of 9 stages),
  `gaussian_training_log.txt`, `total_mapping_time.txt`.

| question | answer |
|---|---|
| which kernel dominates? | **`BACKWARD::render`**, 4.92 ms median (3.16–5.38 over 33 blocks), 91.6 % of rasteriser time, ~71 % of a 6.9 ms real-scale training step |
| does the candidate preserve the original semantics? | no candidate was implemented; the required matching conventions are listed in §4 |
| micro-benchmark speed-up | not applicable (no candidate) |
| full training-path speed-up | not applicable |
| PSNR/SSIM/LPIPS/memory cost | not applicable; LPIPS unmeasured |

Normal online default behaviour is unchanged (the profiler is off unless the switch is set), and no
positive result was manufactured.
