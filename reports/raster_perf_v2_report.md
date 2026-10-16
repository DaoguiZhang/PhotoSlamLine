# raster-perf-v2 — warp-aggregated `BACKWARD::render` write-back: implemented, **FAILED correctness**, slower

Branch `feat/mapper-budget-v1`. This round was allowed to modify the CUDA backward kernel. A candidate
**was implemented and is runnable and switchable inside one binary** — and it **fails the correctness
gate**, so, by the rule fixed in advance (*"numerical failure must be fixed first, it must not enter
formal training or be called a speed-up"*), **no 100-step check and no 5000-step paired run were
performed and no acceleration is claimed.**

## 1. What was built (this is a real candidate, not an interface list)

`cuda_rasterizer/backward.cu`:

* `renderCUDA<C, AGG>` — the kernel is now templated on the backend; **both** instantiations live in
  the same binary.
* `warp_agg_add(addr, val)` — warp-aggregated write-back: the active lanes that act on the same
  Gaussian are summed in-register, and **one** lane issues **one** `atomicAdd` per gradient component.
* All 7 write-back sites route through `AGGADD(ptr, val)`: `dL_dcolors` (per channel),
  `dL_dmean2D.x/.y`, `dL_dconic2D.x/.y/.w`, `dL_dopacity`.
* Runtime switch `PHOTO_SLAM_RASTER_WARP_AGG=0|1`, and the backend actually used is printed once
  (`[RasterBackend] BACKWARD::render = ...`) so a run cannot be mislabelled.
* `PHOTO_SLAM_RASTER_AGG_SELFCHECK=1` — on **identical inputs, identical forward buffers and identical
  `dL_dpixels`**, launches reference, reference again (noise floor) and the candidate, and reports
  per-group `max_abs` / `max_rel` for `mean2D`, `conic2D`, `opacity`, `colors`, plus the three
  kernel times. The **reference** result is the one written to the caller's outputs, so enabling the
  self-check cannot change training semantics.

The aggregation precondition was checked in the code first, as instructed: in the inner loop
`global_id = collected_id[j]` is read from **shared memory**, so it is uniform across the block at a
given step, and the lanes writing a given component all write to the **same Gaussian** — different
Gaussians cannot be mixed. Lanes that left via `continue` (`power > 0`, `alpha < 1/255`) or had
reached `last_contributor` are not in `__activemask()` and are excluded.

## 2. Correctness result — **FAIL**

Run `PHOTO_SLAM_RASTER_AGG_SELFCHECK=1 bash scripts/run_budget_capture.sh office0 abl_out/agg_check_r1`,
at P = 143 515 Gaussians:

| | noise floor (reference vs reference) | candidate vs reference |
|---|---|---|
| `mean2D` max_abs | 1.16e-09 | **2.54e-03** |
| `conic2D` max_abs | 1.67e-05 | **7.20e+00** |
| `opacity` max_abs | 4.42e-09 | **1.05e-03** |
| `colors` max_abs | 5.82e-10 | **7.47e-04** |
| max relative | ≤ 4.5e-02 (on near-zero entries) | **up to 8.3e+04** |

The reference is reproducible to ~1e-9 absolute, so the difference is **not** float summation order:
the candidate is **wrong by 3–8 orders of magnitude** on every group. The maximum relative error is
inflated by near-zero entries, but the absolute errors on `conic2D` (7.2) and `mean2D` (2.5e-3) are of
the same order as the gradients themselves, i.e. contributions are being **mis-assigned or
mis-counted**, not merely re-ordered.

One concrete bug was found and fixed on the way (first run at P≈2000–2754, errors even larger): my
lane enumeration used `__fns(act, 0, i + 1)`, but `__fns` searches *after* `base`, so `i+1` skipped the
lowest active lane and the last iteration read an out-of-range lane. Changing it to `__fns(act, 0, i)`
plus a `lane < 0` guard did **not** fix the error, so at least one further cause remains.

Leading hypotheses for the residual error (not yet isolated):

1. the participating-lane set at the write-back point is not exactly the set of lanes holding a
   contribution for that Gaussian — e.g. lanes that `break` out of the inner loop early
   (`last_contributor` / transmittance cut-off) may still be counted in `__popc(act)` while their
   register `val` is stale, or vice versa;
2. the per-lane `val` expression is evaluated **before** the call for every active lane, but the
   control flow that produces `val` (the `T` recursion, `accum_rec`, `last_alpha`) is per-lane state
   that must be advanced by exactly the same set of lanes the reference would have used;
3. `__shfl_sync(act, val, lane)` with a non-contiguous `act` may need an `__syncwarp`-style
   convergence guarantee that `__activemask()` alone does not provide at every one of the 7 sites.

## 3. Speed result — the candidate is **slower**, not faster

From the same self-check (single kernel timings on identical inputs):

| P | reference | reference (repeat) | candidate |
|---|---|---|---|
| 2 003 | 0.430 ms | 0.225 ms | 2.119 ms |
| 2 754 | 0.331 ms | 0.294 ms | 3.185 ms |
| 143 515 | 8.387 ms | 10.954 ms | **62.419 ms** |

At real scale the candidate is **≈7× slower than the reference**. This is consistent with the cost
model of the reduction I chose: the enumeration performs `__popc(act)` shuffles **per site**, i.e. up
to 32 shuffles × 7 sites = up to 224 shuffle instructions per (pixel, Gaussian) contribution, against
the ~7 atomics it replaces. The shuffle cost dominates, so even a *correct* version of this particular
reduction would not be a speed-up at this occupancy.

**No claim is made about atomics or bandwidth.** No counter-level evidence was collected, and the
measurement shows the candidate loses for a reason (reduction instruction count) that is independent
of whether the atomics were contended.

## 4. What was deliberately not done

* No 100-step check and no 5000-step paired run — the candidate fails the correctness gate.
* No quality/PSNR/SSIM/LPIPS comparison — there is no valid candidate to compare.
* The candidate remains reachable only through `PHOTO_SLAM_RASTER_WARP_AGG=1`; **the reference is the
  default** and the forward path, loss formulas, precision, SH handling, opacity handling, culling
  rules and training schedule are untouched.
* The self-check run's `[RasterProf] BACKWARD::render` figure (≈59 ms) is an artifact of launching
  three kernels per step while self-checking; it is **not** a measurement of the candidate.

## 5. Answers to the five required questions

1. **Is the candidate correct?** **No.** Errors 3–8 orders of magnitude above a ~1e-9 reproducible
   noise floor, on every parameter group. Runner-up bug fixed, at least one cause remains.
2. **Is `BACKWARD::render` faster?** **No — ≈7× slower** at 143 515 Gaussians.
3. **Is the full training faster?** **Not measured** and not claimable; testing stopped at the
   correctness gate.
4. **Quality / memory cost?** Not measured; no valid candidate.
5. **Is the gain above the paired-run time fluctuation?** Not applicable — there is no gain.

## 6. Delivery and correction

* modified files: `cuda_rasterizer/backward.cu`, `cuda_rasterizer/backward.h`,
  `cuda_rasterizer/rasterizer_impl.cu` (the count is threaded into `BACKWARD::render` for the
  self-check).
* one command: `PHOTO_SLAM_RASTER_AGG_SELFCHECK=1 bash scripts/run_budget_capture.sh office0 abl_out/agg_check_r1`
  (add `PHOTO_SLAM_RASTER_WARP_AGG=1` to use the candidate as the authoritative backend).
* identifiers and raw data: `abl_out/agg_check_r0/run.log`, `abl_out/agg_check_r1/run.log`
  (`git_commit.txt`, `binaries.sha256`, configs alongside).
* **Correction to the previous report**, as requested: `D = 53 119` is only the difference of the net
  change `clone + split − prune`; **it cannot by itself establish that the arm with fewer injected
  points densified more, nor that it "over-compensated"**. That wording is withdrawn. No source audit
  is added for it in this round.
* Loop closing: not started, and it should not be started until a candidate shows a repeatable
  full-training gain with quality preserved.

## 7. Next concrete step

Isolate the aggregation on a **synthetic micro-case** (one 16×16 tile, a handful of Gaussians, known
contributions, `P` small enough to enumerate by hand) and compare lane-by-lane against the reference,
before any further work at real scale. That answers, with a minimal failing example, whether the fault
is the participating-lane set or the reduction itself; only then is it worth reconsidering the
reduction strategy — and given the measured ~7× regression, the next strategy should reduce
**instruction count per contribution**, not just the number of atomics.
