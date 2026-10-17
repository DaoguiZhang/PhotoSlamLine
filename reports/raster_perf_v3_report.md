# raster-perf-v3 — reduction correctness fixed; the enumeration-based candidate is **correct but ~8× slower**

Branch `feat/mapper-budget-v1`. Reference backend remains the default.

**Round outcome (one of the two outcomes the brief allowed): a correct implementation with no gain,
and terminating evidence for this route.** The brief's other outcome — correct and faster — was not
achieved.

## 一 The primitive: `__fns` measured, and two bugs found

A standalone device test (`fnsSelfTestKernel`, 32 threads, **no Gaussian rendering involved**) prints
`__fns(mask, base=0, offset)` for the required masks. Measured table (`0xffffffffffffffff` = invalid):

| mask | off0 | off1 | off2 |
|---|---|---|---|
| `FFFFFFFF` | 0 | 0 | 1 |
| `FFFFFFFE` (lane 0 clear) | **invalid** | 1 | 2 |
| `80000000` (only bit 31) | invalid | 31 | invalid |
| `00000001` (only lane 0) | 0 | 0 | invalid |
| `AAAAAAAA` (alternating, lane 0 clear) | invalid | 1 | 3 |
| `00050005` (lanes 0,2,16,18) | 0 | 0 | 2 |

**Measured semantics:** `offset = 0` only *tests* bit `base` (no search); for `offset = k ≥ 1` it
returns the **(k−1)-th set bit at or after `base`**; when there is no such bit it returns
`0xFFFFFFFF`.

**Correction of the previous report.** The v2 report claimed that `__fns(act, 0, i + 1)` "searched
after `base`", skipped the lowest active lane, and that changing it to `i` was the fix. **That was
wrong.** The measurement shows `i + 1` was the *correct* enumeration and changing it to `i` made
things **worse**: `offset = 0` only tests bit 0, so `i = 0` returned `invalid` for any mask without
lane 0 (then shuffled lane −1), and every subsequent `i` was shifted by one. The v2 explanation is
withdrawn and the enumeration is restored to `i + 1`, with the invalid return value explicitly tested
(`lane < 0` ⇒ no shuffle) and the value kept `unsigned`.

**Second bug, and it was the real integration fault.** The block is 16×16 (`BLOCK_X = BLOCK_Y = 16`),
so a warp spans **two tile rows** and the hardware lane is
`(threadIdx.y * BLOCK_X + threadIdx.x) & 31`. The candidate used `threadIdx.x & 31`, which **collides
between the two rows**: two threads per warp satisfied the leader test, so the aggregated sum was
written **twice**. That is precisely the check the brief asked for ("use the real hardware lane or the
full linear thread number, not just `threadIdx.x`"), and it was the dominant cause of the v2 failure.

Lane/warp layout confirmed: linear tid = `threadIdx.y * 16 + threadIdx.x`; warp *w* covers rows
2w and 2w+1 of the tile. Non-contributing lanes are excluded via `__activemask()`, which is the
contributing set here.

## 二–三 State of the implementation

* **A (reference)** — unchanged, default.
* **B (restructured control flow, plain `atomicAdd`)** — **not implemented.** It requires removing both
  `continue`s *and* the `!done` exit from the inner-loop condition so that all 32 lanes stay converged
  at the write-back point; that refactor was not done.
* **C (tree reduction on top of B)** — **not implemented.** The helper `warp_tree_sum()`, a standard
  5-stage `__shfl_down_sync(0xffffffff, …)` tree with the stated contract (full warp participates,
  non-contributors pass `0.0f`), is in the file but **unused**, because using it without B's
  convergence guarantee would be exactly the "sparse mask on a plain shuffle tree" mistake the brief
  forbids.

So the correct candidate that exists today is the **v2 kernel with the two bugs fixed** — i.e. the
per-lane enumeration, which is what the measured numbers below refer to.

## 四 Correctness: **PASS**

`PHOTO_SLAM_RASTER_AGG_SELFCHECK=1`, identical inputs / forward buffers / `dL_dpixels`, reference run
twice for the noise floor:

| P = 148 727 | noise floor (ref vs ref) | candidate vs reference |
|---|---|---|
| `mean2D` max_abs | 1.11e-09 | **1.68e-08** |
| `conic2D` max_abs | 1.34e-05 | **7.44e-05** |
| `opacity` max_abs | 6.52e-09 | **4.25e-08** |
| `colors` max_abs | 2.10e-09 | **1.49e-08** |
| max_rel | ≤ 1.2e-02 | ≤ 5.1e-02 |

The candidate's error is now the same order as the reference's own run-to-run reproducibility, i.e.
**float summation order only** — no missing, duplicated or mis-assigned contributions. All seven
write-back sites were aggregated at once, so the per-group table above already covers `mean2D`,
`conic2D`, `opacity` and `colors`; the downstream `xyz`/scale/rotation/SH gradients are produced from
these by `BACKWARD::preprocess`, whose inputs are now equal to the reference's to float precision.

Two further scales from the same run confirm it independently:

| | P = 159 452: noise floor | P = 159 452: candidate |
|---|---|---|
| `mean2D` max_abs | 9.02e-10 | **2.85e-09** |
| `conic2D` max_abs | 4.29e-06 | **3.43e-05** |
| `opacity` max_abs | 1.37e-09 | **9.33e-09** |
| `colors` max_abs | 4.37e-09 | **1.51e-08** |
| max_rel | ≤ 2.1e-02 | ≤ 3.6e-02 |

(same pattern for the P≈1 900 cases, e.g. `conic2D` 6.3e-06 against a 4.3e-06 noise floor).


Note the previous round's failure was **not** caused by the aggregation idea: it was caused by a
colliding lane id writing every aggregated sum twice, plus a mis-typed shuffle index.

## 五 Performance: correct, but **~8× slower** — route terminated

| P | reference | reference (repeat) | candidate |
|---|---|---|---|
| 148 727 | 10.01 ms | 9.70 ms | **81.56 ms** |
| 159 452 | 5.61 ms | 5.27 ms | **36.01 ms** |

Timed as three separate single-kernel launches on identical inputs (the self-check's own timing), **not**
as a total for the three-call sequence. Cold: the per-lane enumeration costs `__popc(act)` shuffles per
site, i.e. **up to 32 shuffles × 7 sites = up to 224 shuffle instructions per (pixel, Gaussian)
contribution**, against the ~7 atomics it replaces. The reduction mechanism, not the atomics, is what
costs. Register/spill data was not captured.

Per the pre-agreed stopping rule — *"if a correct C still shows no gain, stop this warp-aggregation
route, keep the code and the negative result, and choose another implementation from actual kernel
evidence"* — **this route is terminated as a performance candidate.** Code is retained behind
`PHOTO_SLAM_RASTER_WARP_AGG=1` (default off) and the result is negative on speed but positive on
correctness.

## 六 Training gate: **not entered**

The gate requires correct **and** a micro-benchmark gain. The candidate is correct but slower, so the
100-step check and the 5000-step pair were **not** run.

Two gate items recorded for whoever continues:

1. **`SELFCHECK` currently returns the reference result to the caller.** For a real candidate training
   run that substitution must be disabled so the candidate's gradients are the ones returned, and it
   must be confirmed that each backward executes exactly one backend.
2. The 5-stage tree in `warp_tree_sum()` is the obvious next cost-reduction step, but it **requires**
   the B refactor first (no `continue`, no `!done` exit in the inner loop) so that all 32 lanes are
   converged at the write-back point.

## 七 Delivery

* modified: `cuda_rasterizer/backward.cu` (primitive test, `warp_tree_sum` helper, corrected lane id
  and enumeration), plus the earlier v2 changes (`backward.h`, `rasterizer_impl.cu`).
* commands:
  * primitive + correctness + timing: `PHOTO_SLAM_RASTER_AGG_SELFCHECK=1 bash scripts/run_budget_capture.sh office0 abl_out/agg_fix`
  * candidate as the authoritative backend: add `PHOTO_SLAM_RASTER_WARP_AGG=1`
* identifiers / raw data: `abl_out/agg_fix/run.log` (`[FnsTest]`, `[AggSelfCheck]`),
  `abl_out/fns_test/run.log`; `git_commit.txt` and `binaries.sha256` alongside.
* default behaviour unchanged; the forward path, loss formulas, precision, SH, opacity handling,
  culling and training schedule are untouched.
* not done, as instructed: no loop closing, no new sampling, no structural loss, no large experiment
  matrix.

### Next step, from evidence

The measured cost is **instruction count in the reduction**, so the next attempt must cut shuffles per
contribution, not atomics: (1) do the B refactor; (2) use the 5-stage tree (5 shuffles per component
per warp) instead of `__popc` enumeration; (3) if that still does not win, aggregate **per warp over
the whole inner-loop body** rather than per site, or reconsider the tile organisation — rather than
tuning this candidate's parameters.
