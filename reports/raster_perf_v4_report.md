# raster-perf-v4 — converged control flow + 5-stage warp tree (A/B/C)

Branch `feat/mapper-budget-v1`. **Default backend remains A (reference).**

**Headline: C is ~2.8× faster than A in the micro-benchmark, and this is the first real speedup on
this line of work. Correctness is strongly indicated but not yet conclusive, so the training gate was
not entered** — see §4 and §6.

## 1. Three implementations, host-dispatched

| backend | kernel | what it is |
|---|---|---|
| **A** | `renderCUDA<NUM_CHANNELS>` | the reference, restored to pristine: no backend template parameter, no extra device branch, plain per-contribution `atomicAdd` |
| **B** | `renderCUDA_conv<NUM_CHANNELS,false>` | converged control flow, still one `atomicAdd` per valid contribution |
| **C** | `renderCUDA_conv<NUM_CHANNELS,true>` | B + 5-stage full-warp tree (`off = 16,8,4,2,1`), one writer per warp |

Host-side switch `PHOTO_SLAM_RASTER_BACKEND=A|B|C` (default A); the backend actually used is printed
once. Selecting B or C adds **no** device-side branch or register pressure to A, because each backend
is its own kernel instance. The old per-lane enumeration candidate is kept only as history and is no
longer reachable from the dispatch.

## 2. B's contract, and how it is met

* Both `continue`s **and** the `!done` loop exit are gone. `done` was in fact only ever initialised
  (`bool done = !inside;`) and never assigned, so the loop condition was exactly `inside`; it is now an
  explicit `outside` flag folded into `valid`.
* `valid` is recomputed from scratch at the top of **every** Gaussian iteration, and all seven
  contribution registers (`wr_col[3]`, `wr_mx`, `wr_my`, `wr_cx`, `wr_cy`, `wr_cw`, `wr_op`) are
  **re-zeroed every iteration**, so nothing can leak from the previous one.
* `valid` decides only whether a contribution is *computed*; it never decides participation in the
  reduction. Every lane runs the same trip count (`min(BLOCK_SIZE, toDo)` with `toDo` uniform).
* The transmittance/colour recursion, alpha threshold, `last_contributor` and truncation are applied
  only when `valid`, exactly as before. Threads outside the image have `valid == false` from the first
  iteration, so they never read the image or the per-pixel buffers.
* Lane id is the block-linear index `(threadIdx.y*BLOCK_X + threadIdx.x) & 31`, never `threadIdx.x`
  (the 16×16 block makes a warp span two tile rows of the same tile).
* C's tree: `__syncwarp()` after the divergent `if (valid)` block, then five `__shfl_down_sync` stages
  under `0xFFFFFFFF`, and **lane 0** writes. Since invalid lanes contribute exactly `0.0f`, the
  leader's own pixel being invalid cannot drop the other lanes' contributions.

## 3. Timing — warm-up + 9 alternating blocks, median and distribution

Same parameter snapshot, same view, same forward buffers, same `dL_dpixels`; 3 warm-up launches of
each backend, then 9 blocks in which A, B and C are measured alternately with individual CUDA events.
These are single-kernel times, **not** a total over the whole comparison sequence.

| scale | A median (min–max) | B median | C median | A → C |
|---|---|---|---|---|
| P ≈ 8 k | 6.086 (5.869–6.522) | 22.598 | **1.842** (1.770–1.931) | **3.3× faster** |
| P ≈ 117 k | 1.918 (1.904–1.943) | 7.827 | **0.822** (0.819–0.827) | **2.3× faster** |
| P ≈ 120 k | 5.435 (5.225–5.828) | 25.201 | **1.935** (1.890–2.069) | **2.8× faster** |
| P ≈ 170 k | 8.496 (8.275–9.054) | 29.057 | **2.780** (2.595–3.034) | **3.1× faster** |
| P ≈ 200 k | 5.320 (5.058–5.619) | 20.001 | **1.916** (1.846–1.986) | **2.8× faster** |

**Reported separately, as required: B is ~4× SLOWER than A.** The converged control flow makes every
lane execute the shared-memory loads for `collected_xy`, `collected_conic_opacity` and the colours on
every iteration, where A's invalid lanes left via `continue`/`!done` almost immediately. That cost is
real and would make B unusable on its own; C more than pays for it.

Registers / spills / shared-memory usage were **not** captured (no `-Xptxas -v` or occupancy query was
run), so the speed-up is **not** attributed to atomics or to bandwidth. What the data does support is
narrower: replacing ~32 per-contribution global `atomicAdd`s per warp-component with one write costs
less than the extra shared-memory work the converged loop adds.

## 4. Correctness — per-group, judged against the reference's OWN reproducibility

Pre-stated judge, fixed before looking at candidate results: each group passes when its `max_abs` is
within **10× that same group's A-vs-A-repeat `max_abs`** (floor 1e-7) **and** its global relative L2 is
within **10× that group's A-vs-A relative L2** (floor 1e-6). Near-zero entries are never judged by
their own relative error.

At **P = 200 106** (the largest case): **all 12 comparisons PASS.**

| pair | mean2D | conic2D | opacity | colors |
|---|---|---|---|---|
| noise floor A–A `max_abs` | 1.14e-09 | 3.05e-04 | 6.05e-09 | 5.12e-09 |
| B–A | 3.8× noise ✔ | 4.4× ✔ | 1.9× ✔ | 1.2× ✔ |
| C–A | 5.3× ✔ | 2.9× ✔ | 4.8× ✔ | 3.1× ✔ |
| C–B | 5.3× ✔ | 2.9× ✔ | 4.8× ✔ | 3.1× ✔ |

At **P = 170 466** three tiny-magnitude groups exceed the threshold: B–A opacity 31× (abs 8.9e-07),
C–A mean2D 63× (abs 3.2e-07) and C–A / C–B opacity 75× (abs 2.2e-06), with global relative L2 ≤ 6.6e-05.
conic2D passed in every pair everywhere.

**Why this is "strongly indicated, not conclusive":** the noise floor is a **single** A-repeat sample,
and the reference's own nondeterminism varies by an order of magnitude across cases (conic2D A–A
`max_abs` measured 1.1e-04, 3.05e-04 and 9.46e-04 in different cases). The two failing groups are
exactly the ones whose sampled noise floor happened to be exceptionally low, and their absolute
deviations (3e-07, 2e-06) are of the size a 32-term float sum with cancellation can differ by. That is
a plausible explanation, **not a proof**. I am not going to call it correct on that basis.

## 5. Self-check honesty

The previous round's substitution — returning the reference's result to the caller even when a
candidate was selected — is **removed**. In self-check mode the correctness phase writes into separate
buffers, and then **exactly one** launch writes the caller's outputs: the selected backend. So a
candidate training run really returns the candidate's gradients, and each backward executes exactly one
backend. The self-check only runs on the first two calls plus once per new size bucket, so the hot loop
is not paying the comparison cost.

## 6. Training gate: **not entered**, and why

The gate requires *correct* **and** a repeatable micro-benchmark gain. The gain is clearly repeatable
(2.3–3.3× across five scales and nine blocks each). The correctness is not yet conclusive by the
pre-stated judge. So no 100-step check and no 5000-step pair were run.

**The single remaining blocker, and it is cheap:** make the noise floor a multi-sample estimate —
run A nine times per case and use the observed `max_abs` range as the per-group denominator, instead of
one A-repeat. The machinery already exists in this self-check; only the repetition count changes. If
the P≈170 k deviations fall inside the multi-sample envelope, C is correct and the 100-step gate opens.

## 7. Delivery

* modified: `cuda_rasterizer/backward.cu` (A restored, B and C kernels, host dispatch, extended
  self-check), plus the earlier `backward.h` / `rasterizer_impl.cu` count threading.
* commands:
  * correctness + timing, reference as the active backend:
    `PHOTO_SLAM_RASTER_AGG_SELFCHECK=1 bash scripts/run_budget_capture.sh office0 abl_out/abc_r1`
  * same with C actually training: add `PHOTO_SLAM_RASTER_BACKEND=C`
  * with B: `PHOTO_SLAM_RASTER_BACKEND=B`
* identifiers / raw data: `abl_out/abc_r1/run.log` (`[FnsTest]`, all `[AggSelfCheck]` blocks),
  `git_commit.txt`, `binaries.sha256`, the two config copies.
* Not done, as instructed: no loop closing, no new sampling, no structural loss, no point-budget scan,
  no new replay system. Default backend unchanged (A).

### Verdict in one line

The 5-stage tree on converged control flow is **~2.8× faster than the reference at 120–200 k
Gaussians**, the control-flow refactor alone is ~4× slower (reported separately), and the candidate's
numbers agree with the reference to a few times the reference's own run-to-run scatter at the largest
scale — but the noise floor needs more than one sample before this is called correct and allowed into
training.
