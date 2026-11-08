# mapper-budget-v1 — 在线规模诊断与 inactive 补点接收预算

Branch `feat/mapper-budget-v1` in `/workspace/code/PSL-perf`, base `b5e835f8e2`
(`perf/mapper-min-cost` stays at that commit, so the previous state is untouched).
All new switches default **off**; normal online behaviour is unchanged.

**Scope actually delivered this round — read this first.** Items 一 and 二 are complete with
measured data. Item 三 is delivered **only** as a real-scale cost curve taken from the capture's own
logs; the kernel-level profile and the standalone snapshot loader are **not done**. Item 四 (the
R100/R50 budget) is **designed but not implemented**, and item 五 (the paired formal comparison) was
**not run**, because the budget variable has no effect until the replay driver exists (see section
四). Nothing in this round is claimed as a speed-up, and no duplicate was claimed to be found.

| item | status |
|---|---|
| 一 save the scene, correct the conclusions | **done** |
| 二 one office0 capture | **done** (with two defects listed in 二.5) |
| 三 real-scale profile | **partial**: cost curve done; kernel profile / snapshot loader not done |
| 四 R100 vs R50 budget | **not implemented** (design + blocking reason in section 四) |
| 五 paired formal comparison | **not run** |

Reproduce the capture (one command):

```bash
bash scripts/run_budget_capture.sh office0 abl_out/budget_capture_office0
```

---

## 一 Saved scene and corrected conclusions

* archive with full checksums: `archive/perf-stage3-audit/CHECKSUMS.sha256` (11 artifacts);
  binaries/libraries hashed in `binaries.sha256`, config in `cfg/`.
* previous branch `perf/mapper-min-cost` left at `b5e835f8e2`; work continues on
  `feat/mapper-budget-v1`.
* four conclusions corrected in `reports/perf_stage3_report.md` before any new measurement. The
  first one is now backed by data — see section 三.

## 二 The single office0 capture

20039 training iterations, 131.71 s billed mapping, 2000 frames, ~193 400 final Gaussians.
Artifacts in `abl_out/budget_capture_office0/`: `creation_audit_*`, `creation_log.bin` (6.39 MB),
`linepop_linepop.csv`, `linepop_lineinherit.csv`, `linesample_linesample.csv`, `snapshots/`.

### 2.1 Why the increment is only 3 — answered by data, not by a threshold

The line-sampling entry is reached **twice in 20039 steps**, and both times:

```
iteration,raw_points,candidates,rejected_voxel_occupied,accepted
5682,3,3,2,1
9925,3,3,2,1
```

The candidate pool itself is **3**. The voxel-occupancy guard rejected 2 of 3 and accepted 1.
So **relaxing the voxel threshold could have added at most 1 more Gaussian** — it is not the cause.
The cause is upstream: the entry that produces line-sample candidates is almost never reached
(`l_positions.size()/3 == 3` on both calls). Per instruction, **no threshold was relaxed** and this is
reported as an open path question, not as a bug that was patched.

### 2.2 The online line population — the +2 does not represent it

| stage | iteration | total | line | ordinary |
|---|---|---|---|---|
| first audited event | 1 | 2 775 | **207** | 2 568 |
| … (monotone decay) | 24 | — | 66 | — |
| | 24 | — | 48 | — |
| | 24 | — | 38 | — |
| | 7 | — | 31 | — |
| | 24 | — | 26 | — |
| snapshot (near-final) | 20 000 | 193 354 | **25** | 193 329 |

Full trajectory in `linepop_linepop.csv`. The online line population **starts at 207 and decays to
25**; the seeding entry contributed **+2**. The line-Gaussian population is therefore *not* the
seeding increment, exactly as the correction in section 一 requires.

### 2.3 Metadata inheritance after topology operations — checked, no defect found

`linepop_lineinherit.csv` records, on **every** creation/removal event (565 events), the alignment of
`is_line_`, the anchor `xyz_init_` and the direction `line_dir_w_` against `xyz_`, plus the number of
line rows whose direction has become zero:

* rows with any misalignment: **0 / 565**
* rows with zero-direction line rows: **0 / 565**

So `prune`, `clone` and `split` **do** inherit the line metadata correctly. No path or metadata error
was found, so nothing was fixed and nothing was renumbered — the instruction explicitly allows this
outcome.

### 2.4 Creation accounting — split-parent removal does **not** exist here

| entry | added | removed |
|---|---|---|
| `map_point_seed` + `inactive_geo_densify` (add) | 177 338 | 0 |
| `line_seed` (add) | 2 | 0 |
| `clone` | 18 105 | **0** |
| `split` | 171 970 | **0** |
| `prune` | 0 | **176 102** |

Measured: **`clone` and `split` only ever insert**. All 176 102 removals sit in the `prune` bucket,
whose mask is `opacity ∪ lazy-gradient ∪ oversized`. `densifyAndSplitWithLineAwareness` reads
`xyz_sel = getXYZ().index({selected_mask})` and appends; it never removes the parent. So in this
implementation there **is no split-parent deletion to separate out** — the correction resolves to a
code fact that is now stated instead of implied, and the earlier "~47 % of insertions are wasted"
reading is withdrawn rather than re-derived. Peak 195 241 → final 193 354; conservation holds
(initial 2 041 + insertions − removals = current).

Identity association: the per-source identity the user asked for is recorded only where it exists —
this build stores **no** `MapPointID` / `MapLineID` per Gaussian, and the inactive-geo depth sampler
must not be forced to carry one. That gap is stated, not filled by proximity.

### 2.5 Defects in this round's capture instrumentation

1. **Only one of the two requested snapshots fired.** `PHOTO_SLAM_SNAPSHOT_AT=9000,20000` produced
   `snapshots/model_step20000.pt` only (193 354 Gaussians, `default_sh=3`, 55 MB, full tensors).
   Verified cause: the online run executes **both** training loops — `trainForOneIterationErrorGuided()`
   (called at `src/gaussian_mapper_line.cpp:920`) and `trainForOneIteration()` (at
   `src/gaussian_mapper_line.cpp:761`/`805`) — and the snapshot trigger is only in the first. Fixing it
   needs the same hook in `trainForOneIteration()` plus a re-capture.
2. `creation_log.bin` records kind + iteration + xyz + rgb (+ direction for line seeds) but **not**
   the per-step training-view schedule, so a replay cannot yet reproduce the captured view order —
   only the recorded creation stream. Both are required before the paired replay is meaningful.

## 三 Real-scale cost curve (partial — no kernel profile)

Taken from the capture's own per-step log (`gaussian_training_log.txt`) — **no new code, no
re-run**, and explicitly **not** a kernel profile.

| Gaussian count bucket | steps | mean ms/step | median ms/step |
|---|---|---|---|
| 0 – 40 k | 1 404 | 4.687 | 4.0 |
| 40 – 80 k | 1 200 | 5.884 | 7.0 |
| 80 – 120 k | 1 410 | 5.877 | 7.0 |
| 120 – 160 k | 2 920 | 6.552 | 7.0 |
| 160 k + | 13 105 | **6.917** | 7.0 |

first 500 steps (n ≈ 4 418): 4.038 ms mean / 2.0 ms median; last 500 steps (n ≈ 193 354):
6.458 / 6.0. Total billed mapping 131.71 s over 20 039 steps = 6.57 ms/step.

**This is the correction the review asked for, now measured.** ~100× more Gaussians costs only
~1.5× the per-step time, and the per-step cost is already ~4.7 ms while the count is still under
40 k. The frozen pair's decomposition (0.23 ms render forward, 5.92 ms renderer backward at
1 954 Gaussians) therefore **cannot** be scaled by primitive count: a large part of the real per-step
cost is count-insensitive, and the rasteriser backward's share at 193 k is *not* the 88 % measured at
1 954. Any optimisation decision must be taken on a real-scale profile.

Not done, and required before section 四 can be judged: kernel-level attribution
(Nsight/kineto, short segments, per-operator self time without nested double counting), Gaussian–tile
intersection counts, resolution/SH/projection-scale tails at real scale, and a loader that reads
`model_step20000.pt` **without restarting SLAM**.

## 四 R100 vs R50 — design fixed, implementation blocked by one missing driver

The design is as specified: same persisted geometry input stream, shared initialisation, shared
schedule; both arms with all line structural losses off and identical ordinary/line seeding rules;
identical LR, view order, SH and pyramid schedule; normal clone/split/prune/opacity retained;
**variable topology** (this is a budget experiment, so the earlier fixed-count gating must *not* be
applied). R50 takes a deterministic, image-space-stratified half of the *same* candidate pool, with the
actual kept indices and their hashes saved, and no further ratio scan.

**Why it is not implemented yet.** The frozen replay never reaches
`increasePcdByKeyframeInactiveGeoDensify` — that entry is driven by the SLAM keyframe-processing
thread, which the replay does not run — so a budget applied inside it is currently a no-op. Three
pieces are needed, in this order:

1. a replay driver that calls the inactive-geo entry deterministically from the frozen keyframes at
   the recorded optimizer steps (variable topology, `frozen_fixed_count_=0`; the existing gates at the
   densify sites already permit clone/split/prune in that mode);
2. image-space stratification — `depth_cache_points_` carries XYZ only, so the per-keyframe 2D
   keypoint indices must be carried alongside the cache before an image-space half can be defined;
3. recording the per-step view schedule during capture (defect 2 above), so both arms replay the
   captured order rather than a re-generated one.

If candidates are pre-generated in the replay, the candidate-**generation** cost is omitted; that must
be reported as an omission and must not be counted as an optimisation gain.

## 五 Pre-registered acceptance (fixed before any R50 measurement)

Recorded now, so it cannot be relaxed after seeing results. R50 counts as a candidate only if, on the
paired replay over the common input stream with the profiler off:

* **quality** — `|ΔPSNR| ≤ 0.05 dB`, `|ΔSSIM| ≤ 0.0010`, `|ΔLPIPS| ≤ 0.0020` versus R100, on the
  shared evaluation views from the same checkpoint, with no NaN and no non-finite gradients;
  (0.05 dB is just above the previously measured same-scene OFF/OFF drift of ≤ 0.043 dB at 5000 steps)
* **cost** — mapping wall clock reduced by **≥ 3 %**, with the actual optimizer-update count reported
  alongside so a shorter run cannot masquerade as a faster one;
* **no compensation claim** — if R50 starts with fewer points but clone/split densify them back and the
  total time does not fall, that is reported as "the budget was offset by the other entries".

Also recorded for the formal run: mapping wall clock and update count, creation/densify/prune time,
Gaussian count versus step with peak and final, whether clone/split compensated, PSNR/SSIM/LPIPS and
memory. No three-seed or multi-scene expansion.

## 六 Deliverables and what must not be conflated

Modified files: `include/gaussian_model_line.h`, `src/gaussian_model_line.cpp`
(creation-batch log, full snapshot save/load, line-population + inheritance audit, capture hooks),
`include/gaussian_mapper_line.h`, `src/gaussian_mapper_line.cpp` (snapshot trigger, line-sample
audit, `maybeSnapshotAndAudit`, `lineSampleAuditLog`), `scripts/run_budget_capture.sh`,
`reports/perf_stage3_report.md` (corrections), this report.

Input identifiers: commit recorded in `abl_out/budget_capture_office0/git_commit.txt`, binaries in
`binaries.sha256`, config in `gaussian_settings.used.yaml` / `orb_settings.used.yaml`, sequence
`office0`, dataset `/workspace/code/SEGS-SLAM/datasets/replica/office0` (2000 frames).

Raw metrics: `creation_audit_events.csv` + `_summary.txt`, `linepop_linepop.csv`,
`linepop_lineinherit.csv`, `linesample_linesample.csv`, `creation_log.bin`,
`snapshots/model_step20000.pt` + `state_step20000.txt`, `gaussian_training_log.txt`.

The four things that must stay distinct, and where this round stands on each:

1. **real-scale performance diagnosis** — section 三 (cost curve only; kernel profile missing);
2. **point-budget reduction strategy** — section 四 (designed, not implemented, not measured);
3. **identity or spatial dedup** — **not attempted**; no `MapPointID`/`MapLineID` exists per Gaussian
   and the voxel guard covers only the line-sampling path (section 二.4);
4. **full online SLAM acceleration** — **not attempted**, and not implied by anything here.

This round's budget experiment does not depend on first proving duplication, and this round does not
claim to have found any. Kept limitations: the initialisation colour source still cannot be fully
traced; the front end is not reproducible across processes, so the capture's recorded stream is the
unit of reuse (item 四.3 is what makes replay from it valid); the capture is a single run whose wall
clock is not a speed result.
