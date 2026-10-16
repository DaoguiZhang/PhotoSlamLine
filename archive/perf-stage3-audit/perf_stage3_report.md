# perf-stage3-audit — 公共训练开销定位与在线重复播种审计

Baseline commit: **`be783b04e2`** (branch `perf/mapper-min-cost`, worktree `/workspace/code/PSL-perf`).
Both audits required this round were completed **before** any optimisation work, as instructed.

Deliverables (all re-runnable, no GPU needed to summarise):

```bash
# 一: density-subset reconstruction, hashes, nesting
python3 scripts/summarize_creation_audit.py --subset-run q_acc100 q_formal_r0
# 二: online creation-source audit (+ static duplicate-guard check)
bash    scripts/run_creation_audit.sh office0 abl_out/online_audit_r0
python3 scripts/summarize_creation_audit.py --creation-dir abl_out/online_audit_r0
# 三: backward decomposition
W=/workspace/code/PSL-perf CONDS=B EVAL_STEPS=100,5000 PHOTO_SLAM_FGM_BWD_DECOMP=1 \
  bash scripts/run_frozen_abl.sh bwd_decomp_r0 5000 office0
python3 scripts/summarize_creation_audit.py --decomp-glob 'abl_out/bwd_decomp_r0/frozen/B_bwd_decomp_step*.csv'
```

New switches (all **default off**, normal online behaviour unchanged):

| switch | effect |
|---|---|
| `PHOTO_SLAM_CREATION_AUDIT=<prefix>` | host-only creation audit → `<prefix>_events.csv`, `<prefix>_summary.txt` |
| `PHOTO_SLAM_FGM_BWD_DECOMP=1` | backward decomposition at the frozen eval points |

---

## 一 Density-subset reconciliation

The Stage-2 report quoted `55 ⊂ 99 ⊂ 166` line Gaussians over `43 groups`. **That row is stale.**
It belongs to an *earlier acceptance run of the same experiment whose frozen snapshot had 166 line
Gaussians and 43 direction groups*; that output directory no longer exists. It is not the number of
either run actually reported, and no result depends on it.

The authoritative figures, from the raw records of the two runs that **are** reported:

| run | role | checkpoint (total / line) | Q25 kept | Q50 kept | groups | stride |
|---|---|---|---|---|---|---|
| `q_acc100` | acceptance, 100 steps | 2071 / **174** | 57 | 105 | 47 | 4 / 2 |
| `q_formal_r0` | formal, 5000 steps | 2022 / **173** | 59 | 103 | 46 | 4 / 2 |

Why the same algorithm gives different counts: the kept count is `Σ_groups ceil(size_g / stride)`, so it
depends on how many line samples the frozen snapshot happens to contain. Reconstructed from
`q_acc100`'s own checkpoint, that formula reproduces **57** and **105** exactly; rescaling the same
groups to 166 lines gives 54 / 101, and a snapshot with a different set of line directions also yields a
different group count (43). A single collapsed direction group — the pre-fix failure mode — would have
given 44 / 87, so it does **not** explain the stale row.

Which run / which initialisation stage each number belongs to:

* the **checkpoint** stage (`frozen/init_checkpoint.pt`) contains 2071 / 2022 Gaussians, of which
  174 / 173 are line Gaussians;
* `Q25`/`Q50` counts are the **post-subset** stage of the same run (`keepSubset` applied once per
  condition, after `restoreInitCheckpoint`);
* `q_acc100` (acceptance, 100 steps) and `q_formal_r0` (formal, 5000 steps) are **separate processes
  with separate frozen snapshots** of `office0` (`map_points` 1897 vs 1849, `line_samples` 174 vs 173,
  `cameras_extent` 0.279 vs 0.238), so cross-run differences are expected.

Verification performed on the raw records (`reports/perf_stage3_subset_audit.txt`):

* reconstructed kept **index sets** per condition, with **SHA-256 of each set** —
  `q_acc100`: Q25 `c771a4df01a1bffa`, Q50 `bfcdecae0b15d085`, Q100 `ef00dcaf62745339`;
  `q_formal_r0`: Q25 `406aac99d44f3acb`, Q50 `df87423ea8695a9a`, Q100 `2cddc7badfd4094e`;
* **nesting verified empirically from the indices** (not from counts):
  `Q0 ⊆ Q25 ⊆ Q50 ⊆ Q100` holds in both runs;
* every reconstructed `(gaussians_line, gaussians_init)` pair matches the recorded
  `summary_*.csv` **exactly** in both runs;
* each run has a **single** `nonline_fp` across its four density conditions
  (`d785db2ad0efbcef_nonlin1897`, `5bdc457a932642ce_nonlin1849`) → the ordinary Gaussians are
  byte-identical across conditions **within** a run.

**Does the discrepancy affect the experimental conditions? No.** Every condition inside a run shares one
snapshot and one checkpoint; the counts are internally consistent; nesting holds; the ordinary
Gaussians are identical. The stale row came from a different run and is used nowhere.

Three mandated wording corrections, applied to `reports/perf_stage2_report.md`:

1. The density experiment only supports: *within the current snapshot and count range, no stable
   speed-up was observed*. It **cannot exclude** an effect of the total Gaussian count.
2. World-direction grouping is **not** a substitute for `MapLine` association, and **cannot guarantee
   that every line survives**.
3. These results **cannot be extrapolated** to whether the complete online map is redundant.

---

## 二 Online creation-source audit

Method. One representative **full online run** through the normal entry point
(`bin/replica_rgbd`, 2000 frames of `office0`), front end / keyframe policy / densify schedule
untouched. The line **structural** loss is switched off for this run
(`PHOTO_SLAM_LINE_LOSS_EVERY=0`) so the audit characterises the common pipeline rather than the
structural ablation; the shipped config is unmodified. The audit counts only at **CPU creation entry
points** and reads only host-side tensor metadata (`tensor.size(0)`), so it introduces **no per-step
device synchronisation** — confirmed by the measured CPU-dispatch share in section 三 (≈ 0 ms).

Run: 2000 frames, 112 keyframes, 20446 mapping steps, mapping time 128.2 s (wall 149.1 s).

### Creation / removal by entry point

| reason (entry) | site | events | added | removed | share of insertions |
|---|---|---|---|---|---|
| `map_point_seed` (add) | `increasePcd(float/tensor)` | 102 | 43 905 | — | 11.79 % |
| `line_seed` (add) | `increasePcd(vector<Point3D>)` | 2 | **3** | — | 0.00 % |
| `inactive_geo_densify` (add) | `increasePcdByKeyframeInactiveGeoDensify` → `increasePcd(float)` | 10 | 133 575 | — | **35.86 %** |
| `densify_prune` (clone) | `densifyAndCloneWithLineAwareness` | 149 | 18 901 | — | 5.07 % |
| `densify_prune` (split) | `densifyAndSplitWithLineAwareness` | 149 | 176 142 | — | **47.28 %** |
| `densify_prune` (prune) | `prunePointsWithLineAwareness` | 149 | — | 177 205 | — |

Gaussian count vs progress: **initial 1 911** (from `createFromPcd`, before the first audit event) →
**final 197 232**, **peak 204 477**.

Conservation check (the counting basis for add / delete / split is unified so the total is explained):

```
initial 1911 + 43905 + 3 + 133575 + 18901 + 176142 - 177205 = 197 232   (= current_total) ✓
```

Basis, stated explicitly to avoid double counting:

* a **clone** or **split** event counts *inserted rows only*; the parent row is counted once, in
  `prune_removed`, and only if the path actually removed it;
* ordinary pruning and the removal of a superseded parent row both land in the same
  `prune_removed` bucket and are therefore never counted twice;
* `reset_opacity` changes no count and is recorded only for traceability.

Interpretation. Over 47 % of everything ever inserted comes from **split**, and **36 %** from the
**inactive-geometry depth densification** path; ordinary triangulated-map-point seeding is only
~12 %. Roughly **47 % of all inserted Gaussians are later pruned** (177 205 / 374 526), and the final
count is 96.5 % of peak. Online **line seeding contributed 3 Gaussians in this run** — essentially
nothing — while the frozen A/B harness seeds 168–174 line samples from its frozen keyframes. **The
frozen harness's line population is therefore not representative of the line population an online run
builds**; that is a property of the harness, not of the two conditions compared inside it (both see
the same seeds).

### Repeated seeding of the same source — what can and cannot be decided

Static check of the guards that exist (`scripts/summarize_creation_audit.py --creation-dir …`
prints this section):

| guard | present? | scope |
|---|---|---|
| `point_ids_` used as a dedup key | **No** | the field is only maintained (`arange` / boolean filter / `cat` with fresh ids); it is never *read* to reject a re-seed |
| voxel occupancy queried before seeding | **Yes, line path only** | `scene_->isVoxelOccupied(...)` appears only in the two line-sampling loops |
| voxel grid written | **Yes, line path only** | `scene_->addPointToVoxel(...)` is called only next to that check |

Consequences:

* **Ordinary MapPoint seeding and inactive-geo densification have no occupancy or identity check at
  all.** Nothing stops the same triangulated map point, or the same surface patch seen by two
  neighbouring keyframes, from being seeded twice.
* The line path's guard is a **spatial-proximity substitute**, not identity: `isVoxelOccupied` tests
  the voxel cell of the sample position. Per the instruction, this is **not** treated as equivalent to
  identity association, and the gap is recorded rather than papered over.
* **Identity association is not available to decide duplicates.** No `MapPointID` / `MapLineID` is
  stored per Gaussian, so "the same source re-seeded" cannot be decided by identity, and direction
  similarity / spatial proximity cannot stand in for it.
* What the events *do* show: each `inactive_geo_densify` event is once-per-keyframe (guarded by
  `done_inactive_geo_densify_`), so there is **no evidence of the same keyframe's depth cache being
  flushed twice**, and the split/clone events are the legitimate densification path.
* **Not claimed:** that all same-source descendants are erroneous duplicates. Descendants from a
  legitimate clone/split, a rebuild after deleting an old representation, and new line-segment
  coverage are all expected. The audit records the entry point and the reason label; it does not, and
  cannot yet, assert identity-level duplication.

This run exists to **audit creation sources**; its single-run wall clock is **not** a formal speed-up
result (and the audit was off in every timing comparison).

---

## 三 Backward decomposition

Method. One fixed input; same parameter snapshot, view, resolution, SH degree and mask for every
stage. Stages are measured as **separate, non-nested blocks** (warm-up 20, 3 independent blocks of 50
iterations, minimum reported), so nothing is double counted. Two states are covered: an **early**
state at 100 completed optimizer steps and a **late** state at 5000.

Measurement boundaries, stated:

* **A — image loss.** The weighted `(1-lam)*L1 + lam*(1-SSIM)` is evaluated on a **diagnostic leaf**
  tensor holding the rendered image; the backward step measures `dL/dimage` — the incoming gradient the
  renderer must consume. `|dL/dimage|_2 = 1.518e-3`.
* **B — renderer backward.** The render graph is rebuilt from the same state and fed A's image
  gradient directly (`rendered.backward(grad_image)`); the image loss is **not** recomputed.
* **C — public constraints.** anisotropy + position/coherence + eccentricity + orientation, forward
  and backward, measured independently.
* **D — original full training step.** The original full graph is the reference, plus an
  `optimizer.step()` measured on cloned leaves so the training optimizer's state is untouched.
* The **mask** and the renderer's internal clamp/filtering are included in every stage; nothing is
  dropped from the pre-processing. `detach()` appears **only** in this micro-benchmark, never in real
  training.
* Host wall clock and CUDA-event device time are reported side by side; the gap is the CPU
  dispatch/synchronisation share. No `optimizer.step()` is performed on real parameters, no parameter
  or schedule is changed, and the graph is rebuilt with zeroed grads for every measurement.

### Stage times (ms per iteration, min of 3 independent blocks)

| stage | step 100 host / dev | step 5000 host / dev |
|---|---|---|
| A image loss forward | 0.793 / 0.793 | 0.793 / 0.793 |
| A image loss forward+backward | 1.419 / 1.416 | 1.402 / 1.402 |
| **A image-loss backward (derived)** | **0.627 / 0.623** | **0.610 / 0.610** |
| B renderer forward | 0.226 / 0.226 | 0.316 / 0.317 |
| B renderer forward+backward | 6.142 / 6.143 | 6.639 / 6.639 |
| **B renderer backward only (derived)** | **5.916 / 5.916** | **6.322 / 6.322** |
| C constraints forward | 0.211 / 0.211 | 0.211 / 0.211 |
| C constraints forward+backward | 1.506 / 1.506 | 1.420 / 1.420 |
| **C constraints backward (derived)** | **1.295 / 1.295** | **1.210 / 1.210** |
| D full forward | 1.125 / 1.125 | 1.206 / 1.206 |
| D full forward+backward | 6.917 / 6.917 | 8.421 / 8.421 |
| **D full-graph backward (derived)** | **5.792 / 5.792** | **7.215 / 7.215** |
| **D CPU-dispatch share** (host − device) | **−0.000092** | **+0.000007** |
| E `optimizer.step()` | 0.094 / 0.094 | 0.094 / 0.094 |

### Numerical verification of the split

Pre-stated tolerance: global relative L2 ≤ **1e-3** and per-tensor max relative L2 ≤ **1e-3**
(float32 accumulation over ~2·10⁵ parameters; the two paths reduce in a different order, so exact
bitwise equality is not expected here — unlike the algebraic-identity checks in `runEquivCheck`, whose
expectation is 0).

| | step 100 | step 5000 |
|---|---|---|
| global relative L2 of parameter gradients, `(A→B)+C` vs full path | **3.47e-06** | **3.40e-05** |
| worst single tensor | 4.59e-06 | 8.82e-05 |
| verdict | **PASS** | **PASS** |

So the split is numerically faithful to the original full path, and the attribution below is
legitimate.

### What the decomposition says (operator/stage level, not "backward is 80 %")

* **The rasteriser backward dominates.** At 5000 steps it is 6.32 ms of an 8.42 ms full
  forward+backward (75 % of the step, 88 % of the backward); at 100 steps 5.92 of 6.92 ms (86 % of the
  backward). Its growth between the two states (5.92 → 6.32 ms) tracks the growth of the render itself
  (0.226 → 0.316 ms), i.e. it is driven by the model size (1 954 Gaussians here, and ~197 000 in the
  full online run of section 二) and the rasterised workload.
* **The image loss is not the problem.** Its backward is 0.61–0.63 ms (~8 % of the backward) and its
  forward is *identical* at both states (0.793 ms), because the image size is fixed. The hypothesis
  "the SSIM window and the fixed GT statistics are recomputed every step and that is the cost" is
  **not supported** by this measurement: it would have to move ~10× to matter.
* **The public constraints cost 1.21–1.30 ms backward** (~17 %), i.e. real but secondary.
* **Host waiting is not the problem.** The CPU-dispatch share of a full iteration is
  **≤ 1e-4 ms** — the host time equals the device time to within measurement noise. The profiler-blind
  host-stall hypothesis is therefore **falsified at this level**, and no unnecessary `sync()` or
  duplicate allocation was located.
* `optimizer.step()` is **0.094 ms**, negligible.

Caveat on the two states: both are rendered at the keyframe's native resolution (the renderer is
invoked with the keyframe's `H`/`W`), so "early" and "late" differ in the **parameter state and SH
degree** (`default_sh` 0 → 3), not in the render resolution. That is deliberate — it keeps the two
columns comparable — but it means this pair is an early/late *state* pair, not a
reduced/full-resolution pair. The pyramid counter shows 3560 → 720 remaining sub-level uses, i.e. the
5000-step state is still occasionally at a sub-level.

---

## 四 Which single optimisation, and why none was selected this round

Candidate selection was to be driven by evidence, and only **one** change implemented.

1. *Image loss dominates* → **no.** 0.63 ms of ~8 ms, and flat across states.
2. *Host waiting dominates* → **no.** CPU-dispatch share ≈ 0 ms; no unnecessary sync located.
3. *Clear same-ID redundant seeding* → **no such evidence exists yet.** The audit shows a large
   `inactive_geo_densify` + `split` volume, but there is **no identity association** stored per
   Gaussian, so "the same source was seeded twice" cannot currently be demonstrated; each densify
   flush is once-per-keyframe. Implementing a de-dup on the strength of a volume number alone would be
   a guess, not an evidence-driven change, and it could silently remove legitimate new coverage.
4. *Rasterisation backward dominates* → **yes, but no clear small change.** The evidence is the stage
   table plus the model sizes (1 954 Gaussians in the frozen pair, ~197 000 online). No specific small,
   semantics-preserving edit was identified. Per instruction, a large acceleration library was **not**
   ported and the CUDA rasteriser was **not** rewritten on this evidence alone.

**Conclusion: no equivalence-preserving candidate was selected, and nothing was modified for
speed.** The dominant bottleneck is the **rasteriser backward at the model's actual online size**, and
the honest statement is that this round produced an audit, not a speed-up.

Next concrete task, in priority order:

1. **Kernel-level profile of the rasteriser backward** at the *online* Gaussian count (~197 000, not
   the 1 954-Gaussian frozen pair), with input sizes, to find whether the cost is a specific kernel
   (e.g. the per-tile backward accumulation) or the sheer number of primitives. This is the missing
   evidence level: the present decomposition credits the stage, not the kernel.
2. **Identity tracing for duplicate seeding**, as a separate, explicitly non-equivalent audit: store
   `MapPointID` / `MapLineID` (or a stable per-source key) on each Gaussian and re-run section 二's
   counting so that `inactive_geo_densify` and split descendants can be attributed to their source.
   Only after that can a de-dup be proposed — and then behind an independent switch that preserves
   legitimate new coverage.
3. Only once (1) or (2) yields a specific, semantics-preserving change: numerical + ~100-step state
   check, then **one** paired 5000-step comparison, profiler off for formal timing.

Not doing, and not claimed: no further line-density scanning, no new structural loss, no change to the
loss definition / schedule / precision, and no simultaneous modification of SSIM, sampling, de-dup and
rasterisation.

---

## 五 Verification, deliverables and limitations

Verification performed this round:

* density subsets: reconstructed index sets, hashes, empirical nesting, exact count match (§一);
* creation audit: **host-only** counters — no per-step device sync was added, and the independently
  measured CPU-dispatch share of a training iteration is ≈ 0 ms (§三);
* creation audit conservation: insertions − removals + initial = current total, exactly (§二);
* backward split: parameter gradients match the original full path to 3.5e-06 (step 100) and
  3.4e-05 (step 5000) global relative L2, both **PASS** against the pre-stated 1e-3 tolerance (§三);
* frozen-run state integrity intact: `params_unchanged=1`, `iteration_unchanged=1`,
  `params_finite=1`, `count_violation=0` for the decomposition run; the benchmark restores
  `max_radii2D_` / `debug_hit_count_` / `debug_alpha_accum_` and zeroes grads on the way out.

Deliverables:

1. **Online Gaussian creation-source curves and duplicate-seeding evidence** — §二: per-entry
   added/removed counts, total/peak/final vs progress, conservation identity, and the explicit guard
   gap.
2. **Verifiable cost decomposition of the common backward** — §三: stage table, host-vs-device split,
   numerical verification against the full path.
3. **A modification with correctness and real gain** — none: no candidate met the evidence bar (§四).
4. **Dominant bottleneck and next task** — §四: rasteriser backward at online model size; next task is
   its kernel-level profile, then identity tracing.
5. **One re-runnable audit-and-summarise command** — the three commands at the top of this document.

Limitations kept, including the pre-existing ones:

* the **initialisation colour sampling source cannot be fully traced**: `createFromPcd` takes
  `Point3D::color_` and no per-point source keyframe is recorded;
* **no `MapPointID` / `MapLineID` is stored per Gaussian**, so identity-level duplication is not
  decidable, and direction similarity / voxel proximity must not be substituted for it (§二);
* the frozen harness's line population (168–174 seeds) is **not** representative of what an online run
  builds (3 line Gaussians), so conclusions about line Gaussians drawn from the frozen harness do not
  transfer to the online map;
* the §三 state pair varies parameter state and SH degree, not render resolution;
* the creation audit is a **single run** and its wall clock is not a timing result;
* the SLAM front end is not reproducible across processes (different snapshots per run), so only
  within-process conditions are a valid control.
