# room1 model-side follow-up (r2): extreme scales and frozen-render counterfactual

## Boundary of the evidence

- Offline rescoring is verified; the r0 fixed-12 table is not changed.
- The r1 continuation was a different evaluation set and a different terminal
  state. Its `+1.93 dB` gain cannot cancel the r0 `1.75 dB` end-gap; it is only
  evidence of remaining fit headroom on its own fixed training views.
- `frame000000` has zero camera motion yet still loses 3.61 dB (A) / 10.50 dB
  (C) from 1000 to 1500. That excludes the frame's own pose change as a direct
  explanation, but does not exclude indirect effects of other training cameras.
- No online rerun is made here; no loss, densify threshold or production
  parameter is changed.

## 1. Scale convention and class statistics

`scaling_` is log-scale; every number below is
`exp(scaling_)`, in the same unit as `xyz_` and camera translations (Replica
scene units; not independently converted to metres). The three axes are
reported separately, not just their maximum.

`scale_stats.csv`:

| side / checkpoint | class | n | max-axis p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|
| A 1000 | line | 925 | 0.0380 | 0.1155 | 0.3568 | 1.647 |
| A 1000 | ordinary | 258859 | 0.00631 | 0.0592 | 0.1467 | 2.287 |
| A 1500 | line | 977 | 0.0328 | 0.0981 | 0.1664 | 0.775 |
| A 1500 | ordinary | 390710 | 0.00483 | 0.0488 | 0.1257 | 1.404 |
| C 1000 | line | 1313 | 0.0326 | 0.0847 | 0.2637 | 1.720 |
| C 1000 | ordinary | 269922 | 0.00633 | 0.0609 | 0.1569 | **12.496** |
| C 1500 | line | 1844 | 0.0282 | 0.0658 | 0.1703 | 1.652 |
| C 1500 | ordinary | 386207 | 0.00544 | 0.0557 | 0.1455 | **11.888** |

The C ~12 maximum is an **ordinary Gaussian**, not a line Gaussian. At 1000 it
is row 23377 (`point_id=86132`) with axes `[1.345, 0.896, 12.496]`, ratio `14.0`,
opacity `1.0`, ref KF `0`; at 1500 it is row 19260, same point id, axes
`[1.119, 0.809, 11.888]`. This is one-axis expansion, not the line
`parallel/transverse` pattern. The largest line Gaussian is only about `1.72`
(C 1000) / `1.65` (C 1500).

`line_eccentricity.csv` confirms the line scale is longitudinally long and
transversely thin:

- C 1500: `s_parallel` p01/p50/p95 = `0.00745 / 0.02816 / 0.06583`;
  `s_perp` p01/p50/p95 = `1.30e-5 / 4.61e-5 / 1.09e-4`;
- eccentricity `s_perp/s_parallel`: p50 `0.00163`, p99 `0.00198`, max `0.00230`;
- `n(s_parallel < 1e-5) = 0`; `n(s_perp > s_parallel) = 0`.

The code still contains the original unbounded form in both
`computeLineShapeConstraintRef` and `computeLineShapeConstraint`:

```text
eccentricity_loss = sum(mean(s_perp axes) / (s_parallel + 1e-6))
lambda_ecc = 0.002
```

It is mathematically unbounded as `s_parallel -> 0`. The original term still
acts on every line Gaussian, but the current snapshots and the fixed-anomaly
counterfactual do **not support it as the main cause** of this round's 1500
drop: no line row is near the singular limit, no line row has transverse
expansion, and zeroing the selected line rows is essentially a no-op.

## 2. Visibility of the extreme rows

`anomaly_rows.csv` contains, for each fixed anomaly row, the ref KF and
center-projection visibility over all checkpoint cameras using the saved mask.
For the top-20 set:

| side / checkpoint | line visible / 20 | ordinary visible / 20 |
|---|---:|---:|
| A 1000 | 13 | 4 |
| A 1500 | 12 | 6 |
| C 1000 | 8 | 6 |
| C 1500 | 10 | 6 |

The C maximum ordinary row at 1000/1500 has `ref_kf=0`, `visible_kfs=0`; its
ref-KF center projects outside the image. This is a center-only visibility
test; a large footprint could still touch the image, so “not visible” here is
not a contribution proof. It does mean the maximum scale cannot be equated with
a harmful pixel contribution.

## 3. Frozen-render opacity counterfactual

Fixed rule, written before rendering (`anomaly_rule.txt`):

> top 20 rows by `max(exp(scaling_))` within each class, per snapshot,
> deterministic tie by ascending row index.

No PSNR was used to choose rows. For each variant, a fresh snapshot load was
used; only `opacity_` of the selected rows was set to logit `-100` (sigmoid
~0). Camera, SH, background, mask, loss and all other model parameters were
unchanged. Full-scene rendering was used; no training. Diff images are
`(0.5 + 2*(original - variant))` for every rendered view.

`cf_summary.csv` (native GS PSNR, dB):

| side / checkpoint | class | original full | variant full | full Δ | original fixed-12 | variant fixed-12 | fixed Δ | frame0 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A 1000 | line | 31.2609 | 31.2606 | −0.00025 | 36.0691 | 36.0688 | −0.00029 | −0.00021 |
| A 1000 | ordinary | 31.2609 | 31.0779 | −0.183 | 36.0691 | 35.7671 | **−0.302** | −0.119 |
| A 1500 | line | 29.2335 | 29.2335 | −0.0000004 | 29.8121 | 29.8121 | −0.00001 | −0.0001 |
| A 1500 | ordinary | 29.2335 | 29.1877 | −0.0458 | 29.8121 | 29.7179 | **−0.094** | −0.086 |
| C 1000 | line | 30.8298 | 30.8302 | +0.00041 | 36.9497 | 36.9485 | −0.0011 | −0.0012 |
| C 1000 | ordinary | 30.8298 | 30.4385 | **−0.391** | 36.9497 | 35.8640 | **−1.086** | −0.965 |
| C 1500 | line | 26.0758 | 26.0759 | +0.00004 | 26.6103 | 26.6103 | +0.00006 | +0.00001 |
| C 1500 | ordinary | 26.0758 | 26.0722 | −0.00367 | 26.6103 | 26.6017 | **−0.0086** | +0.160 |

Interpretation:

- Zeroing extreme **line** rows is essentially a no-op. The line rows are not
  the observed cause.
- Zeroing extreme **ordinary** rows usually **lowers** full and fixed-12 PSNR;
  they are contributing positively, not damaging the render. The single
  exception is C 1500 frame0, which improves by 0.160 dB, but the same C 1500
  fixed-12 set is 0.0086 dB worse, and C 1000 frame0 is 0.965 dB worse.
- The C maximum ordinary row is not even center-visible in its ref KF. The
  observed max scale therefore does not support “the ~12 outlier explains the
  C drop”.
- This is a counterfactual opacity intervention, not a Shapley/contribution
  estimate and not a production pruning rule. The top-20 scan is not expanded
  in the next round.

## 4. Result against the decision rules

The abnormal-line branch is **not supported**:

- line zeroing has no material effect;
- no line row has parallel collapse or transverse expansion;
- the original unbounded eccentricity term still acts on every line Gaussian,
  but these snapshots and the fixed anomaly-row counterfactual do not support
  it as the main cause of the observed drop;
- no loss change is attempted.

The abnormal-ordinary branch is also **not supported as harm**: removing the
fixed extreme rows reduces quality. Therefore the 1500 model-side drop is not
explained by the top-20 extreme scale rows alone.

Per the pre-registered rule, the next branch is **late training allocation**,
not loss or densify changes. Use the implemented `PHOTO_SLAM_STAGE_COUNTERS`
path in the next necessary online run to record per-resolution/SH updates,
training pixels, per-KF usage and stop reason, then test whether returning C's
slow interval to A's natural update opportunities restores quality. LocalBA
consumption and camera-change range still need a real ORB-SLAM3 event record;
they cannot be reconstructed from the current logs.

## 5. Adam step vs global updates

`adam_step_audit.txt` records the issue. In the r1 live terminal:

- host `frozen_opt_steps` = 8712;
- every Adam group step = 8625;
- all six groups have the same step and complete moments/LRs.

The 87 difference is not resolved by a per-iteration log. Code sites that can
affect the relation are recorded:

- `trainingSetup` creates the initial Adam (`gaussian_model_line.cpp:2237`);
- `replaceTensorToOptimizer` carries or initializes step (`~2318-2340`);
- densify/clone/split state extension carries step (`~2600-2640`);
- diagnostic `restoreRecoverySnapshot` can rebuild state (`~4974-5011`);
- arm replay resets only the host counter (`~5110`);
- host increments occur only after `optimizer_->step()` (`~3701`, `~4767`).

Because all six groups share 8625, there is no group-specific reset signature.
The difference could be no-state-update optimizer calls or a state replacement;
it must not be called “87 missed updates” without a per-step audit. No optimizer
change is made.

## 6. Artifacts

- `scale_stats.csv`, `anomaly_rows.csv`, `line_eccentricity.csv`
- `anomaly_rule.txt`, `indices/`
- `cf/` — eight counterfactual runs, each with `offline_psnr.csv`,
  `offline_summary.txt` and `diff/*.png`
- `cf_summary.csv`
- `adam_step_audit.txt`
- source/analysis scripts: `analyze_extreme_scales.py`
- `instrumentation.patch` SHA-256
  `ff71241f1c0a67ae6366d606c104c950b2551600676725512cb449939c91e403`;
  `source_bin_hashes.txt` records the exact offline renderer and diagnostic
  binaries/configs. `bin/`, `libgaussian_mapper.so` and `config/` retain the
  exact r2 artifacts so later counter builds do not delete them. The patch
  applies to production HEAD
  `ec85e80886e2f0494acb87781e075334bcf2ba58`.

Limitations: top-20 is a fixed diagnostic subset, not an all-row causal
decomposition; center visibility is approximate; the offscreen footprint of a
large Gaussian is not measured; the C/A comparison still has different KFs and
topology; and no new online run was made.
