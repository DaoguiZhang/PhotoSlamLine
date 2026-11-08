# room1 variable-topology replay: first divergence

## Setup

One room1 C run was captured at mapper iteration **6999**, one update before the
densification at 7000. The capture is in memory only; the normal loop then
stops. All arms start from the same snapshot, not from a re-run of SLAM:

- 167 KFs, 400630 Gaussians (1982 line), active SH 3, base iteration 6999,
  completed optimizer steps 6998
- `param_fp = 4bd3a983781e0a9c_n400630_nline1982`
- `adam_fp = d557cfcfe949aa6f`
- One fixed `kf_sequence` of 400 views, shared by every arm; 24 fixed eval views
  are excluded from training.
- External scene/KF set is frozen. No MappingOperation is consumed during the
  arms. Normal `trainForOneIterationErrorGuided()` is used with densify/clone/
  split/prune enabled; no frozen-count or fixed-geometry switch.

Target 6999 was selected after one earlier capture run ended at 7526 optimizer
steps (target 8899 was unreachable). It is still inside `densify_until_iter`
(15000), immediately before the next 100-step densification.

## Result (native GS PSNR, fixed views, no grad)

`summary.csv`:

| arm | backend | final Gaussians | line | GS step0 | GS step1 | GS step100 | GS step400 |
|---|---|---:|---:|---:|---:|---:|---:|
| A0 | A | 409570 | 1937 | 25.1414 | 24.9939 | 28.9101 | **29.5710** |
| A1 | A | 409388 | 1939 | 25.1414 | 24.9747 | 28.9070 | **29.6155** |
| C | C | 409375 | 1934 | 25.1414 | 24.9875 | 28.9166 | **29.6305** |

C−A0 = `+0.0595 dB`; C−A1 = `+0.0150 dB`; A0−A1 = `−0.0445 dB`
(absolute `0.0445 dB`) at step 400. The same-backend A0/A1 spread is larger
than the C−A1 difference. This single 400-update capture only supports:

> under this error-guided replay condition, no sustained C quality degradation
> was observed.

It is not a general claim about C. RNG state was not restored across arms, so
the first divergence cannot be attributed to floating-point micro-differences
alone.

Quality already differs at the first update after the common snapshot
(step1: A0/A1/C = 24.9939/24.9747/24.9875), while the 7000 decision is still
identical. A0 and A1 differ there too; this is a replay observation, not a
C-specific implementation error.

## First densification and first different decision

At iteration **7000** (first densification after the snapshot), all three arms
made the same decisions and ended with the same population:

```text
entry: 400630 -> 403558, line 1982 -> 1978
opacity-prune 5554, lazy-line prune 0, clone 3392, split 5090
grad_max 0.00671631, near-threshold 2849
scale_mean 0.014732, opacity_mean 0.642034
photo_loss 0.0192609, struct_loss 0.00648581
```

The top-16 pre-prune rows and their gradients were identical across A0/A1/C at
7000 (top row `347836`, grad `0.00671631`).

The first **different topology decision** appears at iteration **7100**
(`first_decision.csv`):

| arm | opacity prune | clone Δ | split Δ | final total | final line | grad_max | near threshold |
|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | 10863 | 3134 | 6115 | 401944 | 1969 | 0.00699116 | 4759 |
| A1 | 10888 | 3133 | 6079 | 401882 | 1966 | 0.00681147 | 4745 |
| C | 10911 | 3097 | 6043 | 401787 | 1968 | 0.00681154 | 4753 |

`prune_lazy=0` in all arms, so this is an opacity-prune count difference, not
line-lazy pruning. `scale_mean` (0.01426), `opacity_mean` (0.6483–0.6485),
`photo_loss` (0.01484–0.01485) and `struct_loss` (0.00647–0.00648) are nearly
identical at 7100. The difference is concentrated in the high-gradient tail /
near-threshold rows, not a uniform scale/opacity drift.

At 7100 the top-ranked row index is the same (`329767`) but its gradient is
already different (`0.00699116` vs `0.00681147/0.00681154`). After the 7000
clone/split/prune event, row indices are **not** a cross-arm correspondence;
the matching index must not be interpreted as the same Gaussian.

## Interpretation

1. The first observed difference is in **parameters/quality at the first
   replayed update**, while the 7000 topology decision is still identical. The
   same-backend A0/A1 pair shows the same kind of divergence, but because RNG
   state was not restored this cannot be labelled floating-point-only.
2. The first topology decision difference is at iteration **7100**; the first
   densification at 7000 is identical. The divergence is carried by a small
   high-gradient tail; global scale, opacity and structural-loss means remain
   almost equal.
3. C is slightly better than A0; the C−A1 difference is smaller than the
   A0/A1 same-backend spread. No statistical test is possible from one capture.
   This does not show C degrading quality under this replay condition, and it
   does not support threshold compensation.
4. The online A/C matched-frame gap is computed at shutdown. The `frame001300`
   split is an **image-time classification**, not evidence that quality had
   already diverged before the LOOP event. Historical quality requires model
   states captured at event/frame checkpoints, which this capture does not have.
5. The next discriminator should be online pose updates, training-view
   allocation and stop timing. The online A/C matched-frame gap (`+1.0327 dB`
   on 17 common frames) is confounded by different trajectories and shutdown
   timing and cannot be assigned to C from the available data.

## Limits

- One capture, one 400-step replay; A0/A1 are not independent-process repeats.
  The view sequence and camera/scene inputs are shared, but the quantile/audit
  reads force synchronization, and CUDA/std RNG state is not restored.
- `PHOTO_SLAM_TOPO_AUDIT=1` adds device reads at each densification. This is a
  diagnostic run, not a timing result.
- No nonzero GBA event occurred in this capture; GBA default behavior was not
  changed. No nonzero-GBA 100-step evidence is claimed.
- Dense images are not copied to the evidence directory (1.2 GB); PSNR, metrics,
  topology CSVs and logs are.

## Files

- `summary.csv`, `common_state.txt`, `kf_sequence.txt`, `eval_views.txt`
- `A0/metrics.csv`, `A1/metrics.csv`, `C/metrics.csv`
- `A0/topology_steps.csv`, `A1/topology_steps.csv`, `C/topology_steps.csv`
- `audit/topo_topology.csv`, `audit/topo_selected.csv`
- `audit/creation_events.csv`, `audit/creation_summary.txt`
- `first_decision.csv`, `analysis.json`, `analysis_stdout.txt`, `run.log`
