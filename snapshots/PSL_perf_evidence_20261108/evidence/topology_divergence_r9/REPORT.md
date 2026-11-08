# Strict 120-update replay with identical optimizer reconstruction

## Why this run

The earlier strict runs (`r6` seed reset, `r7` captured RNG restore) left a
possible experiment confound: A0 used the live optimizer, while A1/C restored
from a changed row count and rebuilt the optimizer. `r9` forces the SAME
`restoreRecoverySnapshot(..., force_rebuild=true)` path for A0/A1/C, so all arms
rebuild the optimizer and reload the same captured Adam state.

Run was bounded at iteration **6599** because the forced-rebuild production run
ended at 6991, so the earlier 6999 target was unreachable. 120 updates cover
two densifications: **6600** and **6700**.

## Common state

- 162 KF, 370358 GS, 1409 line, active SH 3
- `param_fp=b6ee8e82a2773ac6_n370358_nline1409`
- `adam_fp=51d652e14b5cb1cb`
- 120-view fixed sequence; 24 fixed eval views
- captured CPU/CUDA torch RNG states restored before each arm
- `std::rand` seeded; not called because the fixed sequence is exactly 120
- original `trainForOneIterationErrorGuided`, original densify/clone/split/prune,
  all losses, `opacity_reset_interval=0`

## Native GS PSNR

| arm | backend | final GS | final line | step0 | step1 | step10 | step100 | step120 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | A | 368328 | 1410 | 25.5463 | 25.4741 | 26.65843 | 28.57022 | 28.5705 |
| A1 | A | 368316 | 1412 | 25.5463 | 25.4741 | 26.65843 | 28.56971 | 28.6052 |
| C | C | 368324 | 1412 | 25.5463 | 25.4741 | 26.65834 | 28.56924 | **28.6110** |

- step0/step1 bit-identical; step10 ~`9e-5 dB`.
- step120: C−A0 `+0.0405 dB`, C−A1 `+0.0058 dB`, A0−A1 `−0.0347 dB`.
- C is essentially tied with A1 and above A0. This is not a sustained C
  degradation.

## First decisions

At **6600** complete sets and post-densify/pre-optimizer state are identical:

```text
prune 7056, clone 2664, split 3877
pre-update fp = 89d4f940ea6028d4_n369843_nline1405
grad_max=0.0127795, scale_mean=0.0148415,
opacity_mean=0.619997, struct_loss=0.00459062
```

At **6700** the complete sets differ for every pair, including A0/A1:

| arm | prune | clone | split | final total | final line | pre-update fp |
|---|---:|---:|---:|---:|---:|---|
| A0 | 11198 | 3028 | 6655 | 368328 | 1410 | `0470a24a...` |
| A1 | 11203 | 3017 | 6659 | 368316 | 1412 | `22be2455...` |
| C | 11198 | 3019 | 6660 | 368324 | 1412 | `75e25bfe...` |

The first **parameter** fingerprint divergence is again at step2:

```text
A0 a785c1ced942319c_n369843_nline1405
A1 c025c7320141d529_n369843_nline1405
C  c10a145355149b2b_n369843_nline1405
```

A0/A1 have the same backend, same forced rebuild path and same captured RNG
state, yet differ at step2. This is not a restore-path difference and not an RNG
difference; it is consistent with CUDA/atomic floating-point reduction order.
It also means the first topology divergence at 6700 is not a C-only event.

## Conclusion

- 6600: identical complete decisions and pre-update state.
- step2: first non-RNG, non-restore parameter divergence; A0/A1 already differ.
- 6700: first topology divergence; A0/A1 also differ.
- C is not worse; step120 C−A1 `+0.0058 dB`, within the same-backend repeat
  spread (`A0−A1` absolute `0.0347 dB`).
- Do not change thresholds, LR, refinement budget, or production algorithm.
- Online A/C phase remains unresolved; see `ONLINE_PHASE.md`.

## Limits

One capture, one 120-update replay; A0/A1 same-process repeats. Restoring RNG
does not make CUDA reductions bitwise deterministic. `topo_mask.csv.gz` contains
all selected rows; row indices are not valid after 6600.

## Files

`summary.csv`, `decision_summary.csv`, `common_state.txt`, `kf_sequence.txt`,
`eval_views.txt`, `A0|A1|C/{metrics,topology_steps,pre_update}.csv`,
`audit/topo_topology.csv`, `audit/topo_selected.csv`,
`audit/topo_mask.csv.gz`, `audit/creation_events.csv`, `analysis.json`,
`analysis_stdout.txt`, `run.log`.
