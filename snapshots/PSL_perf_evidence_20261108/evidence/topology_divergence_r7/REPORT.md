# Strict 120-update variable-topology replay (captured RNG states)

## Control

Base capture: iteration **6999**, 173 KF, 399081 Gaussians (589 line),
`param_fp=f3259aedcc26aff3_n399081_nline589`,
`adam_fp=988b4d768c14f4ab`, active SH 3. 120 real optimizer updates. The view
sequence has exactly 120 entries, so the error-guided sampler never falls back
to live sampling.

RNG audit:

- `std::rand()` serves the error-guided view sampler. With `frozen_kf_replay_`
  and a 120-entry sequence, no live view-selection draw occurs; `std::srand`
  is reset per arm but is not used inside these 120 updates.
- Clone/split use `at::normal`. The captured default CPU and CUDA torch
  generator states are copied into `RecoverySnapshot` and restored before each
  arm. This is the actual captured state, not a new seed.
- `std::random_device` is initialization-only and is not on the replay path.

Normal `trainForOneIterationErrorGuided()`, original densify/clone/split/prune,
all losses and `opacity_reset_interval=0` are retained. Only the view sequence
is fixed by the experiment switch.

## Native GS PSNR, fixed views, no grad

| arm | backend | final GS | final line | step0 | step1 | step10 | step100 | step120 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | A | 403886 | 585 | 23.1857 | 23.1048 | 24.86545 | 27.85946 | **27.7701** |
| A1 | A | 403807 | 586 | 23.1857 | 23.1048 | 24.86545 | 27.85321 | **27.6783** |
| C | C | 403794 | 586 | 23.1857 | 23.1048 | 24.86545 | 27.85285 | **27.6756** |

step0/step1 are bit-identical; step10 differs by `~2e-6 dB`. At step120:

```text
C−A0 = -0.0945 dB
C−A1 = -0.0027 dB
A0−A1 = +0.0918 dB
```

C is essentially equal to A1; A0 is the outlier. The same-backend A0/A1 spread
is much larger than C−A1. This one replay does **not** show C degrading
quality; it shows A0/A1 are not deterministic even with captured RNG states.

## First divergence

At iteration **7000**, the complete selection sets are identical for all arms:

```text
pre-prune opacity: 5793 rows
pre-prune clone candidates: 4362
pre-prune split candidates: 6982
post-prune clone: 4361
post-prune split: 6981
```

The pre-densify gradient/scale/opacity/structural state is also identical:
`grad_max=0.0118955`, `scale_mean=0.0144794`, `opacity_mean=0.641844`,
`struct_loss=0.00195243`. The post-densify / pre-optimizer fingerprint is
identical: `1f47cf5c18c40bf9_n404630_nline591`. Step1 post-update fingerprints
are also identical.

The first **parameter** divergence is at step2 (iteration 7001), while the
topology is still unchanged:

```text
A0 575e5163324c8f37_n404630_nline591
A1 094c16db8785d5bb_n404630_nline591
C  1ea285d26c5bed8b_n404630_nline591
```

A0/A1 have the same backend and restored RNG state, so this remaining
divergence is not RNG. It is consistent with CUDA/atomic floating-point
reduction ordering, and cannot be assigned to C alone.

The first **topology decision** divergence is at **7100**:

| arm | opacity prune | clone | split | final total | final line | pre-update fp |
|---|---:|---:|---:|---:|---:|---|
| A0 | 11755 | 3864 | 7147 | 403886 | 585 | `446eb350...` |
| A1 | 11809 | 3839 | 7147 | 403807 | 586 | `dc62a1f6...` |
| C | 11816 | 3829 | 7151 | 403794 | 586 | `e0053140...` |

At 7000 the full sets are exactly equal; at 7100 they differ for every pair,
including A0/A1. `topo_mask.csv.gz` has all selected rows; row indices are only
valid within the pre-prune snapshot and must not be paired after 7000.

## Conclusion

- 7000: complete prune/clone/split decisions identical; pre-optimizer
  fingerprint identical; step1 post-update fingerprint identical.
- 7001 (step2): first non-RNG parameter divergence; A0/A1 also diverge.
- 7100: first topology decision divergence; A0/A1 also diverge.
- No sustained C degradation in 120 strict updates; C−A1 is `-0.0027 dB` at
  step120, while same-backend A0/A1 differs by `0.0918 dB`.
- Do not modify thresholds, LR, refinement budget, or the production algorithm
  from this result.

The online A/C gap cannot be assigned to C topology from this replay. Its phase
still needs event/frame checkpoint states; shutdown scores sliced by image time
are not historical quality. See `ONLINE_PHASE.md` for the existing event-time
evidence and the missing records.

## Limits

One capture, one 120-update replay. A0/A1 are same-process repeats, not
independent statistical replicates. Captured CPU/CUDA RNG states are restored,
but CUDA reduction order remains nondeterministic. `topo_mask.csv.gz` is large
and compressed; row indices are not valid after a prune/clone/split.
