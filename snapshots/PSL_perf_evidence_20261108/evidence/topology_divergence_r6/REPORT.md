# Strict 120-update variable-topology replay

## Control audit

Base capture: iteration **6999**, 164 KF, 408494 Gaussians (864 line), active
SH 3, `param_fp=82ad3316c419a9f2_n408494_nline864`,
`adam_fp=5625cfc3d7c30853`. 120 real optimizer updates; view sequence length is
exactly 120, so the error-guided sampler never falls back to live sampling.

RNG audit:

- `std::rand()` is used by the error-guided view sampler. In these arms
  `frozen_kf_replay_` consumes the pre-generated 120-entry sequence and does not
  call the sampler, so no view-selection random draw occurs. `std::srand(12345)`
  is still reset per arm.
- The clone/split paths use `at::normal`. The default CPU and CUDA torch
  generators are reset per arm with `torch::manual_seed(12345)` and
  `torch::cuda::manual_seed_all(12345)`.
- `std::random_device` members are initialization-time only and are not used in
  replay. No generic RNG framework was added.

The normal `trainForOneIterationErrorGuided()` entry, original densify/clone/
split/prune, all losses and `opacity_reset_interval=0` are retained. Only the
view sequence is fixed by the experiment switch.

## Native GS PSNR on fixed views

| arm | backend | final GS | final line | step0 | step1 | step10 | step100 | step120 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A0 | A | 406893 | 867 | 25.2518 | 25.7088 | 27.37590 | 29.48236 | **29.5232** |
| A1 | A | 406825 | 867 | 25.2518 | 25.7088 | 27.37587 | 29.47497 | **29.5376** |
| C | C | 406790 | 868 | 25.2518 | 25.7088 | 27.37590 | 29.47144 | **29.5144** |

- step0 and step1 are bit-identical across all arms; step10 differs only at
  `~3e-5 dB`.
- step120: C−A0 = `−0.0088 dB`, C−A1 = `−0.0232 dB`,
  A0−A1 = `−0.0144 dB`.
- This single strict replay does **not** show sustained C degradation. At
  step120 C is 0.0088 dB below A0 and 0.0232 dB below A1; the latter is about
  1.6× the same-backend A0/A1 spread (0.0144 dB), not a sustained trend. The
  defensible statement is: under this error-guided replay condition with RNG
  reset, no sustained C quality degradation was observed.

## First divergence

At the first densification, iteration **7000**, the complete selection sets
(not just counts or top-16) are identical for A0/A1/C:

```text
pre-prune opacity mask: 7033 rows
pre-prune clone candidates: 3554
pre-prune split candidates: 4988
post-prune clone set: 3554
post-prune split set: 4988
```

The post-densify / pre-optimizer parameter fingerprint is also identical:
`8a6449954c971d93_n410003_nline863`. The post-update step1 fingerprint is
identical as well: `3b629f2af9314a8e_n410003_nline863`.

The first **parameter** divergence after the update is at step2:

```text
A0 c341a96d46ae21a0_n410003_nline863
A1 28fb4b94db37c593_n410003_nline863
C  ee6b13093dca8d93_n410003_nline863
```

A0/A1 are the same backend and have the same RNG reset, so this remaining
step2 divergence is **not RNG**. It is consistent with backend/atomic
floating-point reduction order; it cannot be attributed to C alone.

The first **topology decision** divergence is at iteration **7100**:

| arm | opacity prune | clone | split | final total | final line | pre-update fp |
|---|---:|---:|---:|---:|---:|---|
| A0 | 11080 | 2736 | 5234 | 406893 | 867 | `95a7a9a22c2262de...` |
| A1 | 11101 | 2714 | 5209 | 406825 | 867 | `368f73ddbe352d6f...` |
| C | 11117 | 2718 | 5186 | 406790 | 868 | `117691822dfdfc27...` |

Complete set differences at 7100 (A0 vs A1 / A0 vs C / A1 vs C) are in
`audit/topo_mask.csv.gz`; `decision_summary.csv` is the compact view. At 7000
the sets are exactly equal. At 7100 the
pre-prune clone-candidate set differs by 144/122 (A0/A1) and 180/162 (A0/C);
the post-prune sets differ by thousands of row indices because pruning has
already changed row numbering. **Do not pair rows after 7000.**

## Conclusion

- First complete decision point: identical at 7000.
- First non-RNG parameter divergence: step2 (A0/A1 as well).
- First topology divergence: 7100.
- No sustained C degradation in 120 updates; the C−A1 difference is smaller
  than or comparable to same-backend A0/A1 spread.
- Do not change thresholds, LR, or refinement budget from this result.
- The online A/C gap cannot be assigned to C topology from this replay. Its
  phase (pre-LOOP / event jump / post-event growth) still requires online
  checkpoint states; shutdown scores sliced by image time are not valid.

## Limits

One capture and one 120-update replay. A0/A1 are in the same process and not
independent statistical repeats. The RNG reset controls the sampled stream, but
does not make CUDA reductions bitwise deterministic. `topo_mask.csv` contains
all selected rows and is gzip-compressed; row indices are only valid within the
pre-prune snapshot.
