# mapper-budget-v2-hash — pre-registered manifest

Written **before** any R50 measurement was taken. Nothing in this file may be revised after results
are seen.

Commit: `eaed404973` + the working-tree changes of this round (see `git log` on
`feat/mapper-budget-v1`). Sequence `office0`, RGB-D, dataset
`/workspace/code/SEGS-SLAM/datasets/replica/office0`.

## The one algorithmic variable

| arm | `PHOTO_SLAM_INACTIVE_KEEP` | meaning |
|---|---|---|
| **R100** | 1.0 | accepts every valid inactive-geo候选 in each batch (tensors passed through unchanged) |
| **R50** | 0.5 | accepts `round(0.5 · N)` of the **same** batch |

Everything else is identical between the arms: same init checkpoint, same frozen scene, same
pre-generated training schedule, same variable-topology densification (clone / split / prune /
opacity all run normally; the fixed-count gating is **released** because this is a variable-topology
budget experiment, not the fixed-count convergence experiment), line structural loss **off**,
identical learning rates, SH and pyramid schedule.

## R50 sampling definition (fixed here, and only here)

For candidate `i` of batch `b`:

```
key(b, i) = splitmix64( (kf_id(b) + 1) * 0x100000001B3 ^ (base_index(b) + i) * 0xC2B2AE3D27D4EB4F ^ seed )
```

with `seed = PHOTO_SLAM_INACTIVE_SEED` (default `20260101`), `kf_id(b)` the frozen keyframe that
produced the batch and `base_index(b)` the running count of candidates seen before this batch.

* the `round(0.5·N)` candidates with the **smallest** `key` are kept;
* the kept rows are re-inserted in their **original candidate order**;
* the actual kept indices, their hashes and the per-batch counts are written to
  `<arm>/inactive_kept.csv` and `<arm>/inactive_batches.csv`;
* the ranking consumes **no** training random numbers.

**Explicitly not claimed:** this is a stable hash sample, **not** image-space stratification, and it
makes **no** guarantee about surface coverage. No further ratio (e.g. 0.25, 0.75) will be scanned this
round.

## Cost accounting rules

* The R50 **filter cost is counted** in the arm's wall clock (the mask/index-select runs inside the
  same mapping path).
* The replay supplies the inactive-geo entry from the frozen keyframes, so the **candidate-generation
  cost is not re-measured**; it is identical in both arms and is stated as an omission. Replay wall
  clock must **not** be presented as full online-SLAM time, and must not be compared with the earlier
  capture's wall clock.
* Evaluation, saving and plotting are timed separately from mapping.

## Acceptance (one-sided, Δ = R50 − R100)

A **positive** quality change is never a failure.

* PSNR ≥ **−0.05 dB**, SSIM ≥ **−0.001**, LPIPS ≤ **+0.002**, on the shared evaluation views rendered
  from the same checkpoint and the same schedule.
* No NaN, finite parameters, identical `init_fingerprint` for both arms.
* 3 % is an **engineering speed target**, not a significance test: a single run exceeding 3 % is not
  called a significant acceleration.

Proceed to a second pair with the order swapped **only if** the first pair shows a candidate benefit.
No multi-seed, no multi-scene this round.

## Order of operations this round

1. ~100-step R100-vs-R50 check (init parameters, step-0 render, external event order/payload, actual
   view/camera/SH/pyramid/LR, appended-point counts, loss, parameter update).
2. Full R100 vs R50 paired replay over the same recorded update sequence.
3. Report; if R100 does not reproduce the expected path, fix the replay and do **not** start R50.
