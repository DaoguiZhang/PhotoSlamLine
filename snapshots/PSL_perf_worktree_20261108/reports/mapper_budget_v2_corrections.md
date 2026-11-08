# mapper-budget-v2-hash — correction of the reported results (raster-perf-v1, item 一)

All of this is derived from the **existing** artifacts of `abl_out/bv2_formal_r0` plus the code.
**No experiment was re-run** for these corrections.

## 一.1 Where the missing 53 119 went — an identity, not an attribution

Measured endpoints of the pair:

| | injected (accepted candidates) | final Gaussians | net growth |
|---|---|---|---|
| R100 | 148 345 | 144 734 | 142 680 |
| R50 | **74 176** | **123 684** | 121 630 |
| Δ = R50 − R100 | **−74 169** | **−21 050** | −21 050 |

So **53 119 of the 74 169 injection difference did not survive to the final count.**

Exact identity (with `clone`, `split`, `prune` as net row changes and initial counts identical at
2054 in both arms):

```
Δfinal = Δinjected + (Δclone + Δsplit) − Δprune
−21 050 = −74 169 + D      =>      D = +53 119
```

`D = (clone+split−prune)_R50 − (clone+split−prune)_R100 = +53 119`.

**Reading, stated at the level the data supports:** the arm that received **fewer** injected points
grew **more** through its own densification — by 53 119 rows. This is *not* "new densification
compensating a shortfall in the arm that has more points", and it is *not* "no compensation at all";
it is the opposite direction of the naive expectation, over-compensating.

Window split (from the recorded evaluation points):

| window | Δinjected | Δfinal | implied D |
|---|---|---|---|
| 0 → 1000 | ≈ −14 500 (2 flushes) | −9 505 (30 939 vs 21 434) | ≈ +5 000 |
| 1000 → 5000 | ≈ −59 669 | −11 545 | ≈ +48 124 |

So most of the over-compensation happens **late**, once clone/split has had time to act.

**Consistent hypothesis, not established fact:** with fewer Gaussians each one carries a larger share
of the same image gradient, so more of them cross the densify gradient threshold
(`densifyGradThreshold()`), producing more clone/split per unit of injected points.

**Limitation of this item:** the per-entry creation audit (`PHOTO_SLAM_CREATION_AUDIT`) was **not**
enabled for this pair, so the identity above fixes `D`'s size and sign but **cannot** split it into
clone vs split vs prune. Any per-entry attribution for R100/R50 would need that audit enabled.

## 一.2 What the 23 seconds actually covers

Code verified: the timed region in the replay loop is

```
syncDevice();  t_step0 = now();
  replayDriveInactiveGeo();          // candidate selection/hash filter + increasePcd + scale init
  trainForOneIterationErrorGuided(); // render + loss + backward + densify/clone/split/prune + optimizer.step
syncDevice();  mapping_seconds += now() - t_step0;
```

**Included:** the R50 hash filter and its `index_select`, `increasePcd` for the injected points and
their scale/direction initialisation, the whole training iteration including render, losses,
backward, `densifyAndPruneWithLineAwareness` (clone/split/prune/reset-opacity) and `optimizer.step`
(whose Adam state grows with the model), plus one device sync at each end.

**Excluded (not billed to `mapping_seconds`):**

| excluded | why it matters |
|---|---|
| candidate **generation** (keyframe depth sampling / projection) | replaced by the frozen-keyframe driver; identical in both arms, but the real cost is absent |
| scene construction and the SLAM front end | outside the loop entirely |
| per-condition model restore + fingerprint + checkpoint I/O | before the loop |
| evaluation, including render + PSNR/SSIM + diag | counted separately in `cum_eval_seconds` |
| final PLY/OBJ export | after the loop |

**Naming correction:** `mapping_seconds` here is **training time including the application of the
injected point stream** — it is *not* full online-mapping time and must not be compared with the
earlier capture's wall clock. The v2 report's statement that "the R50 filter cost is counted" is
**correct** (the driver sits inside the timed region); the statement that these seconds equal mapping
time is not, and is withdrawn.

## 一.3 Evaluation protocol — the overlap is total, and that must be stated

* Evaluation is at completed optimizer steps **0, 1000, 5000**, on **12 fixed keyframes** given as
  indices into the keyframe-map order: `0 9 18 27 37 46 55 64 74 83 92 101`.
* The injected stream was driven at a fixed period of 45 steps in ascending keyframe order, so it
  consumed map indices **0 … 110** (111 of the 112 frozen keyframes; the flush points record fids
  `9,19,29,…,109`, and both arms report the *identical* keyframe sequence and identical candidate
  counts per batch).
* Therefore **every one of the 12 evaluation keyframes is also an inactive-geo source keyframe** —
  the same keyframe whose depth-derived candidates were injected, and whose undistorted image
  (`img_undist_`) supplied those points' colours.

Correct protocol label:

> **Frozen-keyframe deterministic injection with fully overlapping evaluation views.** The evaluation
> criterion (GT image) is not trained on, but the evaluation *views* and the *point-source and
> colour-source* views are the same set. The measured PSNR/SSIM are therefore **not independent of
> the injection source**, and a small advantage for the arm that injects more points is expected on
> the source views. This bounds how the R100−R50 gap may be interpreted; it does not by itself
> invalidate the count/time comparison.

A leak-free variant would hold out evaluation views from the injection set; that is **not** done here
and is recorded as a limitation.

## 一.4 Experiment description

The earlier phrasing "replay of the office0 capture" is withdrawn. The correct description is:

> **mapper-budget-v2-hash = deterministic injection of a frozen-keyframe external stream.** The
> geometry input is generated on demand from the frozen keyframes by `replayDriveInactiveGeo()` at
> recorded optimizer steps; it is **not** replayed from the persisted online creation log
> (`creation_log.bin`), and no real SLAM candidate generation is executed.

R100 is retained as the **acceleration baseline** for raster-perf-v1. Ratio scanning is stopped; no
0.25 / 0.75 arm will be run. LPIPS remains **unmeasured** for this pair.
