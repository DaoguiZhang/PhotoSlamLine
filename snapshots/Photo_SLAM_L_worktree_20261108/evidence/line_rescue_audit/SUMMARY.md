# Line-rescue audit: Mono entry difference, natural-opportunity scan, stress test

Continues from `aee2fd2153`. **No claim of rescue benefit is made.** The rescue
is proven *not* to fire on any natural sequence, and under an artificial point
famine it is attempted 14 226 times and succeeds 0 times.

## 1. Monocular entry-point audit (the A′ problem)

`System::TrackMonocularWithLine` dispatches on the line mode:

| `LINE_MODE` | entry | tracking state machine | `Frame` ctor |
|---|---|---|---|
| 0 (=A) | `GrabImageMonocular` | `Track()` — original ORB-SLAM3 mono | **without** LSD extractor |
| ≥1 | `GrabImageMonocularWithLine` | `TrackWithLine()` | with LSD extractor |

So **mono A is not a clean "lines off" control** — it changes the entry function,
the whole state machine and the `Frame` construction. The previous phase's mono
A-vs-B/C comparison was therefore confounded, as flagged.

**A′ = `LINE_MODE=1`** is the correct same-entry point control: identical entry,
identical state machine, lines still detected/matched and MapLines still built,
but **zero line edges in any optimizer**. Verified by grep — the only two gates are
`Optimizer.cc:2620` (pose optimisation: `GetLineMode() >= 2 && GetPoseOptLineEnabled()`)
and `Optimizer.cc:5674` (local BA: `GetLineMode() >= 2 && GetLbaLineEnabled()`).

RGB-D has no such dispatch (`tum_rgbd.cpp` always calls `TrackRGBDWithLine` → 
`GrabImageRGBDWithLine`), so for RGB-D, A is already same-entry.

Per-run init/reset/coverage are recorded in the existing CSVs
(`fed`, `traj_frames`, `coverage`, `longest_seg`, `not_localised`, `kf`). From the
previous phase: RGB-D coverage 1.000 in both scenes; far Mono 0.7482 in every run;
near Mono unstable with 3/15 runs producing **no trajectory at all**
(`SYSTEM-> Reseting active map in monocular case`, 0 KFs).

## 2. Natural rescue-opportunity scan (no GT, no threshold tuning)

New switch `PHOTO_SLAM_DEBUG_LINE_SCAN=1` emits `[LineScan]` once per rescue
decision point, evaluating the LINE evidence **unconditionally** — whether or not
the point verdict failed, and with the rescue switched off, so the trajectory is
never perturbed. `LineSupportEvidence()` was extracted out of
`HasReliableLineSupport()` so both share one implementation.

An **opportunity** = same frame has `oldPtVerdict=FAIL` **and** `lnOk=1`
(enough unique reliable MapLines, inlier ratio OK, directional spread ≥ threshold).

Run in condition **B** (mode 2, rescue OFF) on 7 sequences:

| sequence | decisions | oldPtVerdict=FAIL | opportunities | lnUniq median / max | lnOk=1 |
|---|---|---|---|---|---|
| TUM near RGB-D | 1052 | 0 | **0** | 0 / 5 | 0 |
| TUM far RGB-D | 792 | 0 | **0** | 0 / 4 | 0 |
| TUM near Mono | 433 | 2 | **0** | 0 / 2 | 0 |
| TUM far Mono | 606 | 0 | **0** | 0 / 3 | 0 |
| ETH3D planar_2 | 629 | 0 | **0** | — / 88 | 5 |
| ETH3D planar_3 | 899 | 0 | **0** | — / 131 | 28 |
| ETH3D desk_3 | 1977 | 1 | **0** | — / 32 | 42 |

**Zero opportunities on all seven sequences**, for two independent reasons that
never co-occur:

* **TUM**: the point gate essentially never fails (0–2 frames), *and* the unique
  reliable MapLines are tiny (median 0, max 2–5) versus the required 8 — so even
  a hypothetical gate failure could not be rescued.
* **ETH3D**: line evidence *is* sufficient on 5/28/42 decisions (up to 131 unique
  lines), but on exactly those frames the point gate passes comfortably
  (`ptAfterPO` median 205–485).

## 3. Separately-labelled stress test

Because no natural opportunity exists, a point-observation famine was induced
with a **fixed, identical rule in every condition**, applied only from a fixed
frame id so initialization is untouched, and never touching line observations:

```
PHOTO_SLAM_STRESS_DROP_POINT_RATIO=0.9   PHOTO_SLAM_STRESS_START_FRAME=30
```

implemented as a deterministic drop of POINT edges inside
`Optimizer::PoseOptimizationWithLine` and `Optimizer::PoseOptimization`. It only
moves the **post-optimisation** verdict, so the VO `<15`/`<20` gates are
unaffected and the rescue is triggered at the local-map gate while the point
floor (≥10) may still hold.

Conditions: **P = A′**, B, C. Two rounds on ETH3D `desk_3`, one round on TUM near.

| run | traj frames | scans | rescue attempts | **rescues** | KF |
|---|---|---|---|---|---|
| desk_3 P / B / C | 1 / 1 / 1 | 980 | 1902 / 1902 / 1902 | **0 / 0 / 0** | — |
| near RGB-D P / B / C | 586 / 586 / 586 | 498 | 938 / 938 / 938 | **0 / 0 / 0** | 1 |

* **14 226 rescue attempts across all stress runs, 0 successes.**
* near RGB-D under stress: ATE RMSE **1.011 / 1.010 / 1.013 m** (nominal 0.025 m,
  i.e. ~40× worse), RPE Δ=1 0.0205/0.0226/0.0208 m (nominal 0.0066 m), **1 KF** —
  the trajectories are unusable.
* **P, B and C produced identical trajectories and near-identical ATE**: adding
  line edges (P→B) or enabling the rescue (B→C) changed nothing.

Why the rescue could not fire even here (`[LineRescue]` reason histogram):

| run | reason=1 too few lines | reason=2 degenerate | **reason=4 point floor <10** |
|---|---|---|---|
| near RGB-D (C) | 466 | 0 | 3 |
| desk_3 (C) | 165 | 6 | **780** |

The three near-RGB-D frames that *did* satisfy the line-only criterion had
`ptAfterPO = 0` — the 90 % drop removed the very points the rescue requires.
The drop therefore overshot past the narrow viable window (roughly 10–29
remaining points). This is reported rather than tuned.

## 4. Answers

* **Does the rescue exist naturally?** No. Zero opportunities on 7 sequences.
* **Does it work under stress?** Not demonstrated: 14 226 attempts, 0 successes;
  the dominant rejection is the rescue's own point floor.
* **Are P/B/C different?** No — identical trajectories and ATE, so line edges and
  the rescue changed nothing measurable in either regime.
* **Consequence for the next step**: the precondition "proven to recover an
  accurate pose and sustain tracking" is **not met**, so the tracking list and
  key-frame / map-line insertion gates are **deliberately not evaluated** here.

## 5. Limitations

* The stress test is artificial and must never be presented as a natural result.
* A single drop ratio (0.9) was chosen a priori from the median point counts; it
  overshot. A ratio targeting the 10–29 point window was **not** searched, because
  that would amount to tuning the test to the rescue.
* n = 1–2 rounds per stress condition; the stress runs are, however, strikingly
  deterministic (identical across repeats), because the drop rule is deterministic.
* SLAM-only: no image-quality metric is produced or reported.
* Nothing here evaluates the rescue's *accuracy when it succeeds*, because it
  never succeeded.

## 6. Artifacts

Scripts: `rescue_opportunity_scan.py`, `eth3d_ablation_run.sh`,
`stress_rescue_repeats.sh`, plus the A′ condition and `PHOTO_SLAM_DEBUG_LINE_SCAN`
in `fr3_ablation_run.sh` / `LineMode.h` / `Tracking.{h,cc}`.

Evidence (this folder): `opportunity_scan_7seq.txt`, `opportunity_scan_stress.txt`,
`tum_scan.csv`, `eth3d_scan.csv`, `scan_logs.tar.gz` (all natural scan runs),
`stress_logs.tar.gz` (all stress runs).
