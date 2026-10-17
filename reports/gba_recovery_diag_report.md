# GBA recovery — mechanism diagnosis (post-v1)

Status: **negative / not promoted**. v1 results in `reports/gba_recovery_v1_report.md`
are unchanged. This round only instruments and diagnoses; it does not retune or
re-run the 500-step matrix.

## 1. Command (exact)

```
rm -rf abl_out/gba_recovery_diag && \
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_RECOVERY_DIAG=abl_out/gba_recovery_diag \
PHOTO_SLAM_GBA_RECOVERY_VIEWS=24 timeout 900 bin/tum_rgbd \
  ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_freiburg3_long_office_household.yaml \
  /workspace/code/SEGS-SLAM/datasets/tum/rgbd_dataset_freiburg3_long_office_household \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg3_long_office_household.txt \
  /tmp/gba_diag_run2 no_viewer
```

Raw diagnostics: `reports/gba_recovery_diag/` (same files as
`abl_out/gba_recovery_diag/`): `repeat_consistency.csv`, `repeat_diag.csv`,
`repeat_renders.csv`, `step0_compare.csv`, `event_evt.txt`, `gba_scope_kf.csv`,
`gba_scope_views.csv`, `loop_scope.csv`, `loop_scope_views.csv`,
`non_event_visibility.csv`, `common_state.txt`.
Renders (`*.png`, incl. `image_residual`) stay in `abl_out/` only (large).

Event used by this run: evt=1, map=0, kfs=244, gaussians=79670 (137 line),
iteration=16065, SH active/default = 3.

## 2. Two-run repeat: what differs and what does not

`repeat_consistency.csv`

| check | result |
|---|---|
| step0 params (all 6 groups) | equal (bitwise) |
| step0 Adam state (all 6 groups) | equal |
| param-group LRs | equal |
| Adam `step` counters | equal |
| LR / SH / pyramid schedule fingerprint | equal |
| eval-view sequence | equal |
| step0 renders (all 24 views, PSNR/SSIM/pixel) | equal (`dpsnr=0`, `render_maxdiff=0`) |

`repeat_diag.csv` (A vs B, restored snapshot)

| stage | xyz | scaling | rotation | opacity |
|---|---|---|---|---|
| step0 | 0 | 0 | 0 | 0 |
| first backward | 0 | 0 | 0 | 0 |
| first update | 1.5e-8 | 1.4e-6 | 6.3e-3 | 9.5e-7 |
| step10 | 7.6e-4 | 1.8e-2 | 3.6e-2 | 1.1e-2 |

Interpretation:

* The restore path is faithful at the event: params, Adam moments, Adam step
  counters, LRs and the render are identical. The earlier `lr_equal=0` was a
  real bug (`restoreRecoverySnapshot` did not restore optimizer param-group
  LRs); it is fixed and now reports `1`.
* `post_step_param_max_abs_diff ≈ 0.039` is **not ULP-level**. It is the value
  after ten optimizer updates. The first update already shows a rotation
  difference of 6.3e-3 (rel-L2 6.7e-5), and it accumulates to 3.6e-2 by step 10,
  i.e. the same order as the v1 number. So the v1 repeat mismatch is
  accumulated optimizer/update-path divergence, not a snapshot defect and not
  proof that recovery is impossible.
* Honest caveat: in this run the first-backward gradients are bitwise equal,
  yet the first update still differs (mostly rotation). The fingerprint set
  (params/Adam/LR/schedule) does not cover the whole update path. This is an
  instrumentation gap for the *rotation* group; it does not show a
  restore error, and no implementation error was found that would explain it.
  The `quat_angle_deg` rows report 180 deg as a sentinel and should be ignored.
* Render consequence stays negligible at step 0–1 (PSNR diff ~1e-6 in the v1
  repeat check), i.e. the drift is a tensor-level effect, not a visible one.

## 3. Step-0 deformation decomposition (no `optimizer.step`)

Same frozen state, one event, 24 common eval views. PoseOnly = cameras only;
MeansOnly = centres only (diagnostic); Full = production path. MeansOnly is
**not** proposed as a fix.

| arm | mean ΔPSNR | min | max | positive views | mean ΔSSIM |
|---|---|---|---|---|---|
| MeansOnly | +0.030 | −1.168 | +0.989 | 14 / 24 | +0.0058 |
| Full | +0.030 | −1.180 | +0.993 | 14 / 24 | +0.0058 |

* MeansOnly vs Full: max |ΔPSNR| 0.013 dB, mean 0.003 dB → the immediate effect
  is dominated by centre translation; anchors/rotation contribute ~nothing at
  step 0.
* The gain is **not** global: 10/24 views get worse, worst −1.17 dB. So
  "average PSNR" alone cannot rule out local damage; and because the effect is
  centre-translation dominated, the world-coordinate mean displacement alone
  cannot predict it either.

Per-view (ΔPSNR / predicted pixel displacement on the same view):

| view | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔPSNR | .29 | .30 | .03 | .09 | .13 | −.45 | −.80 | −.56 | −.15 | −.26 | .37 | −1.17 |
| pix mean | 1.2 | 1.2 | 1.5 | 1.5 | 1.3 | 1.3 | 1.3 | 1.6 | 2.0 | 1.9 | 2.4 | 3.2 |

| view | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ΔPSNR | −.81 | −.30 | .09 | .86 | .21 | .58 | .42 | .75 | .99 | .33 | −.14 | −.09 |
| pix mean | 2.3 | 1.9 | 1.9 | 2.2 | 2.3 | 1.9 | 1.7 | 1.5 | 2.2 | 3.0 | 2.1 | 1.5 |

No monotone relation: view 11 is both the largest displacement and the worst
drop, while views 6–7 drop with modest displacement and view 21 gains with
large displacement. Sign of the effect is view-local, not a function of the
mean correction magnitude alone.

## 4. Correction scope: LOOP vs GBA (world and predicted pixel)

`gba_scope_kf.csv` / `gba_scope_views.csv` (GBA event), `loop_scope.csv` /
`loop_scope_views.csv` (last LOOP event before the GBA).

| quantity | GBA (244 KF) | LOOP (242 KF) |
|---|---|---|
| rotation mean / p95 / max (deg) | 0.331 / 0.573 / 0.933 | 0.485 / 1.229 / 1.282 |
| translation mean / p95 / max (m) | 0.0190 / 0.0419 / 0.0431 | 0.0248 / 0.0469 / 0.0476 |
| predicted pixel disp. mean, min–max | 1.19 – 3.21 px | 0.00 – 8.04 px |
| predicted pixel disp. p95, max | 7.43 px | 11.81 px |
| predicted pixel disp. max | 17.3 px | 25.7 px |

* The per-event GBA world delta is small (`event_evt.txt` per-KF: rot 0.08–0.93 deg,
  trans 0.003–0.043 m), but projected onto the eval views it is 1–3 px mean and
  up to 17 px worst, i.e. well above sub-pixel; it is *not* a numerically
  irrelevant deformation.
* LOOP is 1.5–2.5× larger on the same measures — the **same order**, not "much
  larger". A LOOP-controlled 500-step matrix is therefore not justified this
  round.
* LOOP availability: earlier runs reported `available=0` because the capture
  block was inserted into the point-only handler and into an `#if 0` copy of
  `combineMappingOperations_withLine`. It is now in the live handler
  (`LoopClosingBA` case, before the legacy gate mutates poses) and reports
  `available=1, kfs=242`. No LOOP data is fabricated for the earlier runs; those
  runs are marked unavailable.

## 5. Uncoupled (non-event) rows

`non_event_visibility.csv`:

* This run: `unassigned=0`, `scene_not_event=0` → nothing uncorrected.
* Earlier event (recorded in the run-2 log): `unassigned=35`,
  `scene_not_event=0`; all 35 rows were visible in every eval view
  (`n_unassigned_visible=35` on views 0…23). So uncorrected rows are not
  "invisible leftovers" when they exist.
* The 169 rows tied to removed KFs reported in the v1-era log do not recur in
  this run; that count is event/timing dependent and is **not** reproduced here.
  No re-anchoring was added and no new CUDA contribution statistics were
  introduced.

## 6. Decision

* Concrete implementation errors found and fixed: the LR-restore gap, and the
  LOOP-capture placement (diagnostics only). No error was found in the
  production correction math beyond that.
* No stable quality gain: mean +0.03 dB but 10/24 views worse, and MeansOnly ≈
  Full. Keep the v1 negative conclusion. This round did not look for a positive
  result.
* Gaussian deformation at GBA is now **optional**:
  `PHOTO_SLAM_GBA_APPLY_CORRECTION=0` disables it (camera poses are still
  updated, legacy LOOP correction untouched). Default stays ON so earlier
  results remain reproducible. Example:

```
PHOTO_SLAM_RASTER_BACKEND=C PHOTO_SLAM_LINE_LOSS_EVERY=0 \
PHOTO_SLAM_GBA_APPLY_CORRECTION=0 bin/tum_rgbd ... /tmp/out no_viewer
# [GBA-Apply] deformation DISABLED ... camera poses still updated, 228 deltas not applied
# [GBA-Apply] ... moved=0 ... APPLIED
```

* Regression: one normal TUM run without diagnostic env still applies the
  deformation (`moved=82825`, count unchanged 82842→82842) and one run with
  `..._APPLY_CORRECTION=0` keeps `moved=0` while camera poses are updated. Both
  exit 0; no `free(): invalid pointer`.
* Next round, only if needed: a LOOP-controlled comparison (same snapshot, LOOP
  vs GBA deformation), since LOOP scope is now measurable. Not started.
