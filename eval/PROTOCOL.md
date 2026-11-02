# Frozen evaluation protocol — cross-dataset front-end evaluation

Frozen at: 2026-11-02 (before the first batch run).
This file is committed BEFORE any arm measurement and is not edited afterwards.
Any deviation found later is appended to `DEVIATIONS.md`, never silently applied.

## 0. What is being measured

Front-end / tracking robustness of the SEF SLAM pipeline across datasets and
modalities, with a **pre-registered internal ablation** of the line pipeline:

| arm | `PHOTO_SLAM_LINE_MODE` | lines detected/matched/MapLines created | line edges in PoseOptimization | line edges in Local BA |
|-----|------------------------|-----------------------------------------|-------------------------------|------------------------|
| T0  | 0 | no  | no  | no  |
| P1  | 1 | yes | no  | no  |
| P2  | 2 | yes | yes | yes |

T0 is **not** original Photo-SLAM. It is this code base with the line front-end
disabled. No comparison to the upstream published system is claimed in this
protocol.

## 1. Entry

Single entry for every scene and modality: `examples/eval_frontend.cpp`
(built as `bin/eval_frontend`). It instantiates `ORB_SLAM3::System` and calls

* mono: `TrackMonocularWithLine(im, ts, {}, filename)`
* rgbd: `TrackRGBDWithLine(imRGB, imD, ts, {}, filename)`

which are the exact calls the production entries `examples/tum_mono.cpp` /
`examples/tum_rgbd.cpp` make with `PHOTO_SLAM_SLAM_ONLY=1`. Differences from
production entries, all deliberate and all recorded here:

1. No Gaussian mapper, viewer, simple-knn, CUDA rasterizer, imgui or torch is
   linked (verified with `objdump -p`: NEEDED is `libORB_SLAM3.so` + OpenCV only).
2. The mapping-operation queue is drained after each frame
   (`getAtlas()->clearMappingOperation()`), because the mapper that normally
   consumes it is absent. This is the same call production makes.
3. Per-frame tracking state is written to `final_state.csv`.
4. No threshold, initialisation, covisibility, map-acceptance or optimisation
   logic is touched by the entry.

## 2. Arms are selected only by environment

```
PHOTO_SLAM_LINE_MODE   = 0 | 1 | 2          (arm)
PHOTO_SLAM_PO_LINE     = 1                  (unchanged in all arms)
PHOTO_SLAM_LBA_LINE    = 1                  (unchanged in all arms)
PHOTO_SLAM_LINE_RESCUE = 0                  (off in all arms)
PHOTO_SLAM_SLAM_ONLY   = 1
PHOTO_SLAM_NO_PACING   = 0                  (real-time pacing ON, as in production)
```
`PHOTO_SLAM_DEBUG_*` diagnostics are OFF in every run (they can perturb
measured wall-clock time; the diag code paths are not exercised).

## 3. Scenes, modalities, repetitions

Design (fixed before any run): **2 scenes x 2 modalities x 3 arms x 3 repeats**.
The scheduling order is rotated per repeat (Latin-square style) so that no arm
always runs first or last:

* repeat 1: T0 -> P1 -> P2
* repeat 2: P1 -> P2 -> T0
* repeat 3: P2 -> T0 -> P1

n = 3 per cell. **These are not statistical claims**; medians and ranges are
reported, no significance test is performed or implied.

Scenes for the primary batch:

| scene | dataset | frames (mono / rgbd) |
|-------|---------|----------------------|
| cabinet | `rgbd_dataset_freiburg3_cabinet` | 1147 / 1112 |
| large_cabinet | `rgbd_dataset_freiburg3_large_cabinet` | 1011 / 984 |

Modality rules:

* **mono** reads `rgb.txt` directly. Every RGB frame is processed; RGB frames
  with no depth partner are **not** dropped (1147 and 1011 respectively).
* **rgbd** reads the generated association file
  `cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_{cabinet,large_cabinet}.txt`,
  produced by `scripts/gen_assoc.py` with `max_difference=0.02` using TUM's
  greedy best-difference semantics (matching = 1112/1147 and 984/1011;
  unmatched RGB frames are reported, not hidden).
* `_validation` splits are not used. No cropping. No frame dropping. No
  resampling. No re-timing.

## 4. Configuration

Copied from the **same dataset family and same modality**:

| config | template (verbatim copy) |
|--------|--------------------------|
| `cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_cabinet.yaml` | `Monocular/TUM/tum_freiburg3_structure_notexture_near_lowfast.yaml` |
| `cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_large_cabinet.yaml` | same |
| `cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_cabinet.yaml` | `RGB-D/TUM/tum_freiburg3_structure_notexture_near.yaml` |
| `cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_large_cabinet.yaml` | same |

**Zero keys were modified.** Both cabinet sequences are recorded with the same
fr3 sensor, and the templates' calibration (`fx=535.4 fy=539.2 cx=320.1
cy=247.6`, zero distortion, 640x480, `Camera.fps=30`, `Camera.RGB=1`,
`RGBD.DepthMapFactor=5000.0`) was verified against the actual data: all images
are 640x480; depth is uint16 with median nonzero 6560 (cabinet) and 15745
(large_cabinet), i.e. 1.31 m and 3.15 m at factor 5000 — physically consistent
for the two scenes.

Known asymmetry, declared up front: the only monocular fr3 templates in the
repository are `_lowfast` (`nFeatures=5000`, `iniThFAST=5`, `minThFAST=2`),
while the RGB-D template uses `nFeatures=2000`, `iniThFAST=20`, `minThFAST=7`.
Mono and RGB-D are therefore run under different ORB extractor settings. This is
a property of the repository's config set, not a choice made after seeing
results, and it means **mono and RGB-D numbers must not be compared directly**.
No lowfast/normal sensitivity sweep is performed in this round (out of budget).

The 3rd CLI argument (Gaussian mapper settings path) is required by the entry
signature but is **ignored** by the SLAM-only entry; it is recorded in
`eval_entry.txt` under `gauss_cfg_ignored=` for transparency.

## 5. Metrics (definitions fixed now)

Primary (robustness):

1. **Accepted coverage** = frames whose tracking state after the call is `OK`
   (`2`) / frames processed. Read from `final_state.csv`.
2. **Correct-pose coverage** = accepted frames whose pose is within tolerance of
   ground truth under the alignment protocol in §5.4 / all ground-truth-evaluable
   input frames. Reported for a **pre-fixed tolerance set**
   `{0.05 m, 0.10 m, 0.20 m, 0.50 m}` — all four are reported, not just the most
   favourable one. Rotation tolerance joint `{5 deg}`.
3. **Longest continuous correct tracking** = longest run of consecutive frames
   that are (a) accepted and (b) inside the 0.20 m / 5 deg tolerance, in seconds
   (frame index difference / 30 fps for these sequences; the actual timestamp
   difference is used, not the nominal fps).
4. **Lifecycle events**: number of initialisations, number of LOST entries,
   number of recoveries (OK after LOST/RECENTLY_LOST without a new map), number
   of relocalisations, number of map resets (from `[System]` map count and
   `MapEventLog`), final number of maps.
5. Secondary: ATE RMSE (SE3, Sim3), RPE (delta 1 and 30), mean/median tracking
   time per frame, total wall time.

Coverage is reported in three separate columns and never merged:
`accepted`, `finite-exported` (finite pose actually written to
`CameraTrajectory_TUM.txt`), `gt-matched` (pose has a GT sample within
`max_dt=0.02 s`).

### 5.4 Alignment protocol (fixed)

SE(3) rigid alignment (no scale) via `eval_tum_se3_v2.py --mode se3` for the
primary table, and Sim(3) as a secondary column for monocular arms where scale
is unobservable. Trajectory/timestamp interpolation: `--method interp`,
`--max-dt 0.02`. RPE deltas `{1,30}` frames. No per-frame re-alignment. Alignment
is computed once over the whole exported trajectory of a run; a run is not split
or re-aligned per segment.

## 6. Failure and exclusion rules (fixed)

* A run is **valid** only if the process exits with code 0 and
  `eval_entry.txt` exists. Non-zero exit or missing artefacts => the cell is
  reported as `CRASH` and counted, never re-run to get a better number.
* No run may be repeated to replace an unfavourable result. A repeat index is
  used exactly once.
* If the trajectory is empty, ATE is reported as `N/A` and coverage still
  reported.
* Datasets that could not be run are reported as `NOT RUN` with the concrete
  blocker. No number is invented for them.

## 7. Timing protocol

Real-time pacing stays ON (`PHOTO_SLAM_NO_PACING` unset), matching production
behaviour, so wall-clock is dominated by sequence length. Per-frame tracking time
comes from `TrackingTime.txt` measured around the `Track*` call only. All runs
are serialised (one process at a time) so the GPU is idle and timings are
comparable.

## 8. Provenance recorded for every run

`meta.txt` in each run directory contains: full command line, environment
subset, ISO start/end time, exit code, and `sha256` of the binary,
`libORB_SLAM3.so`, the SLAM config, and the association file, plus the git HEAD
of the evaluation tree.

## 9. Disclosure: work done before freeze

Exactly one run was executed before this protocol was committed: the smoke test
`/workspace/eval_runs/smoke/mono_t0` (cabinet, mono, arm T0, 1147 frames,
exit 0, 46.5 s). It exists only to verify that the freshly built binary starts,
initialises and writes artefacts. Its outcome did not influence any threshold,
tolerance, scene selection or arm definition in this document, and it is **not**
part of the 36-run batch (the batch uses its own repeat indices).
