# room1 online A/C diagnostic with model checkpoints (r0)

## Protocol

One pair, same source patch/binary/config/seed, only `PHOTO_SLAM_RASTER_BACKEND`
changed. GBA policy, loss, LR, densify, pyramid and refinement budget unchanged.
Selfcheck and both profilers were unset. Checkpoints were triggered by **input
frame** progress, not by mapper iteration:

`500, 1000, 1500, end(2000)`.

At each checkpoint the run saved:

- full model tensors (`snapshot.pt`, kept in `/tmp`, SHA-256 in
  `snapshot_sha256.txt`)
- per-KF native camera pose, image filename, intrinsics and size (`cameras.csv`)
- undistort masks (`masks.pt`, `mask_cameras.txt`)
- mapper iteration, real optimizer steps, KF count, Gaussian/line counts, SH
  (`meta.txt`)
- native renderer output and per-KF PSNR/GS-PSNR (`psnr*.txt`, `dssim.txt`)
- checkpoint save/render duration (`checkpoint_timing.txt`)

The checkpoint render is a diagnostic operation; its timing is not a production
performance result.

## Progress and update budget

| checkpoint | A input/mapper | C input/mapper | A steps | C steps | A KF | C KF | A GS | C GS | A line | C line | SH |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 500 | 500 / 2136 | 500 / 2828 | 2135 | 2827 | 34 | 38 | 128003 | 164432 | 716 | 1123 | 2/2 |
| 1000 | 1000 / 4656 | 1000 / 5056 | 4655 | 5055 | 83 | 78 | 259784 | 271235 | 925 | 1313 | 3/3 |
| 1500 | 1500 / 7465 | 1500 / 7064 | 7464 | 7063 | 144 | 140 | 391687 | 388051 | 977 | 1844 | 3/3 |
| end | 2000 / 9568 | 2000 / 8338 | 9567 | 8337 | 174 | 171 | 498764 | 442019 | 861 | 1688 | 3/3 |

These are mapper-consumed KFs and real optimizer steps at the moment the input
frame threshold was reached; input progress and mapper progress are not the
same quantity. C had more steps/KFs/Gaussians at 500 and 1000; A had more at
1500 and end.

No LOOP and no GBA event occurred in either run (`Loop Closure Detected=0`,
`[GBA-Apply]=0`). Local BA lines: A=173, C=170; the existing log does not give
their optimizer-step or data-frame consumption points.

## Native GS PSNR

Full checkpoint means (different KF sets, so descriptive only):

| checkpoint | A full | C full | A−C full |
|---|---:|---:|---:|
| 500 | 33.9788 | 36.4518 | −2.4730 |
| 1000 | 31.2610 | 30.8298 | +0.4312 |
| 1500 | 29.2335 | 26.0759 | +3.1576 |
| end | 31.0844 | 28.3053 | +2.7791 |

### Fixed early common set

The fixed set is the **12 frames common at checkpoint 500**; their names are in
`fixed_early_frames.txt`. All 12 remain available at every checkpoint.

| checkpoint | common n | all-common A−C mean | fixed n | fixed A−C mean | fixed A−C median |
|---|---:|---:|---:|---:|---:|
| 500 | 12 | −1.7342 | 12 | **−1.7342** | −1.4701 |
| 1000 | 14 | −0.4933 | 12 | **−0.8803** | −1.0672 |
| 1500 | 20 | +2.7870 | 12 | **+3.2018** | +3.4407 |
| end | 22 | +2.3792 | 12 | **+1.7474** | +1.9184 |

Per-frame values are in `common_frame_psnr.csv`. These are native
`psnr_gaussian_splatting` values from the checkpoint renderer using the saved
training cameras; no new GT alignment or convention conversion was done.

## First measured divergence

There are two different operationalizations, and they must not be conflated:

- **Any A/C difference**: already present at the **first checkpoint, input
  frame 500**. On the fixed 12-frame set C is **1.734 dB higher** than A.
- **C below A**: first measured at **checkpoint 1500** (fixed-set A−C
  `+3.2018 dB`), and still C below A at end (`+1.7474 dB`). This is **not** an
  onset of monotonic C degradation, because C was better at 500 and 1000; it is
  a trajectory crossing.

There is no LOOP/GBA event in either run, so this pair cannot attribute either
observation to an event. Because no pre-500 checkpoint exists, the earliest
formation of any divergence can only be bounded to **before frame 500**, not
localized further.

The gap is not monotonic: C is better at 500/1000, A is better at 1500/end. The
trajectories, KF coverage and update opportunities differ. At 500 C has +692
updates and +4 KFs; at 1500/end A has +401/+1230 updates and more KFs. Do not
infer causality from update count alone; resolution/SH totals, KF usage, queue
state, and local BA consumption are not yet recorded.

## Missing online counters

Not captured by this run, and not inferred from logs:

- per-resolution and per-SH update counts;
- total training pixels;
- per-KF usage counts at each checkpoint;
- map queue length/backlog and consumption progress;
- per-entry creation/pruning counts during the run;
- LocalBA consumption step/frame and camera-change range;
- final stop reason and remaining budget;
- event-time snapshots: neither run produced LOOP/GBA.

The existing logs only support the checkpoint table and the absence of LOOP/GBA.
A second diagnostic run is not taken in this round.

## Evidence

- `summary.csv`, `common_frame_psnr.csv`, `fixed_early_frames.txt`
- `A|C/frame_{500,1000,1500,end}/`
  (`meta.txt`, `cameras.csv`, `masks.pt`, `mask_cameras.txt`, `psnr*`, `dssim`,
  `render_time`, `checkpoint_timing`)
- `run-A.log`, `run-C.log`
- `snapshot_sha256.txt`, `instrumentation.sha256`, `ENVIRONMENT.md`

## Provenance and scoring caveat

- Production baseline commit: `ec85e80886e2f0494acb87781e075334bcf2ba58`
  (`git_commit.txt`).
- Diagnostic source patch: `instrumentation.patch`, SHA-256
  `3d7f56e5b723af488f530c24d29b241165ff8664687e920498f6e69662de863c`.
  It adds input-frame progress, checkpoint save/render and the probe wiring; it
  does not change the GBA policy, loss, LR, densify, pyramid or refinement.
- The reported PSNR values are produced at checkpoint time by the **original
  C++ renderer and GS-PSNR function**, and the per-checkpoint files are saved.
  This round did **not** perform a separate post-training offline re-render
  pass. Model snapshots, cameras and masks are retained in
  `/tmp/psl-room1-control/online-diag-{A,C}/ckpt/frame_*/` for such a later
  offline re-evaluation; the checkpoint renders are diagnostic and their
  timings are not production performance.
