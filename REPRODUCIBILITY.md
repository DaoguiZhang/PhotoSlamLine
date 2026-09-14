# Photo-SLAM-L Monocular office2 diagnosis — reproducibility notes

Branch: `exp/mono-office2-diagnosis` (base `25cebfc979`).

## Environment
- Docker (Ubuntu 22.04), RTX 5090 (32 GB), libtorch CUDA, OpenCV 4.10.0.
- Repo: `/workspace/code/Photo-SLAM-L`; datasets: `/workspace/code/SEGS-SLAM/datasets/replica`.
- Build: ORB-SLAM3 in `ORB-SLAM3/build` (libORB_SLAM3.so), root in `build/`
  (links it). `bin/replica_mono` loads `libORB_SLAM3.so` via RUNPATH
  (`LD_LIBRARY_PATH` can override).

## Final recommended configuration (C0)
- Full point-line: `PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1`
  (these are the defaults).
- LBA line parameters at C0 defaults: `LBA_W=1`, `LBA_TAU=9`, `LBA_DELTA=sqrt(5.991)`,
  `LBA_SIGMA_PX=1` (explicit line-noise sigma, pixels).
- Stability: LBA line write-back serialized under `pMap->mMutexMapUpdate`.
- Optional reproducibility: `PHOTO_SLAM_GAUSSIAN_KF_SEED=<uint>` (Gaussian-side only).

## Single run (office2)
```bash
cd /workspace/code/Photo-SLAM-L
PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
PHOTO_SLAM_DEBUG_MONO_INIT=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1 \
bin/replica_mono ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/Monocular/Replica/office2.yaml \
  cfg/gaussian_mapper/Monocular/Replica/replica_mono_diag.yaml \
  /workspace/code/SEGS-SLAM/datasets/replica/office2 <OUT>/ no_viewer
```

## Batch scripts (no-overwrite policy; refuse existing output dirs)
- `scripts/run_c0_verify.sh` — office2 C0 acceptance (1 run).
- `scripts/run_cross_seq_regression.sh` — office0 + room0, fix/prefix interleaved 3x.
- `scripts/run_seed_ablation.sh` — office2 default vs explicit-seed interleaved 3x.

## Evaluation
`scripts/eval_ate_rpe.py <gt_tum> <est_tum> <ate_out> <rpe_out> <ape_out>`:
global Sim(3)-on-translation alignment (Horn/Umeyama), ATE RMSE, fixed-interval
RPE (1 and 10 frames) translation + rotation. GT: `<seq>/pose_TUM.txt` (Twc).

## Known limitations
- Crash root cause is a *suspected* race (no core dump obtainable — apport pipe);
  the map-mutex fix removes the reproducer but is not proof of the exact race.
- Run-to-run ATE variance is NOT eliminated by fixing the Gaussian shuffle: the
  SLAM keyframe-id set already differs across runs upstream. The Gaussian seed is
  a reproducibility control for Gaussian output, not a full-SLAM determinism knob.
- LBA line noise is the existing assumption sigma=1.0 px (endpoints perpendicular
  pixel distance), made explicit; no evidence supports a different value.
