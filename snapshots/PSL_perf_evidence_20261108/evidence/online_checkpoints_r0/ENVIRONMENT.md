# Online A/C checkpoint run environment (r0)

Both runs used the same source/binary/config and seed; only raster backend changed.
Selfcheck, raster profiler and GPU profiler were unset.

```bash
PHOTO_SLAM_LINE_MODE=2
PHOTO_SLAM_LINE_LOOP=0
PHOTO_SLAM_GBA_APPLY_CORRECTION=0
PHOTO_SLAM_APPEND_STEPS=0
PHOTO_SLAM_SEED=0
PHOTO_SLAM_RASTER_BACKEND=<A|C>
PHOTO_SLAM_ONLINE_CHECKPOINT_DIR=<out>/ckpt
PHOTO_SLAM_ONLINE_CHECKPOINT_FRAMES=500,1000,1500,end
```

Command template:

```bash
bin/replica_rgbd \
  /workspace/code/PSL-perf/ORB-SLAM3/Vocabulary/ORBvoc.txt \
  /tmp/psl-room1-control/config/room1_orb.yaml \
  /tmp/psl-room1-control/config/room1_gaussian.yaml \
  /workspace/code/SEGS-SLAM/datasets/replica/room1 \
  <out>/out no_viewer
```

A run: `online-diag-A`, C run: `online-diag-C`.
Checkpoint model snapshots remain in `/tmp/psl-room1-control/online-diag-{A,C}/ckpt/frame_*/snapshot.pt`;
their SHA-256 is in `snapshot_sha256.txt`. Cameras, masks, meta and native
checkpoint PSNR files are copied here.

Snapshot cost per checkpoint (A / C): save 0.027/0.028, 0.043/0.043,
0.067/0.064, 0.090/0.082 s; render 0.561/0.483, 1.024/1.049, 1.822/1.776,
2.876/2.084 s. These are diagnostic timings; not production performance.

Provenance: production baseline `ec85e80886e2f0494acb87781e075334bcf2ba58`;
diagnostic patch `instrumentation.patch` SHA-256
`3d7f56e5b723af488f530c24d29b241165ff8664687e920498f6e69662de863c`.
The PSNR figures are checkpoint-time values from the original C++ renderer and
GS-PSNR function, not a separate post-training offline re-render. Snapshots,
cameras and masks are retained under `/tmp/psl-room1-control/online-diag-{A,C}`
for a later offline re-evaluation.
