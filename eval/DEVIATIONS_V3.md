# DEVIATIONS, second pass (v3)

Supersedes nothing in `DEVIATIONS.md`; all earlier items stand. Items here were
found after the first manifest was built.

## D10. ETH3D RGB-D depth factor: my generated configs were INVALID_CONFIG

`eval/gen_dataset_configs.py` wrote `RGBD.DepthMapFactor: 1000.0` for ETH3D,
reasoning "uint16 millimetres". That was wrong, and it also **overwrote configs
that already existed in the repository**.

The repository's own config settles it:

```
cfg/ORB_SLAM3/RGB-D/ETH3D/eth3d_desk_3.yaml:RGBD.DepthMapFactor: 5000.0
```

i.e. the official ETH3D depth encoding is the same 1/5000 convention as TUM, so
uint16 depth must be divided by 5000, not 1000. At 1000 every depth would have
been 5× too far, which would have silently destroyed near-field structure.

Actions:
* the generated ETH3D files were removed (`git checkout` restored the tracked
  ones) and are recorded as **INVALID_CONFIG**; they were never executed — the
  scheduler had not reached any ETH3D row;
* `scripts/matrix.py` now points ETH3D **RGB-D** at the official
  `RGB-D/ETH3D/eth3d_<seq>.yaml` for all 11 sequences (all exist);
* ETH3D **mono** needs no depth factor. Official mono configs exist for 6
  sequences; for the other 6 (`mannequin_1`, `mannequin_3`, `plant_scene_1`,
  `sfm_bench`, `sfm_house_loop`, `sfm_lab_room_1`) configs were derived from
  each sequence's own `calibration.txt` plus its actual frame size, with every
  other key copied verbatim from the official mono template. Note the two
  distinct calibrations that this exposed (726.287/354.65/186.47 at 739×458 vs
  726.211/359.20/202.47 at 743×465), i.e. assuming one ETH3D calibration for all
  sequences would also have been wrong.

My earlier statement that Replica/ETH3D "have no configs in the repository" was
false — Replica ships all 16 and ETH3D ships 6 mono + 11 RGB-D. Retracted.

## D11. Scheduler defects found by re-counting from run artefacts

The v2 scheduler was launched but its status counts were taken from the manifest
at creation time, not from the files. Re-counting from `status.txt` files showed
**18 runs with exit code 2** — every TUM RGB-D cell — and one interrupted run.

Cause of the 18: `matrix.py` passed the association file as a *relative* path
while the process CWD was the repository root, so the entry could not open it and
exited 2. Those are **infrastructure failures, not algorithm failures**, and they
are kept as `FAILED` with their real exit code; the retry rule (below) allows one
re-run into a sibling `_r1` directory without touching the failed directory.

Fixes in v3:
* absolute paths for every argument;
* single-instance `flock` — a second scheduler now refuses to start;
* frozen per-run timeout (1800 s), recorded as `TIMEOUT`;
* status derived from artefacts on every manifest rebuild, with `PARTIAL` for a
  directory that has no `status.txt` (never reused);
* retry policy: `FAILED` and `PARTIAL` may be executed once into `<dir>_r1`; the
  original is preserved.

Live status at the moment of the restart, counted from files:

```
COMPLETED  20   FAILED  18   PARTIAL  1   PENDING 438   BLOCKED 15   (total 492)
```

The 15 BLOCKED are 9 `TUM freiburg1_room RGB-D` (no config in the repository and
no same-sensor template to derive one from) and 6 upstream rows (no upstream
binary; T0 is not a substitute).

## D12. The ~20° rotation residual is an alignment-frame degeneracy, not a pose bug

Requested checks were run as synthetic cases in `eval/test_coverage.py`, and all
pass: estimate == GT, a known SE(3) applied to the estimate, a known Sim(3) with
scale 2 recovered as 0.5, 100 Hz GT vs 10 Hz video, quaternion/rotation-only
error recovered as 20.000°, and alignment `R ≈ I` where expected. So the
quaternion convention and the pose direction used here are self-consistent.

The 20° on real data is a property of the *global* alignment, not of tracking:
on `large_cabinet/mono/T0/rep1` the alignment-invariant RPE rotation error is
**0.70° rms at Δ=30** while the global-alignment rotation residual is **20.5°**
median. A single position-only Umeyama fit cannot determine rotation about a
near-planar path's normal, so the residual lands entirely in the global frame.

Consequence, unchanged on purpose: the frozen thresholds stay as frozen. The
joint 0.20 m + 5° gate therefore reads 0.000 for these cells, and it is reported
as such next to the translation-only column, the lower bound and the
undetermined fraction. No metric was redefined and no threshold was relaxed to
make the number look better.
