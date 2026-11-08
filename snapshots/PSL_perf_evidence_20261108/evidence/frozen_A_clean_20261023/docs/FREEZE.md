# Frozen A clean baseline

Frozen at: 2026-10-23

- Source: `source/source_snapshot.tar.gz`
- Patch: `docs/densify_config_fix.patch`
- Binary: `bin/replica_rgbd`
- Shared libraries: `bin/*.so`
- Configs: `config/room1_orb.yaml`, `config/room1_gaussian.yaml`
- Command: `docs/command.txt`
- All hashes: `SHA256SUMS`

This archive is the source/binary used by the clean A runs.  It is read-only.
Do not rebuild into this directory.  The alias compatibility work lives in the
separate `alias_compat_fix.patch` / isolated tree and does not replace this
archive.

Normal run requirements:

```text
PHOTO_SLAM_LINE_MODE=2
PHOTO_SLAM_LINE_LOOP=0
PHOTO_SLAM_GBA_APPLY_CORRECTION=0
PHOTO_SLAM_APPEND_STEPS=0
PHOTO_SLAM_SEED=0
PHOTO_SLAM_RASTER_BACKEND=A
viewer: no_viewer
stage/Adam/checkpoint/shutdown/selfcheck/profiler: off
```
