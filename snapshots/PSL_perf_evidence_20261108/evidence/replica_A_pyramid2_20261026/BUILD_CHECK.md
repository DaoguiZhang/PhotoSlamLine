# Independent source snapshot build check

Date: 2026-10-26

Snapshot source: `/tmp/psl-source-room1-pyramid2-20261026`  
Separate build tree: `/tmp/psl-source-room1-pyramid2-20261026-build`

Checks performed:

1. The snapshot tree was scanned for forbidden artifacts: no `build/`, `bin/`,
   `lib/`, `results/`, `evidence/`, images, PLY, checkpoints, archives or
   compiled objects were present.
2. CMake and build-script references were inspected. The only `.so` paths are
   generated ORB-SLAM3/DBoW2/g2o outputs produced by the build sequence.
3. Rebuilt serially in the independent directory:
   - `ORB-SLAM3/Thirdparty/DBoW2`
   - `ORB-SLAM3/Thirdparty/g2o`
   - `ORB-SLAM3/Thirdparty/Sophus`
   - `ORB-SLAM3`
   - top-level target `replica_rgbd`

Result: all builds succeeded.

Freshly built `replica_rgbd` SHA256:

`d8c5f105061ec204f147e634a7d1e3440fbd27c63b948fe2c65213b5cba05147`

The frozen training binary remains untouched:

`5919857e7a89947b2c5256a8d47ccaf7ca6952f65a9f96218754543d9989d6ae`

`ORBvoc.txt` is runtime data and was not part of this build check. The
snapshot build does not require the omitted vocabulary archive.
