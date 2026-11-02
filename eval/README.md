# Cross-dataset front-end evaluation — how to reproduce

All commands assume the root of this worktree (an independent `git worktree` of
the fix branch `fix/line-reproj-triangulate-po-reeval`, commit `d3214194`).

## Build

```bash
# 1. third party (not vendored as binaries, built from source)
cd ORB-SLAM3/Thirdparty/DBoW2/build && cmake .. && make -j12
cd ORB-SLAM3/Thirdparty/g2o/build   && cmake .. && make -j12
# 2. vocabulary
cd ORB-SLAM3/Vocabulary && tar xzf ORBvoc.txt.tar.gz
# 3. ORB-SLAM3 shared library (new variant = line outlier re-evaluation ON)
bash scripts/build_eval_orb.sh
# 4. SLAM-only evaluation entry
bash scripts/build_eval_entry.sh
```

Verified artefacts of the run reported in `eval/REPORT.md`:

| artefact | sha256 |
|---|---|
| `bin/eval_frontend` | `2c7280b9488b88450992633009002322be7ffb5be4a36eae81673fa3f7c554de` |
| `ORB-SLAM3/lib/libORB_SLAM3.so` | `eb716e53f47167bb4c93b6f69ab7ab9fda9ce58fd523ea4ecad4d120e2854580` |

`bin/eval_frontend` links ONLY `libORB_SLAM3.so` + OpenCV (no Gaussian mapper,
viewer, imgui, simple-knn, CUDA rasterizer or torch). It has an embedded
`RUNPATH` pointing at `ORB-SLAM3/lib`, so it does not silently pick up another
build of the library.

## Configuration provenance

Four configs were added, each a **verbatim copy** of the same-dataset,
same-modality template (see `eval/PROTOCOL.md` section 4 for the full audit):

```
cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_cabinet.yaml
cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg3_large_cabinet.yaml
cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_cabinet.yaml
cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg3_large_cabinet.yaml
```

## Associations

`cfg/ORB_SLAM3/RGB-D/TUM/associations/fr3_cabinet.txt` and
`.../fr3_large_cabinet.txt` were generated with

```bash
python3 scripts/gen_assoc.py <ds>/rgb.txt <ds>/depth.txt \
        --max_difference 0.02 --out <assoc.txt>
```

(`associated` counts: 1112/1147 and 984/1011; the tool prints the unmatched
count instead of hiding it).

## Run

```bash
# single run
env PHOTO_SLAM_LINE_MODE=0 PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 \
    PHOTO_SLAM_LINE_RESCUE=0 PHOTO_SLAM_SLAM_ONLY=1 \
    bin/eval_frontend ORB-SLAM3/Vocabulary/ORBvoc.txt <slam_cfg> <ignored> \
        <sequence_dir> <out_dir> {mono|rgbd} [assoc_file]
# full pre-registered batch (36 runs, resumable, refuses to overwrite)
bash scripts/run_batch.sh
```

## Metrics

```bash
python3 eval/metrics.py --root /workspace/eval_runs/cabinet \
        --out /workspace/eval_runs/cabinet_metrics
```

Writes `per_run.csv`, `summary.json`, `summary.md`. Metric definitions are the
frozen ones in `eval/PROTOCOL.md` section 5.
