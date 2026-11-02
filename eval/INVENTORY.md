# Dataset inventory and run manifest (fixed before the batch)

Root searched: `/workspace/code/SEGS-SLAM/datasets/{tum,replica,eth3d}`.
Nothing was downloaded and nothing is awaited.

## 1. Six required categories

| # | category | data present | entry support | GT | status |
|---|----------|--------------|---------------|----|--------|
| 1 | Replica monocular | 8 scenes `office0..4`, `room0..2` (`results/frame*.png`) | needs image-index loader (`replica-mono`), not implemented in `eval_frontend` | `pose.txt` + `pose_TUM.txt` present | **NOT RUN (blocked: loader)** |
| 2 | Replica RGB-D | same + `results/depth*.png` | needs `replica-rgbd` loader and a Replica config (`RGBD.DepthMapFactor` for Replica is not `5000`) | same | **NOT RUN (blocked: loader + depth-scale calibration)** |
| 3 | TUM monocular | fr1_desk, fr1_desk2, fr1_room, fr1_xyz, fr2_xyz, fr3_long_office_household, fr3_structure_notexture_{near,far}, fr3_cabinet, fr3_large_cabinet | **yes** (`rgb.txt`) | `groundtruth.txt` | **RUN (cabinet, large_cabinet)** |
| 4 | TUM RGB-D | same set + association files | **yes** (`assoc` arg) | `groundtruth.txt` | **RUN (cabinet, large_cabinet)** |
| 5 | ETH3D monocular | 11 sequences (`desk_3, mannequin_1, mannequin_3, planar_2, planar_3, plant_scene_1, sfm_bench, sfm_garden, sfm_house_loop, sfm_lab_room_1, table_7`), each with `rgb.txt` | **yes** (`rgb.txt`) but no ETH3D config was prepared and no run was made | `groundtruth.txt` | **NOT RUN (blocked: config + schedule)** |
| 6 | ETH3D RGB-D | same, `associated.txt` present | **yes** (`assoc` arg) | same | **NOT RUN (blocked: config + schedule)** |

Only categories 3 and 4 were executed. The other four are reported as NOT RUN
with the concrete blocker; no number is invented for them. This is a scope
shortfall against the request, recorded here rather than papered over.

## 2. Executed batch (36 runs)

`2 scenes x 2 modalities x 3 arms x 3 repeats`:

| scene | dataset path | mono frames | rgbd pairs |
|-------|--------------|-------------|------------|
| cabinet | `datasets/tum/rgbd_dataset_freiburg3_cabinet` | 1147 | 1112 of 1147 RGB (35 unmatched, reported) |
| large_cabinet | `datasets/tum/rgbd_dataset_freiburg3_large_cabinet` | 1011 | 984 of 1011 RGB (27 unmatched) |

Run root: `/workspace/eval_runs/cabinet/<scene>/<modality>/<arm>/rep<k>/`.
36 cells, no substitutions, no re-runs.

## 3. Required-but-absent items, stated explicitly

* No upstream (original Photo-SLAM) arm was executed: the upstream binary was
  not built in this round, so the "upstream comparison" part of the request is
  **NOT DONE**. T0 must not be read as a stand-in for it.
* No Replica or ETH3D numbers exist at all.
* n = 3 per cell; medians and ranges only, no significance claims.
