# Complete pose products: what is recorded, and why v_final_1 was not enough

## The gap in v_final_1 (measured, not inferred)

A completed v_final_1 run directory (`/workspace/eval_runs/final_runs/ETH3D/desk_3/mono/P1/rep1`)
contained exactly:

| file | what it holds |
|---|---|
| `final_state.csv` | **4 columns** `frame,timestamp,state,state_name`, 2061 rows - no pose, no map |
| `CameraTrajectory_TUM.txt` | 849 rows = only the map that `SelectExportMap` picked |
| `KeyFrameTrajectory_TUM.txt` | 235 rows, same single map |
| `TrackingTime.txt`, `stdout.log`, `eval_entry.txt`, `meta.txt`, `status.txt` | timings, log, identity |

and its own log stated the loss explicitly:

```
[System] SaveTrajectoryTUM: frame entries by reference-KF map: map0=59 map2=849 map3=698 mapTotal=1606
```

Three maps held poses for 1606 of the 2061 input frames; 849 were written. The
other 757 frame poses (`map0` + `map3`) were discarded at exit, and nothing on
disk allowed them to be recovered: `final_state.csv` has no pose and no map id,
and the per-map split existed only as that aggregate log line. Map lifecycle was
not recorded either - creation/switch/reset were only obtainable if
`PHOTO_SLAM_MAP_EVENTS_CSV` was set, which the matrix did not set, and **merges
were not emitted at all**.

So the v_final_1 artefacts answered "which frames were OK" but not "what pose did
each frame have, in which map, and what happened to the maps". That is the
contract that was open.

## What v_final_2 records (all of it read-only)

| file | columns / rows | added by |
|---|---|---|
| `online_poses.csv` | `frame,timestamp,state,state_name,pose_valid,map_id,ref_kf_id,frame_id,tx,ty,tz,qx,qy,qz,qw`, one row per input frame | entry loop, right after `Track*` returns |
| `map_events.csv` | `seq,event,map_id,kf_id,reason`; events `MAP_CREATE`, `MAP_MERGED`, `MAP_STORED`, `MAP_SWITCH`, `MAP_CURRENT`, `MAP_RESET` | existing `MapEventLog.h` gate, now pointed at the run directory by the entry; `MAP_MERGE` added at both map-discard sites in `LoopClosing` (`MergeLocal`, `MergeLocalWithLine`) |
| `maps_summary.csv` | `map_id,is_in_use,is_bad,n_keyframes,n_mappoints,n_maplines,n_exported_kf,traj_file` | `System::SaveAllMapsTrajectoryTUM` |
| `traj_map<id>.txt` | TUM rows (ts, t, q) of that map's **final** keyframe poses, post-BA | same |
| `final_state.csv` | unchanged 4 columns (kept so existing consumers are unaffected) | - |
| `CameraTrajectory_TUM.txt`, `KeyFrameTrajectory_TUM.txt` | unchanged single-map export (kept for continuity) | - |

`pose_valid=1` means the state was OK and the transform is finite; only those
rows are poses of the frame at the time it was tracked. Rows are never re-derived
or corrected afterwards. `map_id`/`ref_kf_id` come from the frame's own reference
keyframe, so rows from different world frames are never silently stitched.

Online vs final are therefore **separately saved**: `online_poses.csv` is the
pose the system had when the frame was processed; `traj_map<id>.txt` is the
post-optimisation keyframe trajectory of each map. They are evaluated separately
(`coverage.py` consumes the online stream; per-map files are compared against GT
per map).

## Verifying run (v_final_2, ETH3D desk_3 mono P1)

`/workspace/eval_runs/contract_v2/ETH3D/desk_3/mono/P1/rep1`, exit 0:

| check | result |
|---|---|
| `final_state.csv` | 2061 rows, 4 columns |
| `online_poses.csv` | **2061 rows** (one per input frame), 15 columns, `pose_valid=1` on 1648 |
| `map_events.csv` | 8 events: `MAP_CREATE x4`, **`MAP_MERGE` (3 -> 1)**, `MAP_STORED`, `MAP_SWITCH`, `MAP_CURRENT` |
| `maps_summary.csv` | 3 maps: 32 / 244 / 147 keyframes |
| `traj_map*.txt` | 32 / 244 / 147 rows, each equal to its `n_exported_kf` |
| vs v_final_1 | 423 keyframe poses exported in total instead of 244; 757 previously lost frame poses now present |

`eval/acceptance.py` performs exactly these checks mechanically (A: final_state
schema, B: one online pose per input frame with map attribution, C: map events
including a merge when one happened, D: every map's trajectory exists with the
row count it claims, E: the two per-frame files agree on frame/timestamp/state)
and prints PASS/FAIL per run into `ACCEPTANCE_ARTEFACTS.csv`. On this run:
`A..E all PASS`.

## Notes that matter for reading map ids

`LoopClosing` ends a merge with `pMergeMap->ChangeId(pCurrentMap->GetId())`: the
surviving map **takes the id of the map that was absorbed**. So a `map_id` in a
later row can be the id of a map that was discarded earlier. The `map_events.csv`
sequence is the authority for reconstructing that history; the id alone is not.
For `MAP_MERGE` rows the `kf_id` column carries the *absorbing* map id, because a
merge has two maps and the schema has one id field - this is recorded here rather
than left to inference.

## Why these additions cannot change results

They read `Frame::GetPose()`, `Frame::mpReferenceKF`, `Map::GetAllKeyFrames()`
and append to a CSV. No residual, threshold, feature count, initialisation,
rescue rule or accept decision is touched, and nothing in the library reads any
of these files back. The re-measured line-edge counts of `eval/EDGE_COUNTS.md`
are unaffected: the same build path produced them, and the diagnostic switch is
unchanged.
