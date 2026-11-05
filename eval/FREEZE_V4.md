# v_final_4 - the version the supplementary runs use

Frozen 2026-11-05, after the two crash fixes and the coordinate-epoch record,
before the first supplementary cell.

## What changed since v_final_2

| change | why |
|---|---|
| `KeyFrame.cpp`: `mvpMapLines.assign(NL, nullptr)` | the vector was never initialised, so `GetKeyLineInfo` indexed an empty vector (SEGV at 0x0). See `eval/CRASH_FIXES.md` |
| `Frame.cc`: `ComputeLineStereoFromRGBD` validates the ROUNDED depth index | `cvRound(rows-0.4)` is `rows`, one row past the depth buffer |
| `MapEventLog.h`: atomic frame index + **coordinate epoch** advanced inside `Emit` under the event mutex | a merge that happens while frames have no valid pose is otherwise invisible to the evaluator |
| `eval_frontend.cpp`: writes `epoch` per frame | the evaluator splits alignment eras on the recorded epoch instead of inferring |
| `eval/mapmeta.py`: eras split on epoch change and on an anchor's next posed frame; anchored frames themselves are excluded | a merge anchored inside a gap no longer collapses to one era |

## Artefact identity

| artefact | sha256 |
|---|---|
| `lib_rel/libORB_SLAM3.so` | `97fcb4b4da635f4498aa02fe265753d31978700f78f42a440635883120ca64a3` |
| `entry_rel/eval_frontend` | `89454bc6041f2822a03dfa6b3cd8caa31776c3eb667ce0b383fb455117ed160c` |
| `lib_asan/libORB_SLAM3.so` | `b48925fa7bd25cd302146c95d983342ba8d14aa1fe3391a8a0befcd8dc2bc65e` |
| `entry_asan/eval_frontend` | `9e4e301fea921b1dfe78286278a20d9a8be9bd79bcfc1a3eea8218935f69f221` |
| g2o (unchanged) | `e8f7d966cf616a20450932c8e29d8d0d46b8f3d110564a425ee6b54bd8f35716` |
| DBoW2 (unchanged) | `7bb660badc49dfd5c43631fc7482cbbb6686bc013586b3a8bb6ecd98d688da0f` |

Unchanged from v_final_2: vocabulary, configs, evaluator, both GT rule sets,
environment. No threshold, residual, feature count, initialisation, rescue rule or
accept state was touched; no merge was disabled and no free was removed.

## Live check that the epoch record is consistent

`ETH3D/sfm_house_loop` rgbd P2, 556 frames, exit 0. Event log:

```
0,MAP_CREATE,0,0,Atlas::CreateNewMap,-1,1
1,MAP_CREATE,1,115,Atlas::CreateNewMap,247,2
2,MAP_MERGE,1,0,LoopClosing::MergeLocalWithLine,549,3
3,MAP_STORED,1,-1,Atlas::ChangeMap,549,3
4,MAP_SWITCH,0,-1,Atlas::ChangeMap,549,3
5,MAP_CURRENT,0,-1,Atlas::ChangeMap,549,3
```

Per-frame record (epoch, map id, validity):

```
frame   0 epoch=1 map=0 valid=1
frame 206 epoch=1 map=0 valid=0   (lost)
frame 247 epoch=2 map=-1 valid=0  (new map created during this frame)
frame 248 epoch=2 map=1 valid=1
frame 549 epoch=3 map=1 valid=1   (merge during this frame; epoch advanced)
```

The event sequence, the per-frame epoch and the evaluator's era split agree:
frames 0-246 are era 1, 247-548 era 2, and frame 549 (the merge frame) is
excluded from both, with era 3 starting at the next posed frame.

## Supplementary set (frozen)

`scripts/supplement.py build` -> `/workspace/eval_runs/supplement_v4`.

* 339 rows = **162 P1 + 162 P2** (all WithLine cells: both crash fixes are on
  that path) **+ 15 T0** cells whose history was UNDETERMINED only because the
  v2 record has no epoch.
* 147 T0 cells are reused from the v_final_2 record (T0 enters neither fixed
  path; the only shared change, `mvpMapLines.assign`, writes a vector T0 never
  reads).
* The 7 crashed cells are inside the 324 P1/P2 rows and are not added again.
* Nothing outside the original 492 plan is included; the builder asserts that.
