# Frozen version status

`v_final_1` (documented in `eval/FREEZE.md`) is **superseded and must not be used
for new results**. It failed the pose-product contract: `final_state.csv` had no
pose and no map id, `SaveTrajectoryTUM` exported only the selected single map
(849 of 1606 posed frames on ETH3D desk_3), and merges were not recorded at all.
Its 25 attempts are kept as evidence under `/workspace/eval_runs/final_runs/` and
may only support: state statistics, and single-map **conditional** accuracy.
They cannot be quoted as full online correct coverage or as all-map results, and
they are excluded from the v_final_2 statistics.

## v_final_2 (the version the current run set uses)

Frozen 2026-11-04, before the first formal cell of `/workspace/eval_runs/final_runs_v2`.

| artefact | sha256 |
|---|---|
| Release `lib_rel/libORB_SLAM3.so` | `d147774b497577991b5099dfd5c3bca5e3fdfe5b92b30bf1659f456af7c2925c` |
| Release `g2o_rel/lib/libg2o.so` | `e8f7d966cf616a20450932c8e29d8d0d46b8f3d110564a425ee6b54bd8f35716` |
| Release `entry_rel/eval_frontend` | `bedca0ba25ebcd1eb102a2013bdf4f2730f8a987e1bfa69c6d40584b81655ca0` |
| ASan `lib_asan/libORB_SLAM3.so` | `4caadec7cbe152a81c93cd845554837e04348375c6a776fcec7f0a71052b4f6b` |
| ASan `entry_asan/eval_frontend` | `074ff5daa6562aa127283b6627066f7c0940c2fa33040efdb75ee0abb6a67965` |
| `ORB-SLAM3/src/Optimizer.cc` | `13f2a8c7c5dfd54c5f4c51f22feddc39c8c6ffc86cf31981f3dfcb9f95d34d41` |
| `ORB-SLAM3/src/System.cc` | `4d9c6de9e78902c60e0a24255b93d2e3251c46a8f1dce0ae505b2019acfa53cb` |
| `ORB-SLAM3/include/System.h` | `cc410ea62b6e7122a046e5cbe433eb8dd021aeb34a60bf4b6a26698c665998da` |
| `ORB-SLAM3/src/LoopClosing.cc` | `12587d0caa399104c932cdd5baf00690a31a6e1b4705de45d1e03ba3ba1ac9e2` |
| `examples/eval_frontend.cpp` | `ceb9d8a29e5cb4d90d8ec5ec08eabe96ecf59963b2416b5121a59a6f264cdd0d` |
| manifest (492 rows) | `68120c34eec84eca8e6ff6464a536878408a6ee536207d70c885ed59d241c713` |

Unchanged from v_final_1: vocabulary `f8dd027f...`, DBoW2 `7bb660ba...`, cfg tree
`86e96fc9...` (163 files), g2o hashes, evaluator `199862bb...`, both GT rule sets
(`eval/REPLICA_GT.md`), and the whole environment section of `FREEZE.md`.

### What changed between the two versions

Only recording: `online_poses.csv` per input frame with map attribution,
`MAP_MERGE` events at both merge sites, `System::SaveAllMapsTrajectoryTUM` writing
`maps_summary.csv` + `traj_map<id>.txt` for every map, and the entry pointing the
existing map-event CSV at the run directory. See `eval/POSE_PRODUCTS.md` for the
columns, the verification run and the argument that no numeric path is touched.

### Scope of the affected cells

Because the missing products are per-run artefacts, **every** cell of v_final_1 is
affected, including the 25 that ran: none of them carries the new files. The
conservative plan - and the one executed - is therefore to re-run all 486 internal
cells under v_final_2 in a new root, plus the 6 upstream control rows. The old
root is not mixed in.
