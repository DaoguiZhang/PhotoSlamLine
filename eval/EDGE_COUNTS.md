# Line-edge evidence per arm (measured counts, v2)

Supersedes the v1 file. v1 could not answer "how many line edges did the
map-merge bundle adjuster actually add", because the merge site only printed a
gate flag (`lineEdges=on`) and the number of MapLines it was holding. `on` is a
state flag and `nML` counts candidates, so neither is evidence that an edge
exists in the graph. This revision adds one read-only counter at that site and
re-measures; every number below comes from the logs of the runs listed here.

## 1. Instrumentation added (diagnostic only)

`ORB-SLAM3/src/Optimizer.cc`, map-merge bundle adjuster, immediately after the
line inlier test and before the map-update lock:

```
[LineEdgeDiag] site=LBA_MERGE_DONE lnEdgesAdded=N lnActive=M lnToErase=K nML=X enabled=0|1
```

* `lnEdgesAdded` - size of `vpEdgesLineMono`, which is appended to only after
  `optimizer.addEdge()` returned and the endpoint vertices were accepted. This is
  the number of line projection edges actually present in this graph.
* `lnActive` - how many of them g2o still had active (`level()==0`) after the
  final iteration.
* `lnToErase` - how many failed the pre-existing `chi2 > 9.0` inlier test.

The block is behind the existing `IsLineEdgeDiag()` switch
(`PHOTO_SLAM_DEBUG_LINE_EDGES=1`), touches no numeric path, and is skipped
entirely when the diagnostic is off. The pre-existing `site=LBA_MERGE` row is
kept and still reports only the gate.

## 2. Build identities (this measurement)

Nothing is inherited from an older build; the two trees were rebuilt from these
sources with `scripts/build_unified.sh` (`build_rc=0`).

| artefact | sha256 |
|---|---|
| Release `lib_rel/libORB_SLAM3.so` | `d33731eb233da9078a7896e68801dd188da69f60c35f98fcd41fd1fdab9ee860` |
| Release `g2o_rel/lib/libg2o.so` | `e8f7d966cf616a20450932c8e29d8d0d46b8f3d110564a425ee6b54bd8f35716` |
| Release `entry_rel/eval_frontend` | `60d513316619e5d68de5bca276617b4ef455d575de330faef1b55570dad95977` |
| ASan `lib_asan/libORB_SLAM3.so` | `3b93c8b42287ae3b46999911ee65b566b6ecb05f2788c5701ef8bebd340523c9` |
| ASan `g2o_asan/lib/libg2o.so` | `770a184c992d0a362b8e5a2c1feffb570e58c66e21fd3e48d21af6aa76b28462` |
| ASan `entry_asan/eval_frontend` | `a69e6116e74cf8941b4d6ed1966956b7ef42a9874b02756cd3397d6f1e4a5014` |
| shared `DBoW2/lib/libDBoW2.so` | `7bb660badc49dfd5c43631fc7482cbbb6686bc013586b3a8bb6ecd98d688da0f` |
| `ORB-SLAM3/src/Optimizer.cc` | `13f2a8c7c5dfd54c5f4c51f22feddc39c8c6ffc86cf31981f3dfcb9f95d34d41` |

The entry binary is byte-identical to the previous build because
`examples/eval_frontend.cpp` did not change; it is nevertheless bound to the new
library, verified with `ldd`:

```
libORB_SLAM3.so => /workspace/eval_builds/v1/lib_rel/libORB_SLAM3.so
libg2o.so       => /workspace/eval_builds/v1/g2o_rel/lib/libg2o.so
```

**Retraction of record.** Earlier notes and commit message `bebd8a16` referred to
`9a31f4ce...` / `8d38eb66...` as the Release / ASan libraries of the contract
runs, and the message swapped the two. Both are now historical: they describe a
build that does **not** contain the merge counter. Any statement about the
current results must use the hashes in the table above.

## 3. Runs

`freiburg1_room` monocular, 1362 frames, all three arms fed the identical frame
sequence and config (`cfg/ORB_SLAM3/Monocular/TUM/tum_freiburg1_room.yaml`),
`EVAL_ARM=T0|P1|P2`, `PHOTO_SLAM_LINE_MODE=0|1|2`,
`PHOTO_SLAM_PO_LINE=1 PHOTO_SLAM_LBA_LINE=1 PHOTO_SLAM_LINE_RESCUE=0
PHOTO_SLAM_SLAM_ONLY=1 PHOTO_SLAM_DEBUG_LINE_EDGES=1`, fresh output root
`/workspace/eval_runs/edges2/{T0,P1,P2}` (the earlier `edges/` tree is left
untouched as evidence).

## 4. Counts

Sums over every `[LineEdgeDiag]` row emitted by the optimizers of that run.

| arm | exit | PO rows | sum PO `lnEdges` | sum PO `lnIn` | LBA rows | sum LBA `lnEdges` | merge rows | merge `lnEdgesAdded` | `lnActive` | `lnToErase` | `nML` |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T0 | 0 | 0 | - | - | 0 | - | 0 | - | - | - | - |
| P1 | 0 | 1941 | **0** | 0 | 0 | - | 1 | **0** | 0 | 0 | 546 |
| P2 | 0 | 1663 | 16601 | 11281 | 231 | 77861 | 1 | **2816** | 2769 | 48 | 526 |

Raw merge rows, verbatim:

```
P1: [LineEdgeDiag] site=LBA_MERGE lineEdges=gated=1 lineEdges=0 nML=546
P1: [LineEdgeDiag] site=LBA_MERGE_DONE lnEdgesAdded=0 lnActive=0 lnToErase=0 nML=546 enabled=0
P2: [LineEdgeDiag] site=LBA_MERGE lineEdges=on nML=526
P2: [LineEdgeDiag] site=LBA_MERGE_DONE lnEdgesAdded=2816 lnActive=2769 lnToErase=48 nML=526 enabled=1
```

(`lnActive + lnToErase = 2769 + 48 = 2817`, one more than `lnEdgesAdded`, because
one edge whose MapLine was already `isBad()` is counted by neither the erase
vector nor the active set. The three numbers are reported as measured, not
smoothed.)

## 5. What each arm therefore shows

**T0 - line-optimisation diagnostic count is zero.**
The run produced no `[LineEdgeDiag]` row at all (0 PO rows, 0 LBA rows, 0 merge
rows), zero mentions of `MapLine`, and zero mentions of `lnTrackedNonNull` (the
line tracker's per-frame output). The pure-point API `TrackMonocular` was used
and the line code path was never entered. This says nothing about whether point
optimisation ran: T0 tracked normally, exported a trajectory and ended at
`final_state=2` with `ok=914 / 1362`. Because T0 emits no diagnostic row, this
log also carries **no** point-edge counts for T0 - the correct statement is
"line-optimisation diagnostics counted zero", not "no optimisation was
performed". The two `LSD` strings in the log are configuration echoes
(`-Loaded LSD sampling settings`, `LSD Extractor Parameters(to do next):`), not
evidence of execution.

**P1 - the merge gate is exercised, not vacuous.**
The line frontend runs (`lnTrackedNonNull` on all 1941 pose rows, MapLines are
created and matched), pose optimisation adds **0** line edges, and the merge
bundle adjuster is reached once holding 546 MapLines and adds **0** line edges
(`enabled=0`). Zero with 546 candidates in hand is a gate that fired, not a code
path that was never reached.

**P2 - line edges really are added, including in the merge.**
16 601 line edges in pose optimisation (11 281 inliers), 77 861 across 231
local-BA calls, and 2 816 in the single map-merge bundle adjuster, of which 2 769
stayed active and 48 were rejected by the inlier test.
