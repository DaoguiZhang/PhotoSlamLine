# Real line-edge counts per arm (evidence, not booleans)

Input: `freiburg1_room` monocular, 1362 frames, Release build
`lib_rel 8d38eb66aa17bb0848e81f78bec04b0325dad09f35928f4b841f97be50bf15cf`
(loaded via `entry_rel`, binding verified with `ldd`: `libORB_SLAM3.so =>
/eval_builds/v1/lib_rel/...`, `libg2o.so => /eval_builds/v1/g2o_rel/...`),
`PHOTO_SLAM_DEBUG_LINE_EDGES=1`, `PHOTO_SLAM_PO_LINE=1`, `PHOTO_SLAM_LBA_LINE=1`.

Counts are summed over every `[LineEdgeDiag]` line actually emitted by the
optimizers; `lnEdges` is the number of line edges the code added to that
optimizer, `lnIn` the number that survived as inliers.

| arm | exit | PO diag calls | sum PO lnEdges | sum PO lnIn | LBA diag calls | sum LBA lnEdges | LBA_MERGE |
|---|---|---|---|---|---|---|---|
| T0 | 0 | 0 | 0 | 0 | 0 | 0 | not called |
| P1 | 0 | 1934 | **0** | 0 | 0 | 0 | `gated=1 lineEdges=0 nML=567` |
| P2 | 0 | 1955 | **22850** | 15408 | 253 | **113037** | `on nML=249` |

Interpretation:

* **T0** emits no PO/LBA line diagnostics at all, never prints
  `lnTrackedNonNull` (the line tracker is not running) and logs zero references
  to MapLine. The pure-point entry therefore does not enter the line path.
* **P1** shows the line frontend alive (`NL`, `lnTrackedNonNull` on all 1934
  calls) with **zero** line edges in pose optimisation, and the merge BA was
  reached with 567 MapLines in hand and correctly gated to zero. The gate is not
  vacuous: there were MapLines to add and none were added.
* **P2** adds real edges: 22 850 in pose optimisation (15 408 inliers) and
  113 037 across 253 local-BA calls, plus an active merge BA.

Correction: an earlier commit message labelled
`8d38eb66aa17bb0848e81f78bec04b0325dad09f35928f4b841f97be50bf15cf` as the
Release library; it is the **ASan** library. The Release library hash is the one
quoted above. History is not rewritten (no force push); this file is the
correction of record.
