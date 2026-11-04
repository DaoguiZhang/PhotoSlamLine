# Real line-edge counts per arm (evidence, not booleans)

Input: `freiburg1_room` monocular, 1362 frames, Release build
`lib_rel 9a31f4ce87d771f3297cdf56ec43975a17389930034c31660a48c311809a278e`
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

Build identities of record (verified in this session):

| artefact | sha256 |
|---|---|
| Release `lib_rel/libORB_SLAM3.so` | `%s` |
| ASan `lib_asan/libORB_SLAM3.so` | `%s` |
| `entry_rel/eval_frontend` | `60d513316619e5d68de5bca276617b4ef455d575de330faef1b55570dad95977` |
| `entry_asan/eval_frontend` | `a69e6116e74cf8941b4d6ed1966956b7ef42a9874b02756cd3397d6f1e4a5014` |

Correction of record: the commit message of bebd8a16 mislabelled the ASan hash
(`8d38eb66…`) as the Release library. The table above is authoritative. No
force push; history is not rewritten.

Note on the two `LSD` strings that appear in the T0 log: they are configuration
echoes ("-Loaded LSD sampling settings", "LSD Extractor Parameters(to do next):"),
not evidence that the extractor ran. T0's operational evidence is the absence of
`lnTrackedNonNull`, of MapLine references and of any PO/LBA line diagnostic.
