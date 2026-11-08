# densify_from_iter configuration fix and A baseline

Date: 2026-10-23  
Scope: configuration parsing, counting labels, short boundary validation, one
room1 A pair. C/H were not run and no threshold scan / extra refinement was
added.

## 0. Runtime and isolation

- No experiment process was running at the start (`pgrep` was empty).
- All development was done in `/tmp/psl-densify-a`, not in the production
  working tree.
- The pre-change patch, configs, and binaries are in `pre_change/` with
  `SHA256SUMS`.
- Production HEAD remains `ec85e80886e2f0494acb87781e075334bcf2ba58`.
- The final cumulative patch is `densify_config_fix.patch`, SHA-256
  `03f4bcb248ba648a541e08cb332166e6be6b01557689678d5ff887f8fc369f51`.
  It was checked with `git apply --check` against that HEAD and reproduces the
  fixed source.

## 0a. exit 124 root cause and normal exit

This is not a `System::Shutdown()` deadlock and not a slow export.

- The earlier full command was `timeout --signal=TERM --kill-after=20s 600
  bin/replica_rgbd ...` (six arguments, no `no_viewer`).  Mapping and
  natural-end export finished at about 91 s; the process then stayed blocked
  for the remaining ~509 s until the wrapper returned 124.  It created the
  ImGui/GLFW viewer thread.
- With `PHOTO_SLAM_SHUTDOWN_DIAG=1`, `System::Shutdown()` completed in about
  0.3 s: local mapper and loop closer both finished and joined; the mapper
  thread returned after `[PhaseTiming]`; `TotalOperationTime.txt` was never
  reached because main was blocked at `main_before_viewer_join`.
- In the short reproduction the viewer thread printed only
  `viewer_run_enter` and `viewer_glfw_init_done`, then stopped before
  `viewer_window_created`.  It was blocked inside `glfwCreateWindow()`.
  `DISPLAY=:28` exists in this container, but there is no usable X server, so
  a DISPLAY-existence test is not sufficient.
- `gdb -p` was unavailable (`ptrace_scope=1`, no `CAP_SYS_PTRACE`), so the
  evidence is the default-off stage log plus `/proc/<pid>/task/*` state
  capture; no stack is claimed.  Threads were sleeping and the process had been
  stopped at viewer join for over three minutes, long after mapping and
  `System::Shutdown()` completed.
- The fix is lifecycle-only: pass the already-supported `no_viewer` in the
  baseline command, request `pViewer->signalStop()` before joining if a viewer
  exists, and avoid creating a viewer when neither DISPLAY nor
  WAYLAND_DISPLAY is set.  No shutdown, join, export, training, loss, LR, or
  optimizer/densify order was skipped or changed.  Stage logs are off unless
  `PHOTO_SLAM_SHUTDOWN_DIAG=1`.
- Short 400-frame run with `no_viewer`: exit **0**, total operation 15.9385 s.
- Full room1 A/from600 with `no_viewer`: exit **0**, total operation
  **89.0689 s**, no residual process.

## 0b. Actual run conditions

`pair/run_conditions.csv` records the verified and unknown fields.

| arm | LINE_MODE | LINE_LOOP | GBA | seed | viewer | stage/Adam/checkpoint | effective from | status |
|---|---|---|---|---|---|---|---|---|
| clean_a600 | 2 | **0** | `APPLY_CORRECTION=0` | 0 | no_viewer | all off | 600 canonical | clean exit 0 |
| final_a600 | 2 | **0** | `APPLY_CORRECTION=0` | 0 | no_viewer | stage+Adam+checkpoint on | 600 canonical | exit 0, diagnostic |
| from0_full | 2 | **0** | `APPLY_CORRECTION=0` | 0 | omitted; viewer blocked | stage+Adam+checkpoint on | 0 canonical | mapping done, viewer join blocked |
| historical A | 2 (log) | **unknown** | **unknown** | **unknown** | **unknown** | stage on (artifact), Adam unknown, checkpoint on (artifact) | 0 inferred | quality snapshot only |

The clean command explicitly sets `PHOTO_SLAM_LINE_LOOP=0`.  Historical A's
`PHOTO_SLAM_LINE_LOOP` was not logged; when unset the code default is **2**.
It must not be called "same condition" without this qualification.  Historical
A's exact shell command, seed, viewer choice, GBA switch and Adam-diagnostic
state are not preserved and are marked unknown.

## 1. Parser fix (only densify_from_iter semantics changed)

Canonical key:

```text
Optimization.densify_from_iter
```

Resolver rules:

| canonical | legacy `..._` | behavior |
|---|---|---|
| present | absent | use canonical |
| absent | present | use legacy and print `[Config][WARN]` |
| present | present, same value | use canonical, report `legacy_present=1` |
| present | present, different value | print `[Config][ERROR]` and exit 1 |
| absent | absent | use explicit default 500, print warning |

No missing FileNode is converted to zero silently for this setting. The
effective `from`, `until`, and `interval`, their source, and whether the legacy
key was present are printed at startup and written to `runtime_config.txt`.

Parser validation (`parser_tests/`):

| config | actual result |
|---|---|
| canonical only | `from=600`, source `canonical`, legacy=0 |
| legacy only | `from=600`, source `legacy`, warning |
| both, 600 vs 601 | exit code 1, explicit conflict error |
| neither | `from=500`, source `explicit_default_500`, warning |
| room1 | `from=600`, `until=15000`, `interval=100` |

Short boundary run (`runs/short600/`) used the same A settings and stopped at
730 optimizer calls. The first ordinary densify event was at iteration **700**,
not 600:

```text
iteration,kind,rows_before,rows_after,delta,from_iter,from_source,interval
700,ordinary_densify,24016,15029,-8987,600,canonical,100
```

At that event all six parameter TensorImpls were replaced, gradients became
undefined, and `after_step` showed host step 700 with Adam step 699. This is
the existing `backward -> topology replacement -> optimizer.step()` order; the
order was not changed and parent gradients were not copied.

LocalMapping point supplementation is recorded separately in
`localmapping_supplement.csv` (`kind=localmapping_increasePcd`). In the fixed
full run there were 202 such events and 336,336 row additions; they are not
classified as ordinary densification.

## 2. Static configuration-key audit

Files: `static_audit/`.

- 61 gaussian YAML files, 66 keys read by `gaussian_mapper_line.cpp`.
- Room1 files have 8 unique absent keys (the same 8 appear in both
  `RGB-D/Replica/room1.yaml` and `Monocular/Replica/room1.yaml`):
  `Optimization.absolute_max_ratio`, `Optimization.densify_from_iter_`,
  `Optimization.lambda_anisotropy`, `Optimization.line_sample_step`,
  `Optimization.line_sigma`, `Optimization.line_top_k`,
  `Optimization.line_view_weight`, `Optimization.max_anisotropy_threshold`.
- Seven of those are guarded by explicit `.isNone()` defaults; the old
  `densify_from_iter_` is now handled by the resolver above. Room1 has no
  remaining silent-zero key in this list.
- **Important alias finding:** the room1 YAML does contain alternate spellings
  for those seven parameters, but they do not match the code keys. They are
  therefore ignored, and the code default is used. This is not changed in this
  round:

| code reads | YAML alternate spelling | YAML value | actual | bound? |
|---|---|---|---:|---:|
| `Optimization.absolute_max_ratio` | `absolute_max_ratio` | 100.0 | 1500.0 | no |
| `Optimization.lambda_anisotropy` | `lambda_anisotropy` | 0.01 | 0.01 | no (same value) |
| `Optimization.line_sample_step` | `Line.SampleStep` | 0.10 | 0.2 | no |
| `Optimization.line_sigma` | `Line.Sigma` | 3.0 | 3.0 | no (same value) |
| `Optimization.line_top_k` | `Line.TopK` | 3 | 3 | no (same value) |
| `Optimization.line_view_weight` | `Line.ViewWeight` | 2.0 | 2.0 | no (same value) |
| `Optimization.max_anisotropy_threshold` | `max_anisotropy_threshold` | 100.0 | 600.0 | no |

  Material differences are `Line.SampleStep` (0.10 ignored, 0.2 effective),
  `max_anisotropy_threshold` (100 ignored, 600 effective), and
  `absolute_max_ratio` (100 ignored, 1500 effective).  The other aliases
  happen to equal their defaults, so their values are numerically unchanged
  even though they are not bound.  Full detail:
  `effective_config_room1.csv`, `pair/default_key_binding.csv`.
- The alias mismatch is a separate compatibility issue.  It is **not** fixed
  by the densify parser and no parameter value was changed in this round.
- The canonical `Optimization.densify_from_iter` is present in 60/61 configs;
  the only exception is `cfg/gaussian_mapper/Monocular/tandt/1tandt_train.yaml`,
  which is missing 65/66 keys and is an unusable outlier.
- Confirmed other config-key gaps (not changed this round):
  - `Mapper.loop_closure_increased_times_of_use_` is absent from
    `1tandt_train.yaml`; the code reads the trailing-underscore key.
  - `Optimization.weight_line_coherence`, `weight_line_shape_ecc`, and
    `weight_line_shape_ori` are read directly but absent from
    `Stereo/KITTI/KITTI.yaml` and `RGB-D/RealCamera/realsense_rgbd.yaml`.
  - Other missing direct keys in the static table are concentrated in
    `1tandt_train.yaml`; no batch edits were made.

## 2a. Alias compatibility fix (isolated, not in the frozen A)

`alias_compat_fix.patch` implements the same five rules for the seven keys:

1. canonical only -> canonical, source `canonical`;
2. alias only -> alias, warning, source `alias`;
3. both same -> canonical, source `canonical (alias same)`;
4. both conflicting -> `[Config][ERROR]` and exit 1;
5. neither -> explicit default, source `explicit_default`.

Tested configs: `alias_fix/configs/` and `alias_fix/*.log`.

- original alias-only: binds `Line.SampleStep=0.15`, `max_anisotropy_threshold=100`,
  `absolute_max_ratio=100` with alias warnings.
- stable canonical: explicit `0.2`, `600.0`, `1500.0`, source canonical.
- user-intent: explicit `0.10`, `100.0`, `100.0`, source canonical.
- conflict (`Optimization.max_anisotropy_threshold=600` plus alias `100`):
  exit 1 with explicit error.
- same-value canonical+alias: accepted as `canonical (alias same)`.

The alias fix was built in `/tmp/psl-alias-fix`; it does **not** modify the
frozen A binary.  The stable configs in
`cfg/gaussian_mapper/RGB-D/Replica/A_baseline/` use the current clean A actual
values and remove conflicting aliases.  `pair/stable_vs_clean_66.csv` compares
all 66 audited keys: **0 actual-value mismatches**.  The only differences in
the user-intent config are the three intended values in
`pair/user_intent_changes.csv`.

## 3. Update-count semantics

New labels distinguish:

- `training_backward_rounds`: calls to `loss.backward()`.
- `optimizer_step_calls`: calls to `optimizer_->step()` (old
  `frozen_opt_steps_` / old `optimizer_steps` label).
- `adam_advance_total` and `adam_advance_group_*`: groups whose Adam step
  actually increased during that call.

In both full A runs one group set (six active groups) was present. The other
two array slots were empty. The old label `optimizer_steps` is preserved only
as `optimizer_steps_legacy_counter` where historical compatibility is useful;
new progress/meta/CSV labels use `optimizer_step_calls`.

## 4. Controlled room1 A pair (non-complete runs)

The following pair was collected before the viewer/exit fix.  Both processes
completed mapping and wrote snapshots but were terminated at viewer join; these
are retained for the update-count / event analysis, **not** called complete
system runs.  Same source build, config, data, seed 0, `PHOTO_SLAM_LINE_MODE=2`,
`PHOTO_SLAM_LINE_LOOP=0`, `PHOTO_SLAM_GBA_APPLY_CORRECTION=0`,
`PHOTO_SLAM_APPEND_STEPS=0`, backend A. The only active variable was the
effective `Optimization.densify_from_iter`:

- `from0`: canonical key set to 0, actual source `canonical`.
- `from600`: room1 config, actual source `canonical`.

| arm | from | calls | backward rounds | Adam advances/group | ordinary densify | KFs | Gaussians | line Gaussians | mapping s | SLAM_end s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| from0 | 0 | 9538 | 9538 | 9443 | 95 | 176 | 530674 | 708 | 65.288 | 85.454 |
| from600 | 600 | 10223 | 10223 | 10127 | 96 | 183 | 512388 | 891 | 69.285 | 91.282 |

In both arms `calls - Adam_advance = ordinary_densify_events` exactly
(95 and 96). Each ordinary densify replacement leaves the following optimizer
step with no gradient for all six groups. This is why the old
`optimizer_steps` number must not be read as a real parameter-update count.

Model-row event accounting (event-level net row changes, not unique Gaussian
IDs):

| arm | ordinary positive net | ordinary negative net | LocalMapping events / added rows | prune events / deleted instances |
|---|---:|---:|---:|---:|
| from0 | 289,657 | 81,626 | 190 / 321,032 | 190 / 835,171 |
| from600 | 287,489 | 113,083 | 202 / 336,336 | 192 / 838,455 |

Quality uses the existing native C++ `eval_checkpoint_snapshot` flow and
`psnr_gaussian_splatting`, not a new evaluator. All-frame values and
common-image values are kept separately because keyframe selection differs.

| checkpoint | from0 all | from600 all | common n | from0 common | from600 common | delta common |
|---|---:|---:|---:|---:|---:|---:|
| frame_500 | 34.7728 | 31.4617 | 10 | 37.4776 | 33.6427 | -3.8349 |
| frame_1000 | 30.8895 | 31.1656 | 21 | 32.2587 | 31.9146 | -0.3441 |
| frame_1500 | 30.0698 | 30.4656 | 29 | 31.2990 | 30.1811 | -1.1179 |
| frame_end | 30.7778 | 31.9491 | 31 | 32.6059 | 33.1364 | +0.5304 |
| natural_end | 30.6067 | 31.4838 | 31 | 32.4847 | 32.7827 | +0.2980 |

Natural-end common-image details: 22 better / 9 worse for `from600`, median
delta `+0.5648` dB. The all-frame difference is `+0.8771` dB but is not a
paired comparison because the two runs have different keyframe/image sets.
Frame 0 is not a stable indicator: from0 `35.1975` dB, from600 `31.8193` dB.

The historical pre-change A result is also preserved:

| arm | calls | Adam/group | inferred ordinary events | KFs | Gaussians | native PSNR all | common with fixed | common PSNR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| historical legacy A | 9770 | 9673 | 97 | 179 | 496296 | 30.9933 | 48 | 31.5356 |

That run used the same room1 config and seed but the old parser, so its
effective `from` was 0. Its event-level created/deleted log was not preserved;
the 97 count is inferred from `calls - Adam` and is not presented as a
measured event table. The historical `/tmp/psl-room1-control/h-round3/A`
snapshot and offline PSNR files remain unchanged.

Diagnostic timing is not a production speed claim. Both non-complete runs had
stage counters and event logs enabled; `from600` used 685 more optimizer calls
and took `+4.0 s` mapping / `+5.8 s` SLAM_end versus the controlled from0 run.

### 4a. Diagnostic-enabled A/from600 run

This run had stage counters, Adam/topology diagnostics and online checkpoints
enabled.  It is complete (exit 0), but is not the clean baseline.  Same config,
data, seed, backend A, and training switches, with `no_viewer`:

| field | value |
|---|---:|
| effective from / source | 600 / canonical |
| first ordinary densify | iteration 700 |
| optimizer_step_calls | 9655 |
| training_backward_rounds | 9655 |
| Adam advances/group | 9565 |
| ordinary densify events | 90 |
| KFs | 177 |
| Gaussians | 488209 |
| line Gaussians | 1069 |
| mapping seconds | 65.674 |
| SLAM_end seconds | 86.498 |
| total operation seconds | 89.0689 |
| native PSNR all 177 frames | 30.8196 dB |

`9655 - 9565 = 90`, again exactly one Adam advance lost per ordinary densify
replacement.  Event-level row changes: ordinary positive net 263651, ordinary
negative net 98295, LocalMapping 197 events / 321184 added rows, 181 prune
events / 778914 deleted instances.

Quality against the controlled from0 run, joined by `img_name`:

| reference | n common | reference common | final A common | delta | median | better/worse |
|---|---:|---:|---:|---:|---:|---:|
| from0 | 31 | 32.1697 | 31.2581 | **-0.9116** | -0.9672 | 13 / 18 |
| historical old-parser A | 25 | 32.8833 | 31.5185 | **-1.3648** | -0.2549 | 11 / 14 |

This diagnostic-enabled run is below both from0 and historical A on common
images.  Its all-frame mean (30.8196 dB) is above the controlled from0
all-frame mean (30.6067 dB) but below historical A (30.9933 dB).  The frame
sets and online keyframe timing differ, so this is descriptive only.

### 4b. Truly clean A/from600 run

The clean run disabled stage counters, Adam/topology diagnostics, online
checkpoints, shutdown diagnostics, selfcheck and profiler.  No training-time
scoring or extra checkpoint was performed.  Only normal training and final
model/camera/trajectory/native-PSNR export remained.

| field | value |
|---|---:|
| exit code | 0 |
| effective from / source | 600 / canonical |
| first ordinary densify | 700 (verified by the short-run event log; clean full log is intentionally silent) |
| optimizer_step_calls | 10212 |
| training_backward_rounds | 10212 |
| Adam advances/group | 10116 |
| calls - Adam | 96 |
| KFs (natural snapshot) | 185 |
| Gaussians | 520255 |
| line Gaussians | 390 |
| mapping seconds | 70.478 |
| SLAM_end seconds | 85.144 |
| total operation seconds | 88.4992 |
| native PSNR all 185 frames | 32.1042 dB |

The integer-millisecond `total_mapping_time.txt` remains a sum of training
intervals; it is not a complete update-time boundary.  The complete program
timing is `TotalOperationTime.txt = 88.4992 s` (tracking + backend + export).

Quality by `img_name`:

| reference | n common | reference common | clean A common | delta | median | better/worse |
|---|---:|---:|---:|---:|---:|---:|
| from0 | 24 | 31.9078 | 33.0105 | **+1.1027** | +0.7705 | 18 / 6 |
| historical old-parser A | 31 | 32.4065 | 33.1703 | **+0.7638** | +0.7956 | 23 / 8 |
| diagnostic-enabled final_a600 | 26 | 31.5049 | 33.0082 | **+1.5033** | +1.4408 | 17 / 9 |

The clean A is **not below** the historical result on common images in this
run, so no old/new binary bisect is justified.  The difference from the
diagnostic-enabled run is an online-run observation: keyframe selection and
timing differ, and it is not evidence that diagnostics caused the quality
change.  Historical A remains a quality observation, not a strict baseline,
because its LINE_LOOP, seed, viewer, GBA and diagnostic switches are unknown.

### 4c. Three clean A repeats and statistics

The frozen binary/config were reused with identical explicit environment; no
result was discarded.  All three runs used `no_viewer` and all diagnostics off.

| run | exit | PSNR all | KFs | Gaussians | line | step calls | Adam/group | calls-Adam | mapping s | total s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| r1 | 0 | 32.1042 | 185 | 520255 | 390 | 10212 | 10116 | 96 | 70.478 | 88.4992 |
| r2 | 0 | 31.3607 | 183 | 549503 | 705 | 10214 | 10118 | 96 | 70.646 | 88.5939 |
| r3 | 0 | 31.1281 | 172 | 498388 | 1558 | 10094 | 10000 | 94 | 69.301 | 85.7869 |
| mean | 0 | **31.5310** | 180 | 522715 | 884 | 10173.3 | 10078.0 | 95.3 | 70.142 | 87.6267 |
| std | 0 | **0.5099** | 7 | 25646 | 604 | 68.7 | 67.6 | 1.15 | 0.733 | 1.594 |

Common-frame result (intersection of all three runs, **16 images**; these are
not independent repetitions):

| run | common PSNR |
|---|---:|
| r1 | 33.6373 |
| r2 | 33.3683 |
| r3 | 33.0946 |
| mean / std | **33.3668 / 0.2714** |

Pairwise common results versus references are in `pair/clean_a600_vs_refs.csv`.
The three runs vary substantially in KF selection and model size; the PSNR
standard deviation is about 0.51 dB.  This is a normal-run variability result,
not a significance test.

## 5. Direct answers

1. **Is the configuration actually in effect?** Yes. Room1 now reports
   `densify_from_iter=600 source=canonical`; the short run's first ordinary
   densify is at 700; the legacy/missing/conflict cases behave as specified.
2. **Measured effect of early densification?** Across the three clean repeats,
   all-frame PSNR is `31.531 ± 0.510 dB` (mean ± sample SD); common-intersection
   PSNR is `33.367 ± 0.271 dB` (16 images, not independent replicates).  The
   first run was above from0/historical on common images, but r2/r3 moved in
   both directions.  There is no stable code-regression signal, so no old/new
   binary bisect is justified.
3. **Can the fixed A be a baseline?** Yes.  All three clean runs exit 0,
   join mapper/SLAM threads, export model/cameras/trajectories/native PSNR,
   and reconcile calls minus per-group Adam (`96, 96, 94`).  Use the frozen
   binary, stable canonical config and `no_viewer`; keep diagnostics off.
   Historical A remains an observation because its line-loop/seed/viewer/GBA
   settings are unknown.
4. **Alias compatibility?** The isolated alias patch implements canonical /
   alias / conflict / default rules and passes the four parser cases.  Stable
   canonical config matches all 66 clean-A actual values; user-intent changes
   only `absolute_max_ratio`, `max_anisotropy_threshold`, `line_sample_step`.
5. **Remaining confirmed key mismatches?** Room1 has no remaining silent-zero
   densify key after the resolver. Static gaps remain in
   `1tandt_train.yaml` (65/66 keys, including
   `Mapper.loop_closure_increased_times_of_use_`) and in the direct line-weight
   reads of `KITTI.yaml` and `realsense_rgbd.yaml`. These are recorded, not
   batch-fixed.  C/H remain experimental and were not run.

## 6. Artifacts

- `densify_config_fix.patch`
- `artifact_manifest.sha256` (final binary/library, source, patch, snapshot
  hashes)
- `pre_change/` (before-change patch, configs, binaries, `SHA256SUMS`)
- `parser_tests/` (canonical/legacy/conflict/missing logs)
- `runs/short600/` (boundary at 700, `densify_events.csv`,
  `localmapping_supplement.csv`, `adam_topo.csv`)
- `runs/from0_full/`, `runs/from600_full/` (non-complete viewer-hang runs;
  logs, runtime config, event CSVs, stage counters, checkpoint metadata,
  offline PSNR summaries)
- `runs/final_a600/` (diagnostic-enabled exit-0 A/from600 run)
- `runs/clean_a600/` (clean exit-0 A/from600 baseline)
- `runs/clean_a600_r2/`, `runs/clean_a600_r3/` (repeat summaries)
- `pair/clean_a600_repeats.csv`, `pair/clean_a600_vs_refs.csv`
- `alias_fix/` (parser cases, configs, logs)
- `alias_compat_fix.patch` (isolated alias compatibility patch)
- `pair/stable_vs_clean_66.csv`, `pair/user_intent_changes.csv`
- `effective_config_clean_a600.csv`, `effective_config_room1.csv`
- `shutdown/` (stuck viewer logs, stage evidence, short/full exit-0 logs)
- batch script: `scripts/run_replica_A_baseline.sh`
- stable per-scene configs:
  `cfg/gaussian_mapper/RGB-D/Replica/A_baseline/`
- `pair/quality_common_by_img.csv`, `pair/controlled_quality_curve.csv`,
  `pair/natural_end_common_by_img.csv`, `pair/pair_summary.csv`,
  `pair/clean_a600_quality.csv`, `pair/run_conditions.csv`,
  `pair/model_events.csv`, `pair/historical_legacy_A.csv`
- `static_audit/` (`config_key_presence.csv`, `key_summary.csv`,
  `room1_missing_keys.csv`, `summary.md`, script)

## 7. Room1 defaults and final command

Room1's seven code-read keys are missing under the exact names the code
requests, but the YAML has alternate spellings that are currently ignored:

| code read key | YAML alternate | YAML value | actual | source | bound |
|---|---|---|---:|---|---|
| `Optimization.absolute_max_ratio` | `absolute_max_ratio` | 100.0 | 1500.0 | explicit default | no |
| `Optimization.lambda_anisotropy` | `lambda_anisotropy` | 0.01 | 0.01 | explicit default | no (same) |
| `Optimization.line_sample_step` | `Line.SampleStep` | 0.15 (production YAML says 0.10) | 0.2 | explicit default | no |
| `Optimization.line_sigma` | `Line.Sigma` | 3.0 | 3.0 | explicit default | no (same) |
| `Optimization.line_top_k` | `Line.TopK` | 3 | 3 | explicit default | no (same) |
| `Optimization.line_view_weight` | `Line.ViewWeight` | 2.0 | 2.0 | explicit default | no (same) |
| `Optimization.max_anisotropy_threshold` | `max_anisotropy_threshold` | 100.0 | 600.0 | explicit default | no |

The complete 66-key actual/source table for the frozen clean run is
`effective_config_clean_a600.csv`; production room1 is separately in
`effective_config_room1.csv`.  `Optimization.densify_from_iter` is effective
as 600, source `canonical`; `densify_until_iter=15000` and
`densification_interval=100` are from YAML.

Validated command (the `no_viewer` argument is required in this headless
container; omitting it makes `glfwCreateWindow()` block, which is the exit-124
cause):

```bash
F=/workspace/code/PSL-perf/evidence/frozen_A_clean_20261023
ROOT=/workspace/code/PSL-perf
OUT=/tmp/room1_a600_clean
mkdir -p "$OUT"
LD_LIBRARY_PATH="$F/bin" \
env -u CUDA_LAUNCH_BLOCKING \
  -u PHOTO_SLAM_STAGE_COUNTERS \
  -u PHOTO_SLAM_ADAM_TOPO_DIAG_DIR \
  -u PHOTO_SLAM_ONLINE_CHECKPOINT_DIR \
  -u PHOTO_SLAM_SHUTDOWN_DIAG \
  -u PHOTO_SLAM_RASTER_AGG_SELFCHECK \
  -u PHOTO_SLAM_RASTER_PROF \
  -u PHOTO_SLAM_GPU_PROF \
  PHOTO_SLAM_LINE_MODE=2 PHOTO_SLAM_LINE_LOOP=0 \
  PHOTO_SLAM_GBA_APPLY_CORRECTION=0 PHOTO_SLAM_APPEND_STEPS=0 \
  PHOTO_SLAM_SEED=0 PHOTO_SLAM_RASTER_BACKEND=A \
  "$F/bin/replica_rgbd" \
  "$ROOT/ORB-SLAM3/Vocabulary/ORBvoc.txt" \
  "$ROOT/cfg/ORB_SLAM3/RGB-D/Replica/room1.yaml" \
  "$ROOT/cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1.yaml" \
  /workspace/code/SEGS-SLAM/datasets/replica/room1 \
  "$OUT/out" \
  no_viewer
```

Observed clean run: exit 0, total operation 88.4992 s, KFs 185, Gaussians
520255, native PSNR 32.1042 dB all-frame.  The model/PSNR files are local
evidence, not Git content.  C/H remain experimental options and were not run
in this round.

## 8. Replica RGB-D A batch

Script: `scripts/run_replica_A_baseline.sh`.

Validated checks:

- `CHECK_ONLY=1` passed for all eight scenes; each has 2000 frames, explicit ORB
  config, explicit stable Gaussian config and output directory.
- Full room1 script run: exit 0, elapsed 93 s, result:
  `results/replica_A_baseline_script_check_20261023/replica_rgbd_0/room1`.

The script reads frozen A only, uses `no_viewer`, turns all diagnostics and
additional training off, records the real exit code, elapsed seconds, hashes
and per-scene directory, and never invokes an evaluation project.

Full batch command:

```bash
BATCH_TAG=$(date +%Y%m%d_%H%M%S) \
  /workspace/code/PSL-perf/scripts/run_replica_A_baseline.sh
```

Per-scene output:

```text
results/replica_A_baseline_<BATCH_TAG>/replica_rgbd_0/<scene>
```

Failures are recorded in `summary.csv` with non-zero exit and are not rewritten
as success.  Existing result directories are not overwritten by the frozen
archive.

C/H remain experimental; no C/H run was made.

## 9. Eight-scene batch result

Executed serially with the frozen A binary and per-scene stable canonical
configs:

```bash
BATCH_TAG=20261023_A8 \
  /workspace/code/PSL-perf/scripts/run_replica_A_baseline.sh
```

All eight scenes exited **0**:

| scene | exit | elapsed s |
|---|---:|---:|
| office0 | 0 | 87 |
| office1 | 0 | 86 |
| office2 | 0 | 97 |
| office3 | 0 | 99 |
| office4 | 0 | 94 |
| room0 | 0 | 108 |
| room1 | 0 | 92 |
| room2 | 0 | 108 |

Result root:

```text
/workspace/code/PSL-perf/results/replica_A_baseline_20261023_A8/replica_rgbd_0/
```

For every scene the following were checked non-empty:

- `*/ply/point_cloud/*/point_cloud.ply`
- `CameraTrajectory_TUM.txt`
- `KeyFrameTrajectory_TUM.txt`
- `Global_Final_Map_Points.obj`
- `TotalOperationTime.txt`
- `run.log`
- `command.txt`
- `natural_end/snapshot.pt`

No PSNR was computed for this batch and no evaluation project was invoked.
`summary.csv` contains `scene,exit_code,log_path,result_dir`; extended hashes
and config paths remain in `summary_detailed.csv`. No experiment process
remains.
