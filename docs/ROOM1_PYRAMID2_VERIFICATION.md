# room1 pyramid2 configuration fix: verification note

Date: 2026-10-26  
Scope: room1 only. This is a configuration correction, not a new acceleration algorithm.

Eight-scene follow-up: see `docs/REPLICA_A_PYRAMID2_BATCH.md`.

## Source and binary provenance

- Repository source commit used as the base: `ec85e80886e2f0494acb87781e075334bcf2ba58`
- Frozen source snapshot archive SHA256:
  `1c1b0c71c2f4bf88a97f29194e8c911b4acef691260239dae4920fe3dec42027`
- Frozen densify/config patch SHA256:
  `03f4bcb248ba648a541e08cb332166e6be6b01557689678d5ff887f8fc369f51`
- Frozen binary SHA256:
  `5919857e7a89947b2c5256a8d47ccaf7ca6952f65a9f96218754543d9989d6ae`

Core source hashes in the frozen build source:

| file | SHA256 |
|---|---|
| `CMakeLists.txt` | `f3cbad54d0fda137d56d45ddfe3d893bf5e58a80fa2b095d9f2393a761cf1c95` |
| `src/gaussian_mapper_line.cpp` | `30323b7d6ed7d3fe61c6daf9446d1e68585ceaab2f37a1670320c7a81579c68f` |
| `src/gaussian_model_line.cpp` | `09d630eba22d1f30c8fd273c0bb7ef1def46bf063080fb2f90e2fa9358abf3d1` |
| `include/gaussian_mapper_line.h` | `962d6768a50071882564f98bed2b18696a22c0862d57fa0ad253dfdf674073bf` |
| `ORB-SLAM3/src/System.cc` | `bae043d15e25c9e18b765cc6ed64fd2fe5ff147592202fd67bbfd085be65137d` |
| `viewer/imgui_viewer_line.cpp` | `ec8e431aaacd638df496b3e6aaa7358c61bbea03139b7c31cd5e2550393a4ce3` |

## Configuration change

The candidate is the current A8 room1 configuration with exactly one value changed:

```diff
-GausPyramid.num_sub_levels: 3
+GausPyramid.num_sub_levels: 2
```

Files:

- baseline current A8 config: `cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1.yaml`
  - SHA256 `62fadbf35156c1bea52b0965b49be236d59d8260882961bc100ac9dc1fc129f7`
- candidate: `cfg/gaussian_mapper/RGB-D/Replica/A_baseline/room1_pyramid2_candidate.yaml`
  - SHA256 `8e3292bacecaa924820521232381d6f9f161069f64007219fa2648a655ea6462`

No alias compatibility, user-intent settings, loss/LR/densify/optimizer changes were mixed in.

## Six clean runs

All runs used the frozen A binary, seed 0, LINE_MODE=2, LINE_LOOP=0,
GBA deformation off, append off, `no_viewer`, and all diagnostics/checkpoints/
selfcheck/profiler off. All six process exit codes were 0.

| trial | full GS PSNR mean | KF | Gaussians |
|---|---:|---:|---:|
| r1_pyramid3 | 32.7391586725 | 181 | 515835 |
| r1_pyramid2 | 35.5814104297 | 176 | 500226 |
| r2_pyramid2 | 35.5815385220 | 169 | 474977 |
| r2_pyramid3 | 32.1732054365 | 188 | 536884 |
| r3_pyramid3 | 32.0827286671 | 173 | 505445 |
| r3_pyramid2 | 35.2808051164 | 174 | 500578 |

Aggregate:

- pyramid2: `35.4812513560 ± 0.1735915474 dB` (three-run mean ± sample std)
- pyramid3: `32.3316975920 ± 0.3557596246 dB`
- full-run mean improvement: `+3.1495537640 dB`

## Common training-frame PSNR

Frames were paired by `img_name` from each run's existing
`natural_end/cameras.csv`; no KF-index row pairing was used.

| round | common frames | pyramid3 mean | pyramid2 mean | delta (p2-p3) |
|---|---:|---:|---:|---:|
| r1 | 31 | 33.9498299629 | 36.4094483160 | +2.4596183531 |
| r2 | 23 | 32.7472294517 | 35.9627470763 | +3.2155176246 |
| r3 | 35 | 33.2828075409 | 36.0502713885 | +2.7674638476 |

Common frames are not independent repeats. The three round deltas are
descriptive, not a confidence interval.

## Mechanism

Code inspection confirms:

- `num_sub_levels=2`: resolutions `300x170`, `600x340`, then full `1200x680`.
- `num_sub_levels=3`: adds `150x85`, then `300x170`, `600x340`, then full.
- `GausPyramid.sub_level_times_of_use=15` gives 30 low-resolution uses per
  keyframe at 2 levels and 45 at 3 levels.
- Levels are consumed low-to-high; full-resolution training is entered only
  after all per-keyframe sub-level uses are exhausted.

Existing final keyframe-use logs show many keyframes below their sub-level
budget at natural stop, substantially more with pyramid3. The exact causal
path from the extra low-resolution training to lower PSNR remains a mechanism
hypothesis; this snapshot does not claim a direct per-level counter.

## Limits

- Only room1 was verified.
- No claim is made that this configuration is correct for the other seven
  scenes.
- No claim is made that this is a system-level speedup. Runtime changed little
  and model size was smaller in these runs, but KF/Gaussian sets differed.

## Reproduce

The script has no dependency on a local evidence or binary archive.
Supply a built binary, room1 dataset, vocabulary, and a new output directory:

```bash
scripts/reproduce_room1_pyramid2.sh \
  --binary /path/to/bin/replica_rgbd \
  --dataset /path/to/replica/room1 \
  --vocab /path/to/ORBvoc.txt \
  --output /path/to/new/output
```

`--lib-dir` can point at the directory containing the binary's project shared
libraries; it defaults to the binary's directory. The `--orb-config` and
`--gaussian-config` defaults are the repository configs.
