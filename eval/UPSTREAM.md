# Upstream Photo-SLAM control (the 6 non-internal rows)

## What "upstream" means here

The 6 control rows are `TUM cabinet` and `TUM large_cabinet`, monocular, 3
independent processes each. They must run the **original** Photo-SLAM, not this
fork, and not this fork's T0 arm.

Reason, stated because it was previously proposed: T0 of this fork is a
pure-point ORB-SLAM3 entry with the Gaussian mapper switched off
(`PHOTO_SLAM_SLAM_ONLY=1`). It answers a different question (SLAM-only pose
accuracy with no Gaussian branch) and cannot stand in for upstream Photo-SLAM,
which optimises the Gaussian map jointly. T0 stays in the internal 486 cells and
is never counted as an upstream row.

## Traceable copy

| item | value |
|---|---|
| repository | `https://github.com/HuajianUP/Photo-SLAM.git` |
| commit | `f8bfb2f0809c003ccc3fd577dc43c576fcafa4ac` |
| commit date | 2024-06-12 15:53:51 +0800 ("Update README.md") |
| local path | `/workspace/upstream/Photo-SLAM` (cloned, checked out at that commit) |
| submodules | none (`.gitmodules` absent) |

`/workspace/code/Photo-SLAM` was deliberately **not** used as the baseline: it
has no git metadata, so nothing about its provenance can be attested, and the
only reason it was ever considered was that it lacks `GetLineMode`, which says
nothing about which upstream revision it is.

## Dependency audit (performed, all present)

`CMakeLists.txt` needs `Torch`, `CUDA`, `Eigen3`, `OpenCV 4`, `jsoncpp`,
`OpenGL`, `glm`, `glfw3` (`realsense2` optional). Verified on this machine:

| dependency | status |
|---|---|
| libtorch C++ | `/usr/local/lib/python3.10/dist-packages/torch/share/cmake/Torch/TorchConfig.cmake` (torch 2.12.0+cu128) |
| CUDA toolkit | `/usr/local/cuda`, nvcc V12.8.61 |
| `jsoncpp` | `/usr/include/jsoncpp/json/json.h` |
| `glm` | `/usr/include/glm/glm.hpp` |
| `glfw3` | `/usr/include/GLFW/glfw3.h` |
| OpenCV 4.10 | `/usr/local/lib/cmake/opencv4` |
| Pangolin | `/usr/local/lib/cmake/Pangolin` |
| GPU | RTX 5090, driver 595.99.02, sm_120 |

## Build adaptations that must be recorded (not algorithm changes)

1. `CMakeLists.txt:80` pins the rasteriser to `CUDA_ARCHITECTURES "75;86"`.
   This GPU is sm_120, so the arch list must be extended (e.g. `75;86;120`);
   otherwise the extension either fails to load or silently falls back. This is
   a build adaptation and will be recorded with the diff and the resulting
   binary hashes.
2. The fork's shared fixes (`OptimizableTypes.cpp` dangling Eigen expression,
   `LineExtractor.cc` assert, endpoint vertex ownership, merge-BA gate) are
   *this* fork's; upstream at `f8bfb2f` does not contain the line extension, so
   none of them apply. If any turns out to be needed for upstream to compile or
   run, it will be listed explicitly as a common fix with its own hash.
3. Measurement adaptation: upstream's `examples/tum_mono` is the entry used for
   the 6 rows (vocab, its own `cfg/ORB_SLAM3` mono config, its Gaussian config,
   sequence path, output dir, `no_viewer`). Its TUM export and the same
   `eval_tum_se3_v2.py` call are used, so the accuracy criterion is comparable
   to the internal cells.

## Comparability boundaries

* Same input sequences, same frame rates, same GT files, same evaluator, same
  estimator-centric association as the internal cells.
* Upstream runs the **full** pipeline (SLAM + Gaussian), so its wall-clock is not
  comparable with the internal SLAM-only cells. Cost is therefore reported in two
  separate tables: SLAM-only (internal 486) and full-Gaussian (upstream 6), with
  the internal table explicitly marked "no Gaussian mapper".
* No result-based selection: the 6 rows are the two scenes already in the
  original manifest; nothing is added.

## Status

Cloned and audited: **done**. Build: **not yet** -- deliberately scheduled after
the internal 486 cells so that `-j` compilation on the same 24 cores cannot
inflate the `track_ms`/wall measurements that are themselves deliverables. The 6
manifest rows carry `status=BLOCKED, blocked_reason=<upstream binary not built
yet>` until the build and its hashes exist. They will be executed serially
afterwards, with real exit codes, unique directories and no overwriting.
