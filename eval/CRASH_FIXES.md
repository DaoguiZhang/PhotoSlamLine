# The seven SIGSEGV cells: root causes, fixes, and acceptance

All seven failures were ETH3D `sfm_house_loop` (mono P1/P2 rep1, rgbd P1
rep1-3, rgbd P2 rep1/rep3), exit -11 after ~20 s. Two INDEPENDENT defects were
found; the second only became visible once the first was fixed, so they are
reported separately and neither is assumed to explain the other.

## Defect 1 - uninitialised `mvpMapLines` (heap/segment read at 0x0)

ASan, full module set:

```
SEGV on unknown address 0x0 (READ)  T1
#0 KeyFrame::GetKeyLineInfo(...)
#1 MappingOperation::addKeyFrame(KeyFrame*, bool)
#2 Optimizer::LocalBundleAdjustment(...)
#3 LocalMapping::RunWithLine()
```

`GetKeyLineInfo` indexed `mvpMapLines[i]` for `i < NL`, but `mvpMapLines` was
never initialised in the `KeyFrame` constructor and stayed EMPTY. A temporary
`[KF-NL-MISMATCH]` diagnostic printed the invariant violation directly
(`NL=10 mvKeyLines=10 mvpMapLines=0`), so the cause is observed, not inferred.

Fix: size the vector to `NL` in the constructor
(`mvpMapLines.assign(NL, nullptr)`), which is the invariant every reader already
assumes. The index clamp added first is kept as defence in depth.

## Defect 2 - rounded depth index one row past the image

With defect 1 fixed, ASan reported:

```
heap-buffer-overflow READ of size 4
#0 Frame::ComputeLineStereoFromRGBD(cv::Mat const&)
#1 Frame::Frame(...)  #2 Tracking::GrabImageRGBDWithLine(...)
region is a cv::fastMalloc buffer; fault is 2776 bytes past its end
```

`ComputeLineStereoFromRGBD` validated the endpoint coordinates as floats
(`v1 < rows`) and then indexed with `cvRound(v1)`. `cvRound(rows - 0.4)` is
`rows`, so an endpoint just inside the border reads one row past the buffer.
Fix: round first, then validate `uu/vv` (the sibling function
`ComputeLineStereoFromRGBDNew` already did this).

## Acceptance (both builds, both crash paths)

`P2`, `PHOTO_SLAM_LINE_MODE=2`, full 556-frame sequence:

| run | exit | ASan errors | NL-mismatch | frames |
|---|---|---|---|---|
| rgbd Release | 0 | - | 0 | 556/556 |
| rgbd ASan | 0 | **0** | 0 | 556/556 |
| mono Release | 0 | - | 0 | 556/556 |
| mono ASan | 0 | **0** | 0 | 556/556 |

Before the fixes the same cells died at ~20 s. No merge was disabled, no
threshold relaxed, no necessary free removed and no accept state touched.

## Version

| artefact | sha256 |
|---|---|
| `lib_rel/libORB_SLAM3.so` | `e49b377f5e781778343211f01f16ffe5d56694d8e3a11f2526ac3400764fc8a8` |
| `lib_asan/libORB_SLAM3.so` | `02d08745c47a8158a70d691bdd535c723c368372dcfca2bf18d0f0a33cb1435a` |
| `entry_rel/eval_frontend` | `1d3f4f63a172941aeac3eae0f7f0a7b93d53f08bee4e6b93733f3308c123e62d` |
| `entry_asan/eval_frontend` | `5278593a2ff97d45331693a3003d18ae3c99565b984dc5203239a9e1d534e13d` |
| `ORB-SLAM3/src/KeyFrame.cc` | `1a91239a996e34ef5cc2362836dbe86ecb0baa237b57a9139f09a50a744c8399` |
| `ORB-SLAM3/src/Frame.cc` | `9432f4f5042dbe834876b2ba5262942049f734fa47eb66fa1aa861d99b32b4d2` |

## Scope still to run

The two defects sit on the shared path `LocalBundleAdjustment ->
MappingOperation::addKeyFrame -> GetKeyLineInfo` and in the RGB-D frame
constructor, so the affected cells are every P1/P2 cell that executes those
paths, not only the seven that happened to crash. The re-run set is therefore
defined from the call path (all P1/P2 cells in the same scenes/modalities) and
executed serially into fresh directories, with the seven failures kept.
