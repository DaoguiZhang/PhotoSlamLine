# Corrected wait graph (from the full all-thread interrupted backtrace)

Source: `gdb_B2_interrupted_backtrace.txt` (stall at frame 1785, `timeout -s INT 420
gdb --batch -ex run -ex bt -ex "thread apply all bt"`).

## Threads that matter
| T | owner | stack (top) | state |
|---|---|---|---|
| 1 | Tracking (main) | `pthread_mutex_lock` <- `Tracking::TrackWithLine()` <- `GrabImageRGBDWithLine` <- `System::TrackRGBDWithLine` <- `main:195` | **blocked** on a mutex |
| 4 | LoopClosing | `lll_mutex_unlock` <- `MapLine::GetReferenceKeyFrame()` <- `Optimizer::OptimizeEssentialGraphWithLine()` <- `LoopClosing::MergeLocalWithLine()` <- `RunWithLine()` | **inside EGL** |
| 3 | LocalMapping | `usleep` <- `LocalMapping::RunWithLine()` | **idle/normal** |
| 2,5..27 | CUDA + ORBextractor workers | `poll` / cond/futex wait | idle |

## Accurate waiting relationship
1. `Tracking.cc:3055` is the ONLY map lock in `TrackWithLine`:
   `unique_lock<mutex> lock(pCurrentMap->mMutexMapUpdate);` -> T1 waits here.
2. `Optimizer.cc:14193` `unique_lock<mutex> lock(pMap->mMutexMapUpdate);` inside
   `OptimizeEssentialGraphWithLine`, **held across the keyframe (14197), mappoint
   (14218) and MapLine (14257) recovery loops** and released only at function end.
   T4 is inside those loops -> **T4 holds the mutex T1 wants**.
3. `LoopClosing::MergeLocalWithLine` acquires `mMutexMapUpdate` at 3482/3483 and
   3590, but those are SEPARATE scopes; the call to
   `OptimizeEssentialGraphWithLine` at 3667 is made with **no map lock held**.
4. Both LocalMapping stop waits inside `MergeLocalWithLine` (3220/3221 and
   3658/3659) occur **before any lock acquisition** -> the merge does NOT hold the
   map lock while waiting for `isStopped()`.
5. LocalMapping is NOT waiting on `mMutexMapUpdate`; it is in its normal
   `RunWithLine` sleep. `mbNotStop` is only touched from `Tracking.cc`
   (6318/6434/6445/6560), never from LoopClosing.

## Conclusion
Chain: T4 (LoopClosing) holds `pMap->mMutexMapUpdate` inside
`OptimizeEssentialGraphWithLine`'s recovery loops  ->  T1 (Tracking) blocks on the
same mutex at `Tracking.cc:3055`  ->  the frame pipeline stalls. LocalMapping plays
no part. This is **mutex-held-during-a-very-long/never-ending recovery pass**, not a
lock-order deadlock.

## Corrections to earlier statements in this session
- RETRACTED (again, with reason): "the merge holds mMutexMapUpdate while waiting for
  LocalMapping to stop" -- FALSE, both stop waits precede any lock.
- RESTORED, now with direct evidence: the pathological work IS inside
  `Optimizer::OptimizeEssentialGraphWithLine` (frames #4/#5 of thread 4), consistent
  with the earlier observation of one thread at 100% CPU. The MapLine loop at
  14257-14287 is the Photo-SLAM-L addition in that lock scope.
- The user's prescribed shape ("do not hold the map lock while waiting for
  LocalMapping to stop") is ALREADY satisfied by `MergeLocalWithLine`; it is not the
  defect.

## Not done
No fix was implemented: narrowing/removing the 14193 lock scope, or bounding the
MapLine recovery loop, requires validation (B/C x3 + stress-off regression) that was
not completed. No timeout, merge-skip or loop-cap workaround was applied.
