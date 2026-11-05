/**
* Map lifecycle event log (G4) — diagnostic only.
*
* GATE: PHOTO_SLAM_MAP_EVENTS_CSV=<path>   (unset => no I/O, no allocation)
*
* Records the events that change the meaning of a trajectory row:
*   MAP_CREATE   a new Map object exists (new world frame origin)
*   MAP_STORED   current map demoted to stored (an implicit switch)
*   MAP_CURRENT  a map becomes the active map
*   MAP_SWITCH   Atlas::ChangeMap (relocalisation / loop)
*   MAP_RESET    System::Reset or ResetActiveMap, with the reason
*
* Every row carries the wall-clock-free monotonic sequence plus the current
* keyframe id, so a trajectory row can be attributed to the map that produced
* it without stitching independent world frames together.
*/

#ifndef ORB_SLAM3_MAP_EVENT_LOG_H
#define ORB_SLAM3_MAP_EVENT_LOG_H

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

namespace ORB_SLAM3
{
namespace map_events
{

inline bool Enabled()
{
    static const char* p = std::getenv("PHOTO_SLAM_MAP_EVENTS_CSV");
    return p != nullptr && p[0] != '\0';
}

// [MEASUREMENT] Input frame index currently being tracked, published by the
// entry point before each Track* call. Without it an event has only a sequence
// number, and a merge cannot be bound to a frame boundary: any claim that "this
// reference-keyframe jump is that merge" would be an inference from jump size.
// With it, the event names the frame it happened on and the claim is checkable.
inline std::atomic<int>& CurrentFrameIndexRef()
{
    static std::atomic<int> f{-1};
    return f;
}

inline void SetFrameIndex(int f)
{
    CurrentFrameIndexRef().store(f, std::memory_order_relaxed);
}

inline int CurrentFrameIndex()
{
    return CurrentFrameIndexRef().load(std::memory_order_relaxed);
}

// [MEASUREMENT] Coordinate-frame epoch. Incremented INSIDE Emit(), i.e. under
// the same mutex that serialises the event rows, so "an event was written" and
// "the epoch advanced" are one indivisible step rather than two racy ones. The
// entry point reads it after Track* returns, which is ordered after whatever the
// tracker did during that call, and stores it with the frame's pose. The
// evaluator then splits alignment intervals on a change of this number instead
// of guessing from map ids or keyframe jumps.
inline std::atomic<int>& EpochRef()
{
    static std::atomic<int> e{0};
    return e;
}

inline int CurrentEpoch()
{
    return EpochRef().load(std::memory_order_acquire);
}

inline bool AdvancesEpoch(const char* event)
{
    return event && (std::string(event) == "MAP_CREATE"
                     || std::string(event) == "MAP_MERGE"
                     || std::string(event) == "MAP_RESET");
}

inline std::mutex& Mtx()
{
    static std::mutex* m = new std::mutex();
    return *m;
}

inline std::ofstream& Sink()
{
    static std::ofstream* s = nullptr;
    if (s == nullptr)
    {
        s = new std::ofstream(std::getenv("PHOTO_SLAM_MAP_EVENTS_CSV"),
                              std::ios::out | std::ios::trunc);
        if (s->good())
        {
            (*s) << "# map lifecycle events; seq is monotonic per process\n";
            (*s) << "seq,event,map_id,kf_id,reason,frame,epoch\n";
            s->flush();
        }
    }
    return *s;
}

inline void Emit(const char* event, long long mapId, long long kfId,
                 const char* reason)
{
    if (!Enabled()) return;
    static long long seq = 0;
    std::lock_guard<std::mutex> g(Mtx());
    // The epoch advance happens under the same lock as the row write.
    if (AdvancesEpoch(event))
        EpochRef().fetch_add(1, std::memory_order_acq_rel);
    std::ofstream& o = Sink();
    if (!o.good()) return;
    o << (seq++) << ',' << event << ',' << mapId << ',' << kfId << ','
      << (reason ? reason : "") << ',' << CurrentFrameIndex() << ','
      << CurrentEpoch() << '\n';
    o.flush();
    std::cerr << "[MAP-EVENT] " << event << " map=" << mapId
              << " kf=" << kfId
              << " frame=" << CurrentFrameIndex()
              << (reason && reason[0] ? " reason=" : "")
              << (reason ? reason : "") << std::endl;
}

} // namespace map_events
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_MAP_EVENT_LOG_H
