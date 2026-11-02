/**
* Mono initialisation trace context (G1 identity) — diagnostic only.
*
* GATE: PHOTO_SLAM_MONO_TRACE=1
*
* PROBLEM THIS SOLVES
* -------------------
* TwoViewReconstruction::Reconstruct() emits [RECON-TRACE] rows but has no
* access to frame identity: it only receives keypoint arrays. Without identity,
* an attempt id is all we have, and "attempt #99 differs" cannot be mapped back
* to a concrete frame pair or initialisation episode, so it cannot support a
* causal statement about where initialisation first diverges.
*
* Rather than changing the GeometricCamera virtual signature (which would touch
* every camera model and the frozen production semantics), Tracking PUBLISHES a
* small immutable snapshot just before the reconstruct call and
* TwoViewReconstruction READS it. The contract is: the context is only valid
* during the synchronous call that follows Set(); it is not a global history.
*/

#ifndef ORB_SLAM3_MONO_TRACE_CONTEXT_H
#define ORB_SLAM3_MONO_TRACE_CONTEXT_H

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

namespace ORB_SLAM3
{
namespace mono_trace
{

struct Context
{
    bool   valid        = false;
    long long episode   = -1;    // initialisation episode counter (1,2,3...)
    int    initRound    = 0;     // attempts already spent in this episode
    long long refFrameId = -1;   // mnId of the reference (first) frame
    long long curFrameId = -1;   // mnId of the current (second) frame
    double  refTs       = 0.0;
    double  curTs       = 0.0;
    char    refFile[256] = {0};
    char    curFile[256] = {0};
    char    resetReason[64] = {0};
};

inline bool Enabled()
{
    static const char* p = std::getenv("PHOTO_SLAM_MONO_TRACE");
    return p != nullptr && std::strcmp(p, "1") == 0;
}

inline Context& Ctx()
{
    static Context* c = new Context();
    return *c;
}

inline std::mutex& Mtx()
{
    static std::mutex* m = new std::mutex();
    return *m;
}

inline void Set(long long episode, int initRound,
                long long refId, double refTs, const std::string& refFile,
                long long curId, double curTs, const std::string& curFile)
{
    if (!Enabled()) return;
    std::lock_guard<std::mutex> g(Mtx());
    Context& c = Ctx();
    c.valid      = true;
    c.episode    = episode;
    c.initRound  = initRound;
    c.refFrameId = refId;
    c.curFrameId = curId;
    c.refTs      = refTs;
    c.curTs      = curTs;
    std::snprintf(c.refFile, sizeof(c.refFile), "%s", refFile.c_str());
    std::snprintf(c.curFile, sizeof(c.curFile), "%s", curFile.c_str());
}

inline void SetResetReason(const char* reason)
{
    if (!Enabled()) return;
    std::lock_guard<std::mutex> g(Mtx());
    std::snprintf(Ctx().resetReason, sizeof(Ctx().resetReason), "%s",
                  reason ? reason : "");
}

inline void Clear()
{
    if (!Enabled()) return;
    std::lock_guard<std::mutex> g(Mtx());
    Ctx().valid = false;
}

// Copy out so the caller never holds the lock while writing to stderr.
inline Context Snapshot()
{
    std::lock_guard<std::mutex> g(Mtx());
    return Ctx();
}

} // namespace mono_trace
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_MONO_TRACE_CONTEXT_H
