/**
* This file is part of the Photo-SLAM-L fork of ORB-SLAM3.
*
* Event-level audit trail for reconstructing WHY an exported trajectory can
* disagree with the estimate that was produced online.
*
* Motivation (see the mono L30 C failure): the exporter reconstructs a frame's
* world pose as
*     Tcw_export = Tcr * (T_ref<-live) * P_live * Two
* where T_ref<-live is a product of the per-keyframe mTcp snapshot that
* KeyFrame::SetBadFlag writes exactly once, at culling time. Nothing refreshes
* mTcp afterwards, so any later change of the parent's pose (local BA, pose graph
* after a loop, map merge with a scale) makes that product stale. Frames whose
* reference keyframe was culled are then displaced, and the displacement shows up
* as a jump exactly when the reference keyframe switches back to a live one.
*
* To decide whether that is what actually happens - and which update event breaks
* consistency first - this trail records, with a single monotonic clock and a
* sequence number:
*   KFCREATE  a keyframe was created from a frame (id, frame, pose)
*   SETPOS    a keyframe pose changed (id, tag = who changed it, delta)
*   SETBAD    a keyframe was culled (id, parent, stored mTcp, poses at that time)
*   REPARENT  a parent link was rewritten (child, old parent, new parent, caller,
*             whether the child was already bad - i.e. before or after SetBadFlag)
*   ERASEKF   a keyframe was removed from a map (id, map id)
*   MERGE     a map merge applied a Sim3 to a set of keyframes (ids, scale, ...)
*   MAPCREATE / MAPBAD / MAPCLEAR   atlas/map lifecycle
*
* Everything is OFF unless PHOTO_SLAM_AUDIT is set to a file path, so the default
* runtime cost is one predictable branch per hook.
*
* Env:
*   PHOTO_SLAM_AUDIT        = output file path (unset -> disabled)
*   PHOTO_SLAM_AUDIT_IDS    = "318,331,332" (only these keyframes get SETPOS rows)
*   PHOTO_SLAM_AUDIT_RANGE  = "300:400"     (alternative to the id list)
*   (if neither is set, SETPOS rows are written for every keyframe)
* Structural events (SETBAD / REPARENT / MERGE / ERASEKF / lifecycle) are always
* written, because they are rare and they are the ones that explain the chain.
*
* Header-only and inline so that a single instance is shared by the whole shared
* library without touching the build.
*/

#ifndef PHOTO_SLAM_L_AUDIT_TRAIL_H
#define PHOTO_SLAM_L_AUDIT_TRAIL_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

namespace ORB_SLAM3
{
namespace Audit
{

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

inline int& StateRef()
{
    static int nState = -1;   // -1 = not probed, 0 = off, 1 = on
    return nState;
}

inline std::ofstream*& StreamRef()
{
    static std::ofstream* pStream = nullptr;
    return pStream;
}

inline std::mutex& MutexRef()
{
    static std::mutex mMutex;
    return mMutex;
}

inline std::chrono::steady_clock::time_point& T0Ref()
{
    static std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    return t0;
}

inline unsigned long long& SeqRef()
{
    static unsigned long long nSeq = 0;
    return nSeq;
}

// Keyframes selected for the (potentially voluminous) SETPOS rows.
inline std::vector<unsigned long>& IdsRef()
{
    static std::vector<unsigned long> vIds;
    return vIds;
}
inline unsigned long& RangeLoRef()
{
    static unsigned long nLo = 0;
    return nLo;
}
inline unsigned long& RangeHiRef()
{
    static unsigned long nHi = 0;   // 0 = no range
    return nHi;
}

inline void Init()
{
    int& nState = StateRef();
    if(nState >= 0)
        return;

    const char* strPath = std::getenv("PHOTO_SLAM_AUDIT");
    if(!strPath || !strPath[0])
    {
        nState = 0;
        return;
    }
    std::ofstream* pStream = new std::ofstream(strPath);
    if(!pStream->is_open())
    {
        delete pStream;
        nState = 0;
        return;
    }
    *pStream << std::fixed;
    *pStream << "# seq t_ms event fields...   (t_ms = monotonic clock since first use)" << std::endl;
    *pStream << "# SETPOS id tag dt_m dr_deg map_from_pose; structural events carry their own fields"
             << std::endl;
    StreamRef() = pStream;

    const char* strIds = std::getenv("PHOTO_SLAM_AUDIT_IDS");
    if(strIds && strIds[0])
    {
        std::string strList(strIds);
        size_t nPos = 0;
        while(nPos < strList.size())
        {
            size_t nNext = strList.find(',', nPos);
            if(nNext == std::string::npos)
                nNext = strList.size();
            const std::string strOne = strList.substr(nPos, nNext - nPos);
            if(!strOne.empty())
                IdsRef().push_back(std::strtoul(strOne.c_str(), nullptr, 10));
            nPos = nNext + 1;
        }
    }
    const char* strRange = std::getenv("PHOTO_SLAM_AUDIT_RANGE");
    if(strRange && strRange[0])
    {
        std::string strOne(strRange);
        const size_t nCol = strOne.find(':');
        if(nCol != std::string::npos)
        {
            RangeLoRef() = std::strtoul(strOne.substr(0, nCol).c_str(), nullptr, 10);
            RangeHiRef() = std::strtoul(strOne.substr(nCol + 1).c_str(), nullptr, 10);
        }
    }
    nState = 1;
}

inline bool Enabled()
{
    int& nState = StateRef();
    if(nState < 0)
        Init();
    return nState == 1;
}

inline bool Watched(unsigned long nId)
{
    if(!IdsRef().empty())
        return std::find(IdsRef().begin(), IdsRef().end(), nId) != IdsRef().end();
    if(RangeHiRef() > 0)
        return nId >= RangeLoRef() && nId <= RangeHiRef();
    return true;   // no filter: log every keyframe
}

// ---------------------------------------------------------------------------
// Call-site tag (which stage is changing the pose / the parent link)
// ---------------------------------------------------------------------------

inline const char*& TagRef()
{
    static thread_local const char* strTag = "-";
    return strTag;
}

inline const char* Tag()
{
    return TagRef();
}

struct ScopedTag
{
    const char* strPrev;
    explicit ScopedTag(const char* strNew) : strPrev(TagRef())
    {
        TagRef() = strNew;
    }
    ~ScopedTag()
    {
        TagRef() = strPrev;
    }
    ScopedTag(const ScopedTag&) = delete;
    ScopedTag& operator=(const ScopedTag&) = delete;
};

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

inline double NowMs()
{
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - T0Ref()).count();
}

inline std::string Vec3(const Eigen::Vector3f& v)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << v(0) << "," << v(1) << "," << v(2);
    return os.str();
}

inline std::string Quat(const Eigen::Quaternionf& q)
{
    std::ostringstream os;
    os << std::fixed << std::setprecision(6) << q.x() << "," << q.y() << "," << q.z() << "," << q.w();
    return os.str();
}

inline std::string Pose(const Sophus::SE3f& T)
{
    return Vec3(T.translation()) + "|" + Quat(T.unit_quaternion());
}

inline double RelTrans(const Sophus::SE3f& a, const Sophus::SE3f& b)
{
    return (b.translation() - a.translation()).norm();
}

inline double RelAngleDeg(const Sophus::SE3f& a, const Sophus::SE3f& b)
{
    Eigen::Quaternionf dq = (a.unit_quaternion().conjugate() * b.unit_quaternion()).normalized();
    const double d = 2.0 * std::atan2(dq.vec().norm(), std::fabs(dq.w()));
    return d * 180.0 / M_PI;
}

// Raw writer: one line, "seq t_ms <strEvent> <strFields>".
inline void Write(const char* strEvent, const std::string& strFields)
{
    std::ofstream* pStream = StreamRef();
    if(!pStream)
        return;
    std::lock_guard<std::mutex> lock(MutexRef());
    *pStream << ++SeqRef() << " " << std::setprecision(3) << NowMs()
             << " " << strEvent << " " << strFields << std::endl;
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

inline void KFCreate(unsigned long nKFId, unsigned long nFrameId, const Sophus::SE3f& Tcw)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "kf=" << nKFId << " frame=" << nFrameId << " pose(" << Pose(Tcw) << ")";
    Write("KFCREATE", os.str());
}

// A keyframe pose changed. Only written for watched ids, and only when it really
// changed, so the trail stays readable.
inline void SetPos(unsigned long nKFId, const Sophus::SE3f& Tbefore, const Sophus::SE3f& Tafter)
{
    if(!Enabled() || !Watched(nKFId))
        return;
    const double dT = RelTrans(Tbefore, Tafter);
    const double dR = RelAngleDeg(Tbefore, Tafter);
    if(dT < 1e-7 && dR < 1e-5)
        return;
    std::ostringstream os;
    os << "kf=" << nKFId << " tag=" << Tag()
       << " dT=" << std::setprecision(6) << dT << " dR_deg=" << dR
       << " before(" << Pose(Tbefore) << ") after(" << Pose(Tafter) << ")";
    Write("SETPOS", os.str());
}

// A keyframe was culled. Recorded with the parent and the mTcp snapshot that the
// exporter will later multiply, plus the poses that the snapshot is derived from,
// so a later parent-pose change can be measured against it.
inline void SetBad(unsigned long nKFId, long nParentId, const Sophus::SE3f& Tself,
                   const Sophus::SE3f& Tparent, const Sophus::SE3f& mTcp)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "kf=" << nKFId << " parent=" << nParentId
       << " tag=" << Tag()
       << " mTcp(" << Pose(mTcp) << ")"
       << " self(" << Pose(Tself) << ")"
       << " parentpose(" << Pose(Tparent) << ")";
    Write("SETBAD", os.str());
}

inline void Reparent(unsigned long nChildId, long nOldParentId, long nNewParentId,
                     bool bChildBad, bool bOldParentBad)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "child=" << nChildId << " old=" << nOldParentId << " new=" << nNewParentId
       << " child_bad=" << (bChildBad ? 1 : 0)
       << " old_bad=" << (bOldParentBad ? 1 : 0)
       << " tag=" << Tag();
    Write("REPARENT", os.str());
}

inline void EraseKF(unsigned long nKFId, long nMapId, bool bIsBad)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "kf=" << nKFId << " map=" << nMapId << " isbad=" << (bIsBad ? 1 : 0) << " tag=" << Tag();
    Write("ERASEKF", os.str());
}

// A merge applied "s*T" to a set of keyframes, moving them into another map.
// The map pair is set once per merge by SetMergeContext, so the per-keyframe
// rows stay free of extra map lookups inside the locked merge region.
inline long& MergeFromRef()
{
    static thread_local long nFrom = -1;
    return nFrom;
}
inline long& MergeToRef()
{
    static thread_local long nTo = -1;
    return nTo;
}

inline void SetMergeContext(long nFromMapId, long nToMapId)
{
    MergeFromRef() = nFromMapId;
    MergeToRef() = nToMapId;
}

inline void MergeApply(unsigned long nKFId, double dScale, const Sophus::SE3f& Tbefore,
                       const Sophus::SE3f& Tafter)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "from_map=" << MergeFromRef() << " to_map=" << MergeToRef()
       << " scale=" << std::setprecision(6) << dScale
       << " kf=" << nKFId << " dT=" << RelTrans(Tbefore, Tafter)
       << " dR_deg=" << RelAngleDeg(Tbefore, Tafter)
       << " before(" << Vec3(Tbefore.translation()) << ")"
       << " after(" << Vec3(Tafter.translation()) << ")";
    Write("MERGE", os.str());
}

inline void MergeEvent(const std::string& strFields)
{
    if(!Enabled())
        return;
    Write("MERGE_EVENT", strFields);
}

inline void MapCreate(long nMapId, unsigned long nLastInitKFId)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "map=" << nMapId << " last_init_kf=" << nLastInitKFId << " tag=" << Tag();
    Write("MAPCREATE", os.str());
}

inline void MapBad(long nMapId)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "map=" << nMapId << " tag=" << Tag();
    Write("MAPBAD", os.str());
}

inline void MapClear(long nMapId, size_t nKF, size_t nMP)
{
    if(!Enabled())
        return;
    std::ostringstream os;
    os << "map=" << nMapId << " kfs=" << nKF << " points=" << nMP << " tag=" << Tag();
    Write("MAPCLEAR", os.str());
}

inline void Note(const char* strWhat, const std::string& strFields)
{
    if(!Enabled())
        return;
    Write(strWhat, strFields);
}

} // namespace Audit
} // namespace ORB_SLAM3

#endif // PHOTO_SLAM_L_AUDIT_TRAIL_H
