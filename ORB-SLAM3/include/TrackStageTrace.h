/**
* Per-frame tracking stage recorder (diagnostic only).
*
* WHY: the existing per-frame CSV records the NUMBERS (matches before/after PO,
* inliers, local map sizes) but not WHICH function returned false and on which
* criterion. Without that, "the first failure after initialisation" can only be
* guessed. This header adds exactly one integer per frame: the last decisive
* return point reached in the current frame's tracking chain.
*
* READ-ONLY: nothing here computes, matches, optimises, or changes a threshold.
* The value is written once per frame into the existing diagnostic CSV, so no
* extra pass over the frame is performed.
*
* Codes are grouped per function; see TrackStageName().
*/

#ifndef ORB_SLAM3_TRACK_STAGE_TRACE_H
#define ORB_SLAM3_TRACK_STAGE_TRACE_H

namespace ORB_SLAM3
{
namespace track_stage
{

// Stage of the CURRENT frame. 0 = no decisive return recorded yet.
inline int& Current()
{
    static thread_local int s = 0;
    return s;
}

// Last frame that reached a decisive return.
inline int& LastFrameStage()
{
    static thread_local int s = 0;
    return s;
}

inline void Reset() { Current() = 0; }

// Call site of an outstanding ResetActiveMap request. Set immediately before
// the call so System::ResetActiveMap() can record WHO asked, instead of only
// which API was invoked. Reset to "unknown" after the event is emitted.
inline const char*& PendingResetSite()
{
    static thread_local const char* s = "unknown";
    return s;
}

inline void Set(int code)
{
    if (code != 0) Current() = code;
}

inline void Commit()
{
    if (Current() != 0) LastFrameStage() = Current();
}

inline const char* Name(int c)
{
    switch (c)
    {
        case 0:  return "none";
        // ---- point reference-KF tracker -------------------------------
        case 11: return "RefKF.pointMatches_lt_15";
        case 12: return "RefKF.inlierPoints_lt_10";
        case 13: return "RefKF.ok";
        // ---- point motion-model tracker -------------------------------
        case 21: return "Motion.vo_nmatches_le_20";
        case 22: return "Motion.inlierPoints_lt_10";
        case 23: return "Motion.ok";
        // ---- point local map ------------------------------------------
        case 31: return "LocalMap.poseInliers_lt_30";
        case 32: return "LocalMap.ok";
        // ---- line reference-KF tracker --------------------------------
        case 41: return "RefKFL.pointMatches_lt_15";
        case 42: return "RefKFL.inlierPoints_lt_10";
        case 43: return "RefKFL.rescueUsed";
        case 44: return "RefKFL.ok";
        // ---- line motion-model tracker --------------------------------
        case 51: return "MotionL.vo_nmatches_le_20";
        case 52: return "MotionL.inlierPoints_lt_10";
        case 53: return "MotionL.rescueUsed";
        case 54: return "MotionL.ok";
        // ---- line local map -------------------------------------------
        case 61: return "LocalMapL.poseInliers_lt_30";
        case 62: return "LocalMapL.rescueUsed";
        case 63: return "LocalMapL.ok";
        // ---- local map early exits ------------------------------------
        case 71: return "LocalMap.noLocalKeyFrames";
        case 72: return "LocalMap.noReferenceKF";
        default: return "UNKNOWN";
    }
}

} // namespace track_stage
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_TRACK_STAGE_TRACE_H
