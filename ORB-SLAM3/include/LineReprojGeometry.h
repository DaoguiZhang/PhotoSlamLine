/**
* Minimal, dependency-light geometry for the line reprojection check.
*
* WHY THIS HEADER EXISTS
* ----------------------
* The production function LocalMapping::CheckLineReprojection() used to
* project 3D line endpoints WITHOUT applying the keyframe pose. Its two live
* callers (LocalMapping.cc, RGB-D path ~1786 and mono fallback ~1841) both pass
* WORLD coordinates:
*   - RGB-D : KeyFrame::UnprojectDepthLine() returns  mRwc * Xc + Ow   (world)
*   - Mono  : GeometricTools::Triangulate(x_c1, x_c2, Tc1w, Tc2w, x3D) yields
*             x3D in WORLD, and the fallback builds S3D/E3D around x3Dmid with
*             dirWorld = Rwc1 * dirCam                                 (world)
* while GeometricCamera::project() is K * (X/Z, Y/Z) with NO pose.
*
* This header holds the REAL implementation so that it can be compiled and
* unit-tested on the CPU without constructing a KeyFrame / Atlas / camera
* object. LocalMapping::CheckLineReprojection() is a thin wrapper around
* Evaluate() -- the test therefore exercises the shipped code path, not a
* detached re-derivation of the maths.
*
* THRESHOLDS ARE UNCHANGED: 15 px perpendicular distance, cos(20 deg) direction
* agreement, 1 px minimum projected length.
*
* NOTHING in this file allocates, matches, or optimises. Evaluate() is pure.
*/

#ifndef ORB_SLAM3_LINE_REPROJ_GEOMETRY_H
#define ORB_SLAM3_LINE_REPROJ_GEOMETRY_H

#include <Eigen/Core>
#include <cmath>
#include <functional>

namespace ORB_SLAM3
{
namespace line_reproj
{

// Ordered reject reasons. The order is also the funnel order used by the
// lifecycle counters, so a single integer identifies the earliest failing
// stage. Do not renumber without updating the CSV consumer.
enum RejectCode
{
    kAccept = 0,
    kNonFiniteInput = 1,   // world endpoint contains NaN/Inf
    kBehindCameraS = 2,    // z of start point <= kMinDepth in target frame
    kBehindCameraE = 3,    // z of end point   <= kMinDepth in target frame
    kNonFiniteProj = 4,    // projected pixel contains NaN/Inf
    kPerpDistance = 5,     // perpendicular distance > reprojTh
    kProjLength = 6,       // projected length < minProjLen (covers degenerate)
    kDirection = 7         // |cos(angle)| < cosAngleTh
};

struct Params
{
    float reprojTh     = 15.0f;                        // px, unchanged
    float cosAngleTh   = 0.93969262f;                  // cos(20 deg), unchanged
    float minProjLen   = 1.0f;                         // px, unchanged
    float minDepth     = 1e-6f;                        // metric, z guard
};

struct Result
{
    bool  ok          = false;
    int   code        = kAccept;
    float zS          = 0.f;   // depth of start endpoint in target camera frame
    float zE          = 0.f;
    float dStart      = 0.f;   // px perpendicular distance
    float dEnd        = 0.f;
    float lenProj     = 0.f;   // px projected length
    float absCos      = 0.f;   // |cos| of direction agreement
};

inline bool IsFinite3(const Eigen::Vector3f& v)
{
    return std::isfinite(v(0)) && std::isfinite(v(1)) && std::isfinite(v(2));
}

inline bool IsFinite2(const Eigen::Vector2f& v)
{
    return std::isfinite(v(0)) && std::isfinite(v(1));
}

// Perpendicular distance from a pixel to the infinite line defined by
// (lineMid, lineDir) with |lineDir| == 1.
inline float PointLineDistance(const Eigen::Vector2f& pt,
                               const Eigen::Vector2f& lineMid,
                               const Eigen::Vector2f& lineDir)
{
    const Eigen::Vector2f v = pt - lineMid;
    const Eigen::Vector2f perp = v - v.dot(lineDir) * lineDir;
    return perp.norm();
}

/**
 * Corrected check. `XwS` / `XwE` are WORLD coordinates of the two 3D
 * endpoints; they are mapped into the camera frame with (Rcw, tcw) of the
 * target keyframe BEFORE projection.
 *
 * @param Rcw      rotation world->camera of the target keyframe
 * @param tcw      translation part of Tcw (3x1) of the target keyframe
 * @param project  pixel projection functor, camera coordinates -> pixel
 */
inline Result Evaluate(const Eigen::Matrix3f& Rcw,
                       const Eigen::Vector3f& tcw,
                       const Eigen::Vector3f& XwS,
                       const Eigen::Vector3f& XwE,
                       const Eigen::Vector2f& lineMid,
                       const Eigen::Vector2f& lineDir,
                       const std::function<Eigen::Vector2f(const Eigen::Vector3f&)>& project,
                       const Params& p = Params())
{
    Result r;

    if (!IsFinite3(XwS) || !IsFinite3(XwE) ||
        !std::isfinite(Rcw(0,0)) || !std::isfinite(tcw(0)))
    {
        r.code = kNonFiniteInput;
        return r;
    }

    // --- THE FIX: world -> camera, then project -------------------------
    const Eigen::Vector3f XcS = Rcw * XwS + tcw;
    const Eigen::Vector3f XcE = Rcw * XwE + tcw;
    r.zS = XcS(2);
    r.zE = XcE(2);

    if (!IsFinite3(XcS) || !IsFinite3(XcE))
    {
        r.code = kNonFiniteInput;
        return r;
    }
    if (!(XcS(2) > p.minDepth))
    {
        r.code = kBehindCameraS;
        return r;
    }
    if (!(XcE(2) > p.minDepth))
    {
        r.code = kBehindCameraE;
        return r;
    }

    const Eigen::Vector2f pxS = project(XcS);
    const Eigen::Vector2f pxE = project(XcE);
    if (!IsFinite2(pxS) || !IsFinite2(pxE))
    {
        r.code = kNonFiniteProj;
        return r;
    }

    r.dStart = PointLineDistance(pxS, lineMid, lineDir);
    r.dEnd   = PointLineDistance(pxE, lineMid, lineDir);
    if (r.dStart > p.reprojTh || r.dEnd > p.reprojTh)
    {
        r.code = kPerpDistance;
        return r;
    }

    const Eigen::Vector2f d = pxE - pxS;
    r.lenProj = d.norm();
    // Degenerate projection (both endpoints map to the same pixel) is caught
    // here; without this guard the subsequent normalized() yields NaN and the
    // direction test would silently pass.
    if (!std::isfinite(r.lenProj) || r.lenProj < p.minProjLen)
    {
        r.code = kProjLength;
        return r;
    }

    r.absCos = std::fabs((d / r.lenProj).dot(lineDir));
    if (!(r.absCos >= p.cosAngleTh))
    {
        r.code = kDirection;
        return r;
    }

    r.ok   = true;
    r.code = kAccept;
    return r;
}

/**
 * Legacy (pre-fix) implementation, kept ONLY so the diagnostic run can record
 * old-vs-new decisions on the SAME pose snapshot. It must never drive control
 * flow. It reproduces the old behaviour exactly: no pose applied, no depth
 * guard, direction test on normalized() without a degeneracy guard.
 */
inline Result EvaluateLegacy(const Eigen::Vector3f& XwS,
                             const Eigen::Vector3f& XwE,
                             const Eigen::Vector2f& lineMid,
                             const Eigen::Vector2f& lineDir,
                             const std::function<Eigen::Vector2f(const Eigen::Vector3f&)>& project,
                             const Params& p = Params())
{
    Result r;

    const Eigen::Vector2f pxS = project(XwS);
    const Eigen::Vector2f pxE = project(XwE);
    if (!IsFinite2(pxS) || !IsFinite2(pxE))
    {
        r.code = kNonFiniteProj;
        return r;
    }

    r.dStart = PointLineDistance(pxS, lineMid, lineDir);
    r.dEnd   = PointLineDistance(pxE, lineMid, lineDir);
    if (r.dStart > p.reprojTh || r.dEnd > p.reprojTh)
    {
        r.code = kPerpDistance;
        return r;
    }

    const Eigen::Vector2f d = pxE - pxS;
    r.lenProj = d.norm();

    const Eigen::Vector2f projDir = d.normalized();      // may be NaN
    r.absCos = std::fabs(projDir.dot(lineDir));
    if (fabs(r.absCos) < p.cosAngleTh)
    {
        r.code = kDirection;
        return r;
    }

    if (r.lenProj < p.minProjLen)
    {
        r.code = kProjLength;
        return r;
    }

    r.ok   = true;
    r.code = kAccept;
    return r;
}

inline const char* CodeName(int code)
{
    switch (code)
    {
        case kAccept:         return "ACCEPT";
        case kNonFiniteInput: return "NONFINITE_INPUT";
        case kBehindCameraS:  return "BEHIND_S";
        case kBehindCameraE:  return "BEHIND_E";
        case kNonFiniteProj:  return "NONFINITE_PROJ";
        case kPerpDistance:   return "PERP_DIST";
        case kProjLength:     return "PROJ_LEN";
        case kDirection:      return "DIRECTION";
        default:              return "UNKNOWN";
    }
}

} // namespace line_reproj
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_LINE_REPROJ_GEOMETRY_H
