/**
 * Runtime ablation switch for the SLAM line pipeline.
 *
 * Controlled by the environment variable `PHOTO_SLAM_LINE_MODE`:
 *   0 = point-only         (original monocular pipeline; no line extraction,
 *                            matching, or line edges in optimization)
 *   1 = line-frontend only (lines are detected / matched and MapLines are
 *                            created, but NO line edges are added to Pose
 *                            Optimization or Local BA)
 *   2 = full point-line    (current default: line edges used in Pose
 *                            Optimization and Local BA)
 *
 * The value is read once at first use (thread-safe read-only getenv);
 * the default (unset) is 2, which preserves the existing behavior.
 */

#ifndef ORB_SLAM3_LINEMODE_H
#define ORB_SLAM3_LINEMODE_H

#include <cstdlib>
#include <cmath>
#include <string>

namespace ORB_SLAM3
{

inline int GetLineMode()
{
    static const int mode = []() {
        const char* v = std::getenv("PHOTO_SLAM_LINE_MODE");
        if (v == nullptr)
            return 2;
        const std::string s(v);
        if (s == "0")
            return 0;
        if (s == "1")
            return 1;
        return 2;
    }();
    return mode;
}

// Independent toggles for line edges in the two SLAM optimizers (used for the
// B/P/L/C ablation). They only take effect when PHOTO_SLAM_LINE_MODE >= 2
// (i.e., the line frontend is active). Default is enabled to preserve the
// existing full point-line behavior.
inline bool GetPoseOptLineEnabled()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_PO_LINE");
        return !(v != nullptr && std::string(v) == "0");
    }();
    return enabled;
}

inline bool GetLbaLineEnabled()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_LBA_LINE");
        return !(v != nullptr && std::string(v) == "0");
    }();
    return enabled;
}

// Per-optimization line/point edge accounting (PHOTO_SLAM_DEBUG_LINE_EDGES=1).
inline bool IsLineEdgeDiag()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_DEBUG_LINE_EDGES");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

// Monocular-init diagnostics (PHOTO_SLAM_DEBUG_MONO_INIT=1). Mirrors the
// file-local MonoInitLineDebug() in Tracking.cc / LSDmatcher.cc.
inline bool IsMonoInitDiag()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_DEBUG_MONO_INIT");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

// Bounded numeric candidates for the LBA line path (only active when
// PHOTO_SLAM_LINE_MODE >= 2 && PHOTO_SLAM_LBA_LINE is enabled).
// Defaults reproduce the current (C0) behavior exactly.
//
//   PHOTO_SLAM_LBA_TAU   : line outlier chi2 threshold (default 9.0)
//   PHOTO_SLAM_LBA_W     : line information-matrix weight multiplier (default 1.0)
//   PHOTO_SLAM_LBA_DELTA : line Huber delta (default sqrt(5.991))
//
// NOTE: if PHOTO_SLAM_LBA_W is changed, the outlier threshold and Huber delta
// must be rescaled by the caller (tau_new = w*tau_old, delta_new = sqrt(w)*
// delta_old) to keep the weighted-chi2 gating and Huber breakpoint consistent.
inline double GetLbaLineTau()
{
    static const double tau = []() {
        const char* v = std::getenv("PHOTO_SLAM_LBA_TAU");
        return (v == nullptr) ? 9.0 : std::stod(v);
    }();
    return tau;
}

inline double GetLbaLineWeight()
{
    static const double w = []() {
        const char* v = std::getenv("PHOTO_SLAM_LBA_W");
        return (v == nullptr) ? 1.0 : std::stod(v);
    }();
    return w;
}

inline double GetLbaLineDelta()
{
    static const double d = []() {
        const char* v = std::getenv("PHOTO_SLAM_LBA_DELTA");
        return (v == nullptr) ? std::sqrt(5.991) : std::stod(v);
    }();
    return d;
}

// LBA line observation noise: assumed std (pixels) of each endpoint's
// perpendicular reprojection distance to the projected 2D line.
//
// Source/assumption: line detection is single-octave (LSDextractor levels=1,
// cv::line_descriptor::BinaryDescriptor numOfOctave=1), so kl.octave==0 and
// the previous mvInvLevelSigma2[kl.octave] lookup always returned the ORB
// level-0 value 1.0 (sigma=1.0 px). The default 1.0 px reproduces C0 exactly.
//
// NOTE: the information matrix is (1/sigma^2) * I * GetLbaLineWeight(); changing
// sigma_px scales chi2 by 1/sigma^2, which also rescales the Huber input and the
// effective outlier threshold -- it is NOT a pure weight change.
inline double GetLbaLineSigmaPx()
{
    static const double s = []() {
        const char* v = std::getenv("PHOTO_SLAM_LBA_LINE_SIGMA_PX");
        return (v == nullptr) ? 1.0 : std::stod(v);
    }();
    return s;
}

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_LINEMODE_H
