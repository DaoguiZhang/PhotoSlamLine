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
#include <thread>

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

// ---------------------------------------------------------------------------
// Line-rescue: when the point-only success test fails, allow a RELIABLE
// point+line joint constraint to keep the frame alive. Default is OFF, so the
// behaviour with the variables unset is byte-identical to the original code.
// All thresholds are counts / degrees / ratios, i.e. independent of metric
// scale; the motion cap is depth-normalised (see Tracking::HasReliableLineSupport).
// ---------------------------------------------------------------------------
inline bool GetLineRescueEnabled()
{
    static const bool v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_RESCUE");
        return e != nullptr && std::string(e) == "1";
    }();
    return v;
}
inline int GetLineRescueMinLines()
{
    static const int v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_RESCUE_MIN_LINES");
        return e ? std::atoi(e) : 8;
    }();
    return v;
}
inline float GetLineRescueMinAngleDeg()
{
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_RESCUE_MIN_ANGLE_DEG");
        return e ? (float)std::atof(e) : 25.0f;
    }();
    return v;
}
inline int GetLineRescueMinPoints()
{
    static const int v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_RESCUE_MIN_POINTS");
        return e ? std::atoi(e) : 10;
    }();
    return v;
}
inline float GetLineRescueMinRatio()
{
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_RESCUE_MIN_RATIO");
        return e ? (float)std::atof(e) : 0.5f;
    }();
    return v;
}
inline float GetLineRescueMaxMotionRatio()
{
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_RESCUE_MAX_MOTION_RATIO");
        return e ? (float)std::atof(e) : 0.5f;
    }();
    return v;
}
inline bool IsLineRescueDiag()
{
    static const bool v = []() {
        const char* e = std::getenv("PHOTO_SLAM_DEBUG_LINE_RESCUE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

// Pre-bypass gate diagnostics (PHOTO_SLAM_DEBUG_LINE_GATE=1). Evaluated at the
// exact point where the ORIGINAL point-only early return would have fired, i.e.
// BEFORE any rescue bypass, and records the frame id, the point-match count and
// whether the old control flow would have returned false there. This is kept
// STRICTLY SEPARATE from the post-optimisation point verdict so the two can
// never be conflated (the old path firing is not the same as the joint path
// agreeing afterwards).
inline bool IsLineGateDiag()
{
    static const bool v = []() {
        const char* e = std::getenv("PHOTO_SLAM_DEBUG_LINE_GATE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

// Rescue-opportunity scan (PHOTO_SLAM_DEBUG_LINE_SCAN=1). Emits one [LineScan]
// line per decision point that evaluates the LINE evidence on the current frame
// UNCONDITIONALLY -- i.e. whether or not the point-only gate failed, and with
// the rescue itself switched off. This is what lets us count natural rescue
// opportunities on a sequence without enabling, or biasing, the rescue.
// Relay point floor / diagnostics are above. Below: the STRESS-TEST instrument.
inline bool IsLineScanDiag()
{
    static const bool v = []() {
        const char* e = std::getenv("PHOTO_SLAM_DEBUG_LINE_SCAN");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

// ---------------------------------------------------------------------------
// Map-merge / essential-graph segmented diagnostics (PHOTO_SLAM_DEBUG_MERGE=1).
// Default OFF. Emits [MergeDiag] records from BOTH overloads of
// Optimizer::OptimizeEssentialGraphWithLine. The overload taking
// (KeyFrame*, vpFixedKFs, ...) is the one LoopClosing::MergeLocalWithLine calls
// (LoopClosing.cc:3667) and is tagged "EGL2"; the overload taking
// (Map* pMap, ...) is reached from CorrectLoop and is tagged "EGL". Records
// carry the OS thread id so a stall can be attributed to a thread, plus
// per-section and per-100-element timings. Instrumenting only the Map*
// overload previously made an empty stream look like "no merge ran" when merges
// were in fact running through the other one.
// Purely observational: no behaviour change, no extra locking.
// ---------------------------------------------------------------------------
inline bool IsMergeDiag()
{
    static const bool v = []() {
        const char* e = std::getenv("PHOTO_SLAM_DEBUG_MERGE");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return v;
}

inline unsigned long long MergeDiagTid()
{
    return (unsigned long long)std::hash<std::thread::id>{}(std::this_thread::get_id());
}

// ---------------------------------------------------------------------------
// SEPARATELY-LABELLED STRESS TEST (default OFF, never used in normal runs).
//
// Purpose: the natural sequences never make the point-only gate fail, so the
// rescue can only be exercised by inducing that failure. This instrument drops
// a FIXED fraction of POINT observations from pose optimization, starting from
// a FIXED frame index (so initialization is untouched and the rule is identical
// in every condition). Line observations are never modified.
//
//   PHOTO_SLAM_STRESS_DROP_POINT_RATIO  fraction in [0,1); 0 = off (default)
//   PHOTO_SLAM_STRESS_START_FRAME       first frame id the drop applies to
//
// Results obtained with this enabled must always be reported separately from
// natural-sequence results.
// ---------------------------------------------------------------------------
inline float GetStressDropPointRatio()
{
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_STRESS_DROP_POINT_RATIO");
        return e ? (float)std::atof(e) : 0.0f;
    }();
    return v;
}
inline int GetStressStartFrame()
{
    static const int v = []() {
        const char* e = std::getenv("PHOTO_SLAM_STRESS_START_FRAME");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}
// Deterministic selector: drop the i-th point of every k-th matched point.
inline bool StressDropPoint(int frameId, int matchedIndex)
{
    const float frac = GetStressDropPointRatio();
    if (frac <= 0.0f || frac >= 1.0f)
        return false;
    if (frameId < GetStressStartFrame())
        return false;
    int k = (int)(1.0f / frac + 0.5f);
    if (k < 1)
        k = 1;
    return (matchedIndex % k) == 0;
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

// Line-Gaussian structural-loss diagnostics (PHOTO_SLAM_DEBUG_LINE_LOSS=1).
// Prints the per-component line loss (L_ori / L_ecc / L_shape_ori) and the
// number of line-sampled points received from SLAM. Off by default; enabling
// forces an extra CPU-GPU sync when reading the scalar loss values.
inline bool IsLineLossDiag()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_DEBUG_LINE_LOSS");
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

// Shadow-line isolation switch (PHOTO_SLAM_SHADOW_LINE=1).
// When enabled, the line frontend still runs (LSD extraction + line matching in
// Tracking) but the LocalMapping backend does NOT create / cull / fuse MapLines
// and uses the standard point-only NeighborSearch. This isolates the backend
// line bookkeeping from the point pipeline (S1 isolation).
inline bool IsShadowLineMode()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_SHADOW_LINE");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

// Loop-closing line integration switch (PHOTO_SLAM_LINE_LOOP).
//
//   0 = A (C0 baseline): point-only loop closing. System launches LoopClosing::Run
//       (point detection / CorrectLoop / point Essential Graph / point GBA).
//       This reproduces the verified C0 behavior exactly.
//   1 = B: point loop detection + line map/Gaussian sync. Lines are detected with
//       the (point) loop candidates, matched, then corrected / fused / resampled
//       together with MapPoints, but the Sim(3) refinement and the global BA stay
//       point-only (lines are carried through reference-keyframe correction).
//   2 = C (default): B + line residuals participate in the Sim(3) refinement and
//       the global BA adds MapLine vertices + line observation edges.
//
// Line loop mode only takes effect when PHOTO_SLAM_LINE_MODE >= 1 (line frontend
// active). With PHOTO_SLAM_LINE_LOOP=0 the pipeline is identical to C0.
inline int GetLineLoopMode()
{
    static const int mode = []() {
        const char* v = std::getenv("PHOTO_SLAM_LINE_LOOP");
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

// Log channel for line loop closing (PHOTO_SLAM_DEBUG_LINE_LOOP=1).
// Emits: loop candidates found, point/line geometric validation + acceptance,
// Sim(3) scale & correction, MapLine corrected/fused counts, Essential Graph done,
// GBA trigger/finish/write-back, Gaussian sync counts.
inline bool IsLineLoopDiag()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_DEBUG_LINE_LOOP");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

} // namespace ORB_SLAM3

#endif // ORB_SLAM3_LINEMODE_H
