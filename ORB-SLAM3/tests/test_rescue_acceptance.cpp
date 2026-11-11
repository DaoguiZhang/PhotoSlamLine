// Mechanism tests for RescueAccept::evaluate().
//
// These are the three tests the acceptance protocol asks for, expressed
// directly on the acceptance function so that each one is a hard pass/fail
// instead of an interpretation of end-to-end numbers:
//
//   T1 good points + WRONG line candidate      -> reject, and the rejection
//      happens on the matching stage (the pose itself is not blamed).
//   T2 few points whose translation is weak    -> accepted only because the
//      lines supply the missing direction; the identical observation set
//      WITHOUT the lines must be rejected as translation-unobservable.
//   T3 many lines, all nearly parallel         -> reject; a large line count
//      must not turn into a translation solution.
//
// Build (no SLAM, no GPU):
//   g++ -O2 -std=c++14 -I/usr/include/eigen3 -I<repo>/ORB-SLAM3/include \
//       test_rescue_acceptance.cpp -o test_rescue_acceptance
#include "RescueAcceptance.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace RescueAccept;

static int gFail = 0;

static Eigen::Matrix3d makeK()
{
    Eigen::Matrix3d K;
    K << 525.0, 0.0, 319.5,
         0.0, 525.0, 239.5,
         0.0, 0.0, 1.0;
    return K;
}

static Eigen::Matrix4d makePose(double yawDeg, const Eigen::Vector3d& t)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const double a = yawDeg * M_PI / 180.0;
    T.block<3, 3>(0, 0) = Eigen::AngleAxisd(a, Eigen::Vector3d::UnitY()).toRotationMatrix();
    T.block<3, 1>(0, 3) = t;
    return T;
}

static Eigen::Vector2d proj(const Eigen::Matrix3d& K, const Eigen::Matrix4d& T,
                            const Eigen::Vector3d& Xw)
{
    const Eigen::Vector3d Xc = T.block<3, 3>(0, 0) * Xw + T.block<3, 1>(0, 3);
    const Eigen::Vector3d p = K * Xc;
    return Eigen::Vector2d(p(0) / p(2), p(1) / p(2));
}

// A point grid in front of the camera; `spread` controls how much the points
// sample the image plane (small spread = weak translation conditioning).
static std::vector<PointObs> makePoints(const Eigen::Matrix3d& K, const Eigen::Matrix4d& T,
                                        int nx, int ny, double depth, double halfExtent,
                                        int nHeldout = 0)
{
    std::vector<PointObs> out;
    int idx = 0;
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < ny; ++j) {
            const double x = -halfExtent + 2.0 * halfExtent * i / std::max(1, nx - 1);
            const double y = -halfExtent * 0.7 + 1.4 * halfExtent * j / std::max(1, ny - 1);
            Eigen::Vector3d Xc(x, y, depth);
            const Eigen::Matrix4d Tcw = T;
            const Eigen::Vector3d Xw = Tcw.block<3, 3>(0, 0).transpose() *
                                       (Xc - Tcw.block<3, 1>(0, 3));
            PointObs p;
            const Eigen::Vector2d uv = proj(K, T, Xw);
            p.u = uv(0); p.v = uv(1); p.Xw = Xw; p.w = 1.0;
            p.heldout = (nHeldout > 0 && (idx % 7) == 0);
            out.push_back(p);
            ++idx;
        }
    }
    return out;
}

// Lines at a given depth, with image-plane directions taken from `dirsDeg`.
static std::vector<LineObs> makeLines(const Eigen::Matrix3d& K, const Eigen::Matrix4d& T,
                                      const std::vector<double>& dirsDeg, double depth,
                                      double halfLen, int nObs, double pixelShift = 0.0)
{
    std::vector<LineObs> out;
    int id = 100;
    for (size_t k = 0; k < dirsDeg.size(); ++k) {
        const double a = dirsDeg[k] * M_PI / 180.0;
        const Eigen::Vector3d c(-0.6 + 1.2 * (double)k / std::max<size_t>(1, dirsDeg.size() - 1),
                                0.1 * ((int)k % 3 - 1), depth + 0.15 * ((int)k % 4));
        const Eigen::Vector3d d(std::cos(a), std::sin(a), 0.0);
        const Eigen::Vector3d Xw1 = c - halfLen * d;
        const Eigen::Vector3d Xw2 = c + halfLen * d;
        LineObs L;
        L.Xw1 = Xw1; L.Xw2 = Xw2;
        L.a = proj(K, T, Xw1);
        L.b = proj(K, T, Xw2);
        // a deliberate, known error: shift both observed endpoints sideways
        const Eigen::Vector2d nrm = Eigen::Vector2d(-(L.b - L.a)(1), (L.b - L.a)(0)).normalized();
        L.a += pixelShift * nrm;
        L.b += pixelShift * nrm;
        L.mapId = id++;
        L.nObs = nObs;
        L.w = 1.0;
        out.push_back(L);
    }
    return out;
}

static void report(const char* name, const Result& r, bool expectOk, const char* expectReason)
{
    const bool pass = (r.ok == expectOk) &&
                      (expectOk || std::string(ReasonName(r.reason)) == expectReason);
    std::printf("%-46s %s  ok=%d reason=%-22s lines(raw/uniq/acc)=%d/%d/%d "
                "pts=%d inl=%d medE=%.2fpx transObs=%.2fpx rotObs=%.2fpx spread=%.1fdeg\n",
                name, pass ? "PASS" : "FAIL", (int)r.ok, ReasonName(r.reason),
                r.nLinesRaw, r.nLinesUnique, r.nLinesAccepted,
                r.nPoints, r.nPointInliers, r.medianDepth > 0 ? r.medPointReprojPx : -1.0,
                r.transObsPx, r.rotObsPx, r.spreadDeg);
    if (!pass) ++gFail;
}

int main()
{
    const Eigen::Matrix3d K = makeK();
    const Eigen::Matrix4d T = makePose(8.0, Eigen::Vector3d(0.02, -0.01, 0.05));
    Params pr;   // frozen defaults

    // ---------------- T1: good points, WRONG line candidate ----------------
    {
        const auto pts = makePoints(K, T, 6, 5, 2.0, 0.55);
        const std::vector<double> dirs = {0, 30, 60, 95, 130, 160};
        const auto good = makeLines(K, T, dirs, 2.0, 0.45, 3, 0.0);
        const auto bad = makeLines(K, T, dirs, 2.0, 0.45, 3, 30.0);   // 30 px off
        const Result rGood = evaluate(K, T, pts, good, pr);
        const Result rBad = evaluate(K, T, pts, bad, pr);
        report("T1a good points + consistent lines", rGood, true, "");
        report("T1b good points + 30px-off lines", rBad, false, "line_far");
        // The point evidence alone is fine, so the rejection must be the lines.
        const Result rNoLines = evaluate(K, T, pts, {}, pr);
        std::printf("%-46s      (points alone: ok=%d reason=%s transObs=%.2fpx)\n",
                    "T1c same points, no lines at all", (int)rNoLines.ok,
                    ReasonName(rNoLines.reason), rNoLines.transObsPx);
    }

    // ---------------- T2: few points, weak translation, lines supply it ----
    {
        // Points clustered near the optical axis at one depth: the pixel
        // residuals barely change for a translation along the view ray, so the
        // point-only geometry cannot fix scale/translation.
        const auto pts = makePoints(K, T, 3, 2, 2.0, 0.05);
        const std::vector<double> dirs = {0, 35, 70, 105, 140, 165};
        const auto lns = makeLines(K, T, dirs, 2.0, 0.45, 3, 0.0);
        const Result rPtsOnly = evaluate(K, T, pts, {}, pr);
        const Result rBoth = evaluate(K, T, pts, lns, pr);
        report("T2a few clustered points, no lines", rPtsOnly, false, "trans_unobservable");
        report("T2b same points + well-spread lines", rBoth, true, "");
        std::printf("%-46s      transObs %.2fpx (points only) -> %.2fpx (points+lines), "
                    "gain x%.1f\n", "T2c observability gain from the lines",
                    rPtsOnly.transObsPx, rBoth.transObsPx,
                    rBoth.transObsPx / std::max(1e-6, rPtsOnly.transObsPx));
        if (!(rBoth.transObsPx > rPtsOnly.transObsPx * 2.0)) {
            std::printf("%-46s FAIL\n", "T2c lines must add real translation information");
            ++gFail;
        }
    }

    // ---------------- T3: many lines, all nearly parallel ------------------
    {
        const auto pts = makePoints(K, T, 3, 2, 2.0, 0.05);   // weak on purpose
        std::vector<double> dirs;
        for (int k = 0; k < 20; ++k)
            dirs.push_back(10.0 + 0.6 * k);                   // 10..21 deg
        const auto lns = makeLines(K, T, dirs, 2.0, 0.45, 3, 0.0);
        const Result r = evaluate(K, T, pts, lns, pr);
        report("T3 20 lines, all within 11 deg", r, false, "spread");
        std::printf("%-46s      accepted=%d spread=%.1fdeg (thr %.1f) - a big line "
                    "count does not make a translation solution\n",
                    "T3b detail", r.nLinesAccepted, r.spreadDeg, pr.minSpreadDeg);
    }

    // ---------------- control: identical observations must be deterministic
    {
        const auto pts = makePoints(K, T, 5, 4, 2.0, 0.5);
        const std::vector<double> dirs = {0, 45, 90, 135};
        const auto lns = makeLines(K, T, dirs, 2.0, 0.45, 3, 0.0);
        const Result a = evaluate(K, T, pts, lns, pr);
        const Result b = evaluate(K, T, pts, lns, pr);
        const bool same = (a.ok == b.ok) && (a.reason == b.reason) &&
                          std::fabs(a.transObsPx - b.transObsPx) < 1e-12;
        std::printf("%-46s %s\n", "T0 determinism (same input twice)",
                    same ? "PASS" : "FAIL");
        if (!same) ++gFail;
    }

    std::printf("\n%d failure(s)\n", gFail);
    return gFail == 0 ? 0 : 1;
}
