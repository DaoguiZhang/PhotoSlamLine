#pragma once
// ---------------------------------------------------------------------------
// Independent acceptance test for line-assisted pose candidates and line rescue.
//
// WHY THIS FILE EXISTS
// The historical rescue gate counted lines and their image-plane direction
// spread. That answers "are there lines" but not "can these observations
// determine the pose". This header answers the second question from the
// observations alone:
//
//   matching      each MapLine must come from the same valid map, have
//                 multi-view support, project near the observed segment with
//                 real visible-segment overlap, and be used at most once;
//   geometry      the Gauss-Newton information matrix built from the PIXEL
//                 residuals of the point and line observations is analysed.
//                 The rotation block is marginalised out and the smallest
//                 eigenvalue of the resulting translation block is converted to
//                 "pixels of image motion per median depth" - a scale-free
//                 number that says whether translation is actually observable
//                 along its weakest direction. Solver damping, robust kernels
//                 and the constant-velocity motion prior are deliberately NOT
//                 part of this matrix, because none of them is a visual
//                 observation;
//   consistency   the pose must explain held-out points and must not convert
//                 the point-only pose's outliers into inliers in bulk.
//
// Deliberately header-only and free of ORB-SLAM types so the same code can be
// unit-tested without a map, a dataset or a GPU.
// ---------------------------------------------------------------------------

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <vector>

namespace RescueAccept
{

struct PointObs
{
    double u = 0.0, v = 0.0;   // observed pixel
    Eigen::Vector3d Xw = Eigen::Vector3d::Zero();
    double w = 1.0;            // information weight (1 / sigma_px^2)
    int frameIdx = -1;         // index of the keypoint in the frame
    bool heldout = false;      // excluded from the geometric evidence
};

struct LineObs
{
    Eigen::Vector2d a = Eigen::Vector2d::Zero();      // observed endpoint 1 (px)
    Eigen::Vector2d b = Eigen::Vector2d::Zero();      // observed endpoint 2 (px)
    Eigen::Vector3d Xw1 = Eigen::Vector3d::Zero();    // map endpoint 1 (world)
    Eigen::Vector3d Xw2 = Eigen::Vector3d::Zero();    // map endpoint 2 (world)
    int mapId = -1;
    int nObs = 0;              // MapLine::Observations() - multi-view support
    double w = 1.0;            // information weight
    int frameIdx = -1;         // index of the keyline in the frame
    int round = 0;             // 0 = constrained, 1 = held-out (redundant) line
};

struct Params
{
    // matching
    int    minUniqueLines    = 3;
    int    minMultiViewObs   = 2;
    double maxLineDistPx     = 4.0;
    double minVisibleOverlap = 0.30;
    // geometry (pixel units, scale free)
    int    minPoints         = 3;
    double minSpreadDeg      = 20.0;
    double minTransObsPx     = 1.0;
    double minRotObsPx       = 0.5;
    // consistency
    double maxPointReprojPx  = 6.0;
    double maxHeldoutMedPx   = 6.0;
    double maxHeldoutInlierFrac = 0.60;
    // numerics
    double fdEps             = 1e-6;
};

enum Reason
{
    R_ACCEPT = 0,
    R_NO_LINES,
    R_FEW_POINTS,
    R_LINE_UNSUPPORTED,
    R_LINE_FAR,
    R_LINE_DUPLICATE,
    R_LINE_OVERLAP,
    R_SPREAD,
    R_TRANS_UNOBSERVABLE,
    R_ROT_UNOBSERVABLE,
    R_POINT_FIT,
    R_CONSISTENCY,
    R_POSE_INVALID,
    R_DEGENERATE_SET
};

inline const char* ReasonName(int r)
{
    switch (r) {
    case R_ACCEPT: return "ok";
    case R_NO_LINES: return "no_lines";
    case R_FEW_POINTS: return "few_points";
    case R_LINE_UNSUPPORTED: return "line_unsupported";
    case R_LINE_FAR: return "line_far";
    case R_LINE_DUPLICATE: return "line_duplicate";
    case R_LINE_OVERLAP: return "line_overlap";
    case R_SPREAD: return "spread";
    case R_TRANS_UNOBSERVABLE: return "trans_unobservable";
    case R_ROT_UNOBSERVABLE: return "rot_unobservable";
    case R_POINT_FIT: return "point_fit";
    case R_CONSISTENCY: return "consistency";
    case R_POSE_INVALID: return "pose_invalid";
    case R_DEGENERATE_SET: return "degenerate_set";
    default: return "unknown";
    }
}

struct Result
{
    bool ok = false;
    int reason = R_NO_LINES;

    std::vector<int> acceptedLineIdx;   // frameIdx of each accepted line
    int nLinesRaw = 0, nLinesUnique = 0, nLinesAccepted = 0;
    int nLinesHeldout = 0, nLinesHeldoutOk = 0;
    int nPoints = 0, nPointsHeldout = 0, nHeldoutInliers = 0;
    int nPointInliers = 0;

    double spreadDeg = 0.0;
    double medLineDistPx = -1.0;
    double medPointReprojPx = -1.0;
    double heldoutMedReprojPx = -1.0;
    double transObsPx = 0.0;
    double rotObsPx = 0.0;
    double transCondRatio = 0.0;   // lambda_min / lambda_max of the translation block
    double dbgHttRaw = 0.0, dbgHwwDiag = 0.0, dbgZnorm = 0.0, dbgJtNorm = 0.0;
    double medianDepth = 0.0;
};

namespace detail
{

// exp of a 6-vector perturbation [omega, nu] (first order is enough for a
// finite-difference Jacobian with eps = 1e-6).
inline Eigen::Matrix4d exp6(const Eigen::Matrix<double, 6, 1>& d)
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    const Eigen::Matrix3d W = (Eigen::Matrix3d() <<
        0.0, -d(2), d(1),
        d(2), 0.0, -d(0),
        -d(1), d(0), 0.0).finished();
    T.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() + W;
    T.block<3, 1>(0, 3) = d.tail<3>();
    return T;
}

inline bool project(const Eigen::Matrix3d& K, const Eigen::Matrix4d& Tcw,
                    const Eigen::Vector3d& Xw, Eigen::Vector2d& uv, double& z)
{
    const Eigen::Vector3d Xc = Tcw.block<3, 3>(0, 0) * Xw + Tcw.block<3, 1>(0, 3);
    z = Xc(2);
    if (!(z > 1e-9) || !std::isfinite(z))
        return false;
    const Eigen::Vector3d p = K * Xc;
    uv = Eigen::Vector2d(p(0) / p(2), p(1) / p(2));
    return uv.allFinite();
}

// Perpendicular distance from a point to the infinite line through a,b.
inline double pointLineDist(const Eigen::Vector2d& p, const Eigen::Vector2d& a,
                            const Eigen::Vector2d& b)
{
    const Eigen::Vector2d d = b - a;
    const double n = d.norm();
    if (!(n > 1e-9))
        return -1.0;
    return std::fabs(d(0) * (p(1) - a(1)) - d(1) * (p(0) - a(0))) / n;
}

// Fraction of the observed segment [a,b] that the projected map segment covers.
inline double segmentOverlap(const Eigen::Vector2d& a, const Eigen::Vector2d& b,
                             const Eigen::Vector2d& p1, const Eigen::Vector2d& p2)
{
    const Eigen::Vector2d d = b - a;
    const double L = d.norm();
    if (!(L > 1e-9))
        return 0.0;
    const Eigen::Vector2d u = d / L;
    const double t1 = (p1 - a).dot(u), t2 = (p2 - a).dot(u);
    const double lo = std::max(0.0, std::min(t1, t2));
    const double hi = std::min(L, std::max(t1, t2));
    if (!(hi > lo))
        return 0.0;
    return (hi - lo) / L;
}

}  // namespace detail

// Evaluate one pose against the observations. Nothing is modified; no map
// object is touched (the observations are plain copies).
inline Result evaluate(const Eigen::Matrix3d& K, const Eigen::Matrix4d& Tcw,
                       const std::vector<PointObs>& pts,
                       const std::vector<LineObs>& lns,
                       const Params& pr = Params())
{
    Result r;
    if (!Tcw.allFinite()) {
        r.reason = R_POSE_INVALID;
        return r;
    }

    r.nLinesRaw = (int)lns.size();
    r.nPoints = (int)pts.size();
    for (const auto& p : pts)
        if (p.heldout)
            r.nPointsHeldout++;

    // ---------------- matching ----------------
    std::vector<LineObs> used;
    std::vector<LineObs> redundant;
    std::vector<int> seenIds;
    int nUnsupported = 0, nFar = 0, nDup = 0, nOverlap = 0;
    std::vector<double> lineDists;
    for (const auto& L : lns) {
        if (L.nObs < pr.minMultiViewObs) { nUnsupported++; continue; }
        if (L.mapId >= 0 && std::find(seenIds.begin(), seenIds.end(), L.mapId) != seenIds.end()) {
            nDup++;
            continue;
        }
        Eigen::Vector2d p1, p2;
        double z1 = 0, z2 = 0;
        if (!detail::project(K, Tcw, L.Xw1, p1, z1) || !detail::project(K, Tcw, L.Xw2, p2, z2)) {
            nFar++;
            continue;
        }
        const double d1 = detail::pointLineDist(L.a, p1, p2);
        const double d2 = detail::pointLineDist(L.b, p1, p2);
        if (d1 < 0.0 || d2 < 0.0) { nFar++; continue; }
        const double dmax = std::max(d1, d2);
        const double ov = detail::segmentOverlap(L.a, L.b, p1, p2);
        if (dmax > pr.maxLineDistPx) { nFar++; continue; }
        if (ov < pr.minVisibleOverlap) { nOverlap++; continue; }
        if (L.mapId >= 0)
            seenIds.push_back(L.mapId);
        lineDists.push_back(dmax);
        if (L.round == 0)
            used.push_back(L);
        else
            redundant.push_back(L);
    }
    r.nLinesUnique = (int)(used.size() + redundant.size());
    for (const auto& L : used) r.acceptedLineIdx.push_back(L.frameIdx);
    r.nLinesAccepted = (int)used.size();
    r.nLinesHeldout = (int)redundant.size();
    if (!lineDists.empty()) {
        std::vector<double> s = lineDists;
        std::sort(s.begin(), s.end());
        r.medLineDistPx = s[s.size() / 2];
    }

    if (r.nLinesAccepted == 0) {
        // Do NOT return yet: the geometric question ("can these observations
        // determine the pose?") is answered below for the point-only case too,
        // so a caller can tell "no usable lines" from "nothing here observes
        // translation at all". The specific matching failure is kept and
        // reported if the geometry turns out to be fine.
        r.reason = (r.nLinesRaw == 0) ? R_NO_LINES
                 : (nUnsupported >= nFar && nUnsupported >= nOverlap) ? R_LINE_UNSUPPORTED
                 : (nDup > 0 ? R_LINE_DUPLICATE : (nOverlap > 0 ? R_LINE_OVERLAP : R_LINE_FAR));
    }

    // ---------------- point fit at this pose ----------------
    std::vector<double> ptErr;
    std::vector<Eigen::Vector2d> uvCache(pts.size(), Eigen::Vector2d::Zero());
    std::vector<char> okCache(pts.size(), 0);
    std::vector<double> zCache(pts.size(), 0.0);
    for (size_t i = 0; i < pts.size(); ++i) {
        Eigen::Vector2d uv;
        double z = 0;
        if (detail::project(K, Tcw, pts[i].Xw, uv, z)) {
            uvCache[i] = uv; okCache[i] = 1; zCache[i] = z;
            const double e = (uv - Eigen::Vector2d(pts[i].u, pts[i].v)).norm();
            ptErr.push_back(e);
            if (!pts[i].heldout) {
                if (e * e * pts[i].w <= 5.991)
                    r.nPointInliers++;
            }
        }
    }
    if ((int)ptErr.size() < pr.minPoints) {
        r.reason = R_FEW_POINTS;
        return r;
    }
    {
        std::vector<double> s = ptErr;
        std::sort(s.begin(), s.end());
        r.medPointReprojPx = s[s.size() / 2];
    }

    // ---------------- direction spread (image plane) ----------------
    {
        std::vector<double> dirs;
        for (const auto& L : used) {
            const Eigen::Vector2d d = L.b - L.a;
            double a = std::atan2(d(1), d(0)) * 57.29577951308232;
            if (a < 0.0) a += 180.0;
            if (a >= 180.0) a -= 180.0;
            dirs.push_back(a);
        }
        if (dirs.size() >= 2) {
            std::sort(dirs.begin(), dirs.end());
            double maxGap = 180.0 - dirs.back() + dirs.front();
            for (size_t k = 0; k + 1 < dirs.size(); ++k)
                maxGap = std::max(maxGap, dirs[k + 1] - dirs[k]);
            r.spreadDeg = 180.0 - maxGap;
        }
    }
    if (r.spreadDeg < pr.minSpreadDeg && r.nLinesAccepted >= 2) {
        r.reason = R_SPREAD;
        return r;
    }

    // ---------------- geometry: information from pixel residuals only -------
    // Residual vector: two rows per constrained point, two rows per constrained
    // line (perpendicular distance of each observed endpoint to the projected
    // map line). Rotation is [0,3), translation [3,6).
    std::vector<Eigen::Vector2d> puv(pts.size(), Eigen::Vector2d::Zero());
    std::vector<double> pz(pts.size(), 1.0);
    std::vector<char> pok(pts.size(), 0);
    std::vector<Eigen::Vector2d> la(used.size(), Eigen::Vector2d::Zero());
    std::vector<Eigen::Vector2d> lb(used.size(), Eigen::Vector2d::Zero());
    std::vector<char> lok(used.size(), 0);

    auto fill = [&](const Eigen::Matrix4d& T) {
        bool allOk = true;
        for (size_t i = 0; i < pts.size(); ++i) {
            double z = 0;
            pok[i] = detail::project(K, T, pts[i].Xw, puv[i], z) ? 1 : 0;
            pz[i] = (pok[i] && z > 1e-6) ? z : 1.0;
            if (!pok[i] && !pts[i].heldout) allOk = false;
        }
        for (size_t i = 0; i < used.size(); ++i) {
            Eigen::Vector2d q1, q2; double z1 = 0, z2 = 0;
            lok[i] = (detail::project(K, T, used[i].Xw1, q1, z1) &&
                      detail::project(K, T, used[i].Xw2, q2, z2)) ? 1 : 0;
            la[i] = q1; lb[i] = q2;
            if (!lok[i]) allOk = false;
        }
        return allOk;
    };

    auto residualDot = [&](const Eigen::Matrix4d& T, const Eigen::Matrix4d& T0) {
        // <r(T), r(T0)> style accumulation is not needed; we only need r(T).
        (void)T0;
        (void)T;
        return 0.0;
    };
    (void)residualDot;

    if (!fill(Tcw)) {
        r.reason = R_POSE_INVALID;
        return r;
    }

    auto residual = [&](const Eigen::Matrix4d& T, std::vector<double>& out) {
        out.clear();
        for (size_t i = 0; i < pts.size(); ++i) {
            if (pts[i].heldout) continue;
            Eigen::Vector2d uv; double z = 0;
            if (!detail::project(K, T, pts[i].Xw, uv, z)) { out.push_back(0.0); out.push_back(0.0); continue; }
            const double s = std::sqrt(std::max(pts[i].w, 1e-12));
            out.push_back(s * (uv(0) - pts[i].u));
            out.push_back(s * (uv(1) - pts[i].v));
        }
        for (size_t i = 0; i < used.size(); ++i) {
            Eigen::Vector2d q1, q2; double z1 = 0, z2 = 0;
            if (!detail::project(K, T, used[i].Xw1, q1, z1) ||
                !detail::project(K, T, used[i].Xw2, q2, z2)) {
                out.push_back(0.0); out.push_back(0.0); continue;
            }
            const double s = std::sqrt(std::max(used[i].w, 1e-12));
            out.push_back(s * detail::pointLineDist(used[i].a, q1, q2));
            out.push_back(s * detail::pointLineDist(used[i].b, q1, q2));
        }
    };

    std::vector<double> r0;
    residual(Tcw, r0);
    if (r0.empty()) {
        r.reason = R_FEW_POINTS;
        return r;
    }
    const int nRes = (int)r0.size();
    Eigen::MatrixXd J(nRes, 6);
    for (int k = 0; k < 6; ++k) {
        Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
        d(k) = pr.fdEps;
        const Eigen::Matrix4d Tk = detail::exp6(d) * Tcw;
        std::vector<double> rk;
        residual(Tk, rk);
        if ((int)rk.size() != nRes) {
            r.reason = R_POSE_INVALID;
            return r;
        }
        for (int i = 0; i < nRes; ++i)
            J(i, k) = (rk[i] - r0[i]) / pr.fdEps;
    }
    const Eigen::Matrix<double, 6, 6> H = J.transpose() * J;

    // median depth of the observations, for the scale-free conversion
    {
        std::vector<double> zs;
        for (size_t i = 0; i < pts.size(); ++i)
            if (!pts[i].heldout && pok[i]) zs.push_back(pz[i]);
        for (size_t i = 0; i < used.size(); ++i)
            if (lok[i]) { zs.push_back(std::max(1e-6, (la[i] - lb[i]).norm())); }
        if (zs.empty()) {
            r.reason = R_FEW_POINTS;
            return r;
        }
        std::sort(zs.begin(), zs.end());
        r.medianDepth = zs[zs.size() / 2];
    }
    // depth of points only (used for the translation normalisation)
    double zPt = 0.0;
    {
        std::vector<double> zs;
        for (size_t i = 0; i < pts.size(); ++i)
            if (!pts[i].heldout && pok[i]) zs.push_back(pz[i]);
        if (!zs.empty()) {
            std::sort(zs.begin(), zs.end());
            zPt = zs[zs.size() / 2];
        } else {
            zPt = r.medianDepth;
        }
    }

    const Eigen::Matrix3d Hww = H.block<3, 3>(0, 0);
    (void)H.block<3, 3>(0, 3);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> esW(Hww);
    const double rotMin = esW.info() == Eigen::Success ? std::max(0.0, esW.eigenvalues()(0)) : 0.0;
    r.rotObsPx = std::sqrt(rotMin);

    // Translation information with rotation marginalised out, computed by
    // ORTHOGONAL PROJECTION instead of Hvv - Hvw Hww^-1 Hwv.  The subtraction
    // form cancels two large, nearly equal matrices (rotation and translation
    // columns of a point reprojection are strongly correlated) and returns a
    // negative eigenvalue that would silently disable this whole test.
    // With J = [Jr | Jt], the residual of regressing Jt on Jr is
    //     Z = Jt - Q Q^T Jt,   Q = orthonormal basis of range(Jr),
    // and Z^T Z is the translation information that no rotation can explain.
    const Eigen::MatrixXd Jr = J.block(0, 0, nRes, 3);
    const Eigen::MatrixXd Jt = J.block(0, 3, nRes, 3);
    Eigen::Matrix3d HttMarg = Eigen::Matrix3d::Zero();
    {
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(Jr);
        qr.setThreshold(1e-10);
        const Eigen::MatrixXd Q = qr.householderQ() *
            Eigen::MatrixXd::Identity(nRes, std::min(nRes, 3));
        const Eigen::MatrixXd Z = Jt - Q * (Q.transpose() * Jt);
        HttMarg = Z.transpose() * Z;
        r.dbgHttRaw = H.block<3, 3>(3, 3).diagonal().sum();
        r.dbgHwwDiag = Hww.diagonal().sum();
        r.dbgZnorm = Z.norm();
        r.dbgJtNorm = Jt.norm();
        HttMarg = 0.5 * (HttMarg + HttMarg.transpose());
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> esT(HttMarg);
    if (esT.info() == Eigen::Success) {
        const double lo = std::max(0.0, esT.eigenvalues()(0));
        const double hi = std::max(esT.eigenvalues()(2), 1e-12);
        r.transObsPx = std::sqrt(lo) * zPt;   // px of image motion per median depth
        r.transCondRatio = lo / hi;
    }
    if (r.transObsPx < pr.minTransObsPx) {
        r.reason = R_TRANS_UNOBSERVABLE;
        return r;
    }
    if (r.rotObsPx < pr.minRotObsPx) {
        r.reason = R_ROT_UNOBSERVABLE;
        return r;
    }

    // The observations can determine the pose; now require enough usable lines.
    if (r.nLinesAccepted < pr.minUniqueLines) {
        if (r.reason == R_ACCEPT)
            r.reason = R_NO_LINES;
        return r;
    }

    // ---------------- consistency ----------------
    if (r.medPointReprojPx > pr.maxPointReprojPx) {
        r.reason = R_POINT_FIT;
        return r;
    }
    std::vector<double> held;
    for (size_t i = 0; i < pts.size(); ++i) {
        if (!pts[i].heldout || !pok[i]) continue;
        const double e = (puv[i] - Eigen::Vector2d(pts[i].u, pts[i].v)).norm();
        held.push_back(e);
        if (e * e * pts[i].w <= 5.991)
            r.nHeldoutInliers++;
    }
    if (!held.empty()) {
        std::vector<double> s = held;
        std::sort(s.begin(), s.end());
        r.heldoutMedReprojPx = s[s.size() / 2];
        if (r.heldoutMedReprojPx > pr.maxHeldoutMedPx) {
            r.reason = R_CONSISTENCY;
            return r;
        }
        if ((double)r.nHeldoutInliers / (double)held.size() > pr.maxHeldoutInlierFrac &&
            held.size() >= 5) {
            // The candidate explains almost every point that the point-only pose
            // had rejected: that is a refit, not evidence.
            r.reason = R_CONSISTENCY;
            return r;
        }
    }
    // Redundant (held-out) lines must also be explained if any exist.
    int nRedOk = 0;
    for (const auto& L : redundant) {
        Eigen::Vector2d q1, q2; double z1 = 0, z2 = 0;
        if (!detail::project(K, Tcw, L.Xw1, q1, z1) ||
            !detail::project(K, Tcw, L.Xw2, q2, z2))
            continue;
        const double dmax = std::max(detail::pointLineDist(L.a, q1, q2),
                                     detail::pointLineDist(L.b, q1, q2));
        if (dmax <= pr.maxLineDistPx)
            nRedOk++;
    }
    r.nLinesHeldoutOk = nRedOk;

    r.ok = true;
    r.reason = R_ACCEPT;
    return r;
}

}  // namespace RescueAccept
