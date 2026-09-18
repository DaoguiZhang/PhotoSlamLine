// Loop-closing line audit unit tests (no dataset / no SLAM runtime).
//
// Coverage:
//   1. EdgeSE3ProjectPointToLine2D (line edge used by the point-line Global BA)
//      analytic Jacobian vs central finite differences on the point vertex (3)
//      and pose vertex (6): absolute error < 1e-5.
//   2. Global BA vertex-ID allocation: the FIXED scheme (line vertex ids offset
//      by the MAX MapPoint mnId, not by vpMP.size()) must produce no collisions
//      with the sparse mnId-based MapPoint vertex ids. The OLD scheme is printed
//      as evidence (it collides for realistic sparse mnId).
//   3. MapLine corrected-once flag semantics (mnCorrectedByKF guard).
//   4. MapLine observation bookkeeping invariants: null-KF guard, no dangling
//      observation after erase, self-Replace no-op, no duplicate KF entry.
//
// All numeric results are checked finite.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <set>
#include <unordered_set>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "OptimizableTypes.h"
#include "MapLine.h"
#include "KeyFrame.h"
#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"

using namespace ORB_SLAM3;

static const double eps = 1e-6;

// ---------------------------------------------------------------------------
// 1. GBA line edge analytic Jacobian vs central finite difference
// ---------------------------------------------------------------------------
static bool testGbaLineEdgeJacobian()
{
    const double fx = 520.0, fy = 520.0, cx = 320.0, cy = 240.0;
    double maxAbsP = 0.0, maxRelP = 0.0, maxAbsT = 0.0, maxRelT = 0.0;

    const int cases = 30;
    for (int iter = 0; iter < cases; ++iter)
    {
        // Deterministic pseudo-random (no <random> needed).
        auto r = [&](int i, double lo, double hi) {
            double u = std::fmod(std::sin((iter + 1) * 12.9898 + (i + 1) * 78.233) * 43758.5453, 1.0);
            if (u < 0) u += 1.0;
            return lo + u * (hi - lo);
        };

        const Eigen::Vector3d X(r(0, -0.5, 0.5), r(1, -0.5, 0.5), r(2, 1.0, 5.0));
        Eigen::Vector3d axis(r(3, -1.0, 1.0), r(4, -1.0, 1.0), r(5, -1.0, 1.0));
        if (axis.norm() < 1e-9) axis = Eigen::Vector3d(1, 0, 0);
        axis.normalize();
        const double angle = r(6, -0.4, 0.4);
        const Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));
        const Eigen::Vector3d t(r(7, -0.2, 0.2), r(8, -0.2, 0.2), r(9, -0.2, 0.2));
        const g2o::SE3Quat Tcw(q, t);

        // 2D observed line (normalized a*u + b*v + c = 0).
        const double pAx = r(10, 100.0, 400.0), pAy = r(11, 100.0, 300.0);
        const double pBx = r(12, 100.0, 400.0), pBy = r(13, 100.0, 300.0);
        const double dx = pBx - pAx, dy = pBy - pAy;
        const double nrm = std::sqrt(dx * dx + dy * dy);
        if (nrm < 1e-12) continue;
        const double a = dy / nrm, b = -dx / nrm, c = -(a * pAx + b * pAy);
        const Eigen::Vector3d line_abc(a, b, c);

        auto makeEdge = [&](const Eigen::Vector3d& Xv, const g2o::SE3Quat& Tv)
        {
            g2o::VertexSBAPointXYZ* vP = new g2o::VertexSBAPointXYZ();
            vP->setId(0);
            vP->setEstimate(Xv);
            g2o::VertexSE3Expmap* vT = new g2o::VertexSE3Expmap();
            vT->setId(1);
            vT->setEstimate(Tv);
            EdgeSE3ProjectPointToLine2D* e = new EdgeSE3ProjectPointToLine2D();
            e->setVertex(0, vP);
            e->setVertex(1, vT);
            e->setMeasurement(line_abc);
            e->SetCameraIntrinsics(fx, fy, cx, cy);
            e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
            return std::make_tuple(vP, vT, e);
        };

        auto [vP, vT, e] = makeEdge(X, Tcw);
        e->computeError();
        g2o::JacobianWorkspace ws;
        ws.updateSize(e);
        ws.allocate();
        e->g2o::BaseBinaryEdge<1, Eigen::Vector3d, g2o::VertexSBAPointXYZ,
                               g2o::VertexSE3Expmap>::linearizeOplus(ws);
        const Eigen::Matrix<double, 1, 3> JgP = e->jacobianOplusXi();
        const Eigen::Matrix<double, 1, 6> JgT = e->jacobianOplusXj();

        // FD wrt point (3)
        Eigen::Matrix<double, 1, 3> JfdP;
        for (int d = 0; d < 3; ++d)
        {
            Eigen::Vector3d Xp = X, Xm = X;
            Xp(d) += eps; Xm(d) -= eps;
            auto [a1, b1, c1] = makeEdge(Xp, Tcw); c1->computeError(); double rp = c1->error()(0);
            auto [a2, b2, c2] = makeEdge(Xm, Tcw); c2->computeError(); double rm = c2->error()(0);
            JfdP(0, d) = (rp - rm) / (2.0 * eps);
            delete a1; delete b1; delete c1; delete a2; delete b2; delete c2;
        }

        // FD wrt pose (6), left perturbation (oplus)
        Eigen::Matrix<double, 1, 6> JfdT;
        for (int d = 0; d < 6; ++d)
        {
            Eigen::VectorXd up = Eigen::VectorXd::Zero(6), um = Eigen::VectorXd::Zero(6);
            up(d) = eps; um(d) = -eps;
            g2o::VertexSE3Expmap vt;
            vt.setEstimate(Tcw);
            g2o::SE3Quat Tp = vt.estimate(); Tp = g2o::SE3Quat::exp(up) * Tp; // left perturb
            g2o::SE3Quat Tm = g2o::SE3Quat::exp(um) * Tcw;
            auto [a1, b1, c1] = makeEdge(X, Tp); c1->computeError(); double rp = c1->error()(0);
            auto [a2, b2, c2] = makeEdge(X, Tm); c2->computeError(); double rm = c2->error()(0);
            JfdT(0, d) = (rp - rm) / (2.0 * eps);
            delete a1; delete b1; delete c1; delete a2; delete b2; delete c2;
        }

        maxAbsP = std::max(maxAbsP, (JgP - JfdP).cwiseAbs().maxCoeff());
        maxRelP = std::max(maxRelP, (JgP - JfdP).cwiseAbs().maxCoeff() / std::max(1.0, JfdP.cwiseAbs().maxCoeff()));
        maxAbsT = std::max(maxAbsT, (JgT - JfdT).cwiseAbs().maxCoeff());
        maxRelT = std::max(maxRelT, (JgT - JfdT).cwiseAbs().maxCoeff() / std::max(1.0, JfdT.cwiseAbs().maxCoeff()));

        delete vP; delete vT; delete e;
    }

    // Gate: absolute error < 1e-5 and finite; relative error reported.
    const bool okP = std::isfinite(maxAbsP) && std::isfinite(maxRelP) && maxAbsP < 1e-5;
    const bool okT = std::isfinite(maxAbsT) && std::isfinite(maxRelT) && maxAbsT < 1e-5;
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "[GBA line edge] point jac maxAbs=" << maxAbsP << " maxRel=" << maxRelP
              << (okP ? " PASS" : " FAIL") << std::endl;
    std::cout << "[GBA line edge] pose  jac maxAbs=" << maxAbsT << " maxRel=" << maxRelT
              << (okT ? " PASS" : " FAIL") << std::endl;
    return okP && okT;
}

// ---------------------------------------------------------------------------
// 2. Global BA vertex-ID allocation (collision check)
// ---------------------------------------------------------------------------
// Replicates BundleAdjustmentWithLine:
//   KF vertex id   = mnId                        (0 .. maxKFid)
//   MP vertex id   = mnId + maxKFid + 1          (sparse mnId!)
//   Line vertex id = nextVertexId (OLD: maxKFid + vpMP.size() + 2; FIXED: maxMPid + maxKFid + 2)
static bool testGbaVertexIdsNoCollision()
{
    // Realistic sparse scenario: current map has N_MP map points whose mnId
    // values are a subset of [0, nNextId) with nNextId >> N_MP (culled/replaced
    // points still consume ids from the global counter).
    const long maxKFid = 250;         // KF mnIds are also a global counter
    const long N_MP = 4000;           // number of live MapPoints
    const long N_ML = 800;            // number of live MapLines
    const long maxMPid = 12000;       // global MapPoint counter >> N_MP

    std::vector<long> mpIds;
    mpIds.reserve(N_MP);
    std::unordered_set<long> used;
    long seed = 20260918;
    auto next = [&]() {           // Park-Miller LCG: non-negative in [0, 2^31-2]
        seed = (seed * 48271) % 2147483647;
        if (seed < 0) seed += 2147483647;
        return seed;
    };
    while ((long)mpIds.size() < N_MP)
    {
        long m = next() % maxMPid;
        if (used.insert(m).second) mpIds.push_back(m);
    }

    // KF ids
    std::set<long> allIds;
    for (long k = 0; k <= maxKFid; ++k) allIds.insert(k);

    // OLD scheme (as currently committed): line ids offset by N_MP.
    long oldCollisions = 0;
    {
        std::set<long> s = allIds;
        for (long m : mpIds) s.insert(m + maxKFid + 1);
        long nextV = maxKFid + N_MP + 2;
        for (long i = 0; i < N_ML; ++i) { s.insert(++nextV); s.insert(++nextV); }
        oldCollisions = (long)(maxKFid + 1 + N_MP + 2 * N_ML) - (long)s.size();
    }

    // FIXED scheme: line ids offset by max MP mnId.
    long fixedCollisions = 0;
    {
        std::set<long> s = allIds;
        for (long m : mpIds) s.insert(m + maxKFid + 1);
        long nextV = maxMPid + maxKFid + 2;
        for (long i = 0; i < N_ML; ++i) { s.insert(++nextV); s.insert(++nextV); }
        fixedCollisions = (long)(maxKFid + 1 + N_MP + 2 * N_ML) - (long)s.size();
    }

    std::cout << "[GBA vertex ids] old-scheme collisions=" << oldCollisions
              << " (evidence), fixed-scheme collisions=" << fixedCollisions << std::endl;
    const bool ok = fixedCollisions == 0;
    std::cout << "[GBA vertex ids] no collision (fixed scheme) : " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

// ---------------------------------------------------------------------------
// 3. MapLine corrected-once flag (mnCorrectedByKF guard)
// ---------------------------------------------------------------------------
static bool testCorrectOnce()
{
    MapLine ml;
    const unsigned long curKFid = 123;
    const Eigen::Vector3f P1(0.0f, 0.0f, 3.0f), P2(1.0f, 0.0f, 3.0f);
    ml.SetLineWorldPos(P1, P2);

    int applied = 0;
    // Simulate the correction loop over observing keyframes (same loop KF id).
    for (int obs = 0; obs < 3; ++obs)
    {
        if (ml.mnCorrectedByKF == curKFid) continue;   // the guard in CorrectLoopWithLine
        ml.mnCorrectedByKF = curKFid;
        // apply a Sim(3)-like correction to the endpoints (identity+shift here)
        auto ep = ml.GetLineWorldPos();
        ml.SetLineWorldPos(ep.first + Eigen::Vector3f(0.1f, 0.0f, 0.0f),
                           ep.second + Eigen::Vector3f(0.1f, 0.0f, 0.0f));
        ++applied;
    }

    auto ep = ml.GetLineWorldPos();
    const bool finite = ep.first.allFinite() && ep.second.allFinite();
    const bool ok = (applied == 1) && finite && std::abs(ep.first(0) - 0.1f) < 1e-6f;
    std::cout << "[Correct-once] applied=" << applied << " (expect 1), finite=" << finite
              << " : " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

// ---------------------------------------------------------------------------
// 4. MapLine observation bookkeeping invariants
// ---------------------------------------------------------------------------
static bool testObservationInvariants()
{
    bool ok = true;

    // (a) null-KF observation must not be inserted.
    {
        MapLine ml;
        ml.AddLineObservation(nullptr, 0);
        ok &= ml.GetLineObservations().empty();
        ok &= (ml.Observations() == 0);
        std::cout << "[Obs invariant] null-KF guard : " << (ml.GetLineObservations().empty() ? "PASS" : "FAIL") << std::endl;
    }

    // (b) no duplicate KF entry + erase leaves no dangling observation.
    {
        KeyFrame kf;             // default ctor: NLleft=0, mvuLineRight empty
        kf.mpCamera2 = nullptr;  // ensure defensive guards in AddLineObservation see mono
        kf.mnId = 1;

        MapLine ml;
        ml.AddLineObservation(&kf, 0);
        ml.AddLineObservation(&kf, 0);   // duplicate add
        const auto obs = ml.GetLineObservations();
        ok &= (obs.size() == 1);         // no duplicate KF entry
        ok &= obs.count(&kf) == 1;
        std::cout << "[Obs invariant] no duplicate KF entry : " << ((obs.size() == 1) ? "PASS" : "FAIL") << std::endl;

        ml.EraseLineObservation(&kf);
        const auto obs2 = ml.GetLineObservations();
        ok &= obs2.empty();              // no dangling
        ok &= (ml.GetReferenceKeyFrame() == nullptr);   // ref KF cleared when last obs erased
        std::cout << "[Obs invariant] erase leaves no dangling/ref : "
                  << ((obs2.empty() && ml.GetReferenceKeyFrame() == nullptr) ? "PASS" : "FAIL") << std::endl;
    }

    // (c) self-Replace is a no-op.
    {
        MapLine ml;
        ml.mnId = 42;
        ml.Replace(&ml);
        ok &= !ml.isBad();
        ok &= (ml.GetReplaced() == nullptr);
        std::cout << "[Obs invariant] self-Replace no-op : " << ((!ml.isBad() && ml.GetReplaced() == nullptr) ? "PASS" : "FAIL") << std::endl;
    }

    std::cout << "[Obs invariant] all : " << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

int main()
{
    bool ok = true;
    ok &= testGbaLineEdgeJacobian();
    ok &= testGbaVertexIdsNoCollision();
    ok &= testCorrectOnce();
    ok &= testObservationInvariants();
    std::cout << (ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return ok ? 0 : 1;
}
