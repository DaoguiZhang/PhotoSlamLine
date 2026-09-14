/**
 * Deterministic verification for the line-aware loop closing additions:
 *
 *   1. Sim(3) endpoint transform vs. direct Plücker transform give the SAME
 *      infinite line (n_ep, v_ep) == (n_dir, v_dir) under X' = s*R*X + t with
 *      Plücker convention v = P2-P1, n = P1 x v.
 *
 *   2. EdgeSim3ProjectPointToLine2D residual: g2o numerical Jacobian
 *      (edge.linearizeOplus) vs central finite differences on the point vertex
 *      (3 dof) and the Sim(3) vertex (7 dof, left-multiplied perturbation
 *      s(update)*estimate as in VertexSim3Expmap::oplusImpl).
 *
 * No ORB-SLAM runtime or dataset is required.
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "OptimizableTypes.h"
#include "Converter.h"
#include "Thirdparty/g2o/g2o/core/jacobian_workspace.h"

using namespace ORB_SLAM3;

static std::mt19937 rng(20260914);

static Eigen::Vector3d randVec3(double lo, double hi)
{
    std::uniform_real_distribution<double> dist(lo, hi);
    return Eigen::Vector3d(dist(rng), dist(rng), dist(rng));
}

static g2o::Sim3 randomSim3()
{
    std::uniform_real_distribution<double> r(-0.3, 0.3);
    std::uniform_real_distribution<double> t(-0.2, 0.2);
    std::uniform_real_distribution<double> s(0.8, 1.25);
    Eigen::Vector3d axis(r(rng), r(rng), r(rng));
    double angle = 0.1 + 0.3 * std::abs(r(rng));
    if (axis.norm() < 1e-9) axis = Eigen::Vector3d(1, 0, 0);
    axis.normalize();
    Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));
    return g2o::Sim3(q, Eigen::Vector3d(t(rng), t(rng), t(rng)), s(rng));
}

// Analytic Sim(3) transform of a Plücker line L=[n;v] (v=P2-P1, n=P1 x v).
static void transformPluckerDirect(const g2o::Sim3& S, const Eigen::Vector3d& n,
                                   const Eigen::Vector3d& v,
                                   Eigen::Vector3d& nOut, Eigen::Vector3d& vOut)
{
    const double s = S.scale();
    const Eigen::Matrix3d R = S.rotation().toRotationMatrix();
    const Eigen::Vector3d t = S.translation();
    const Eigen::Vector3d Rv = R * v;
    vOut = s * Rv;
    nOut = s * s * R * n + s * (t.cross(Rv));
}

static bool testPluckerConsistency()
{
    bool ok = true;
    for (int iter = 0; iter < 50; ++iter)
    {
        const Eigen::Vector3d P1 = randVec3(-2.0, 2.0);
        const Eigen::Vector3d P2 = P1 + randVec3(0.1, 3.0);
        const g2o::Sim3 S = randomSim3();

        // Endpoint path (what the loop correction does).
        const Eigen::Vector3d Q1 = S.map(P1);
        const Eigen::Vector3d Q2 = S.map(P2);
        Eigen::Vector3d n_ep, v_ep;
        Converter::LineSegmentToPlucker(Q1, Q2, n_ep, v_ep);

        // Direct Plücker transform.
        Eigen::Vector3d n0, v0;
        Converter::LineSegmentToPlucker(P1, P2, n0, v0);
        Eigen::Vector3d n_dir, v_dir;
        transformPluckerDirect(S, n0, v0, n_dir, v_dir);

        const double errN = (n_ep - n_dir).norm();
        const double errV = (v_ep - v_dir).norm();
        const double scale = std::max({1.0, n_dir.norm(), v_dir.norm()});
        if (errN > 1e-9 * scale || errV > 1e-9 * scale)
        {
            std::cerr << "Plucker consistency FAIL at iter " << iter
                      << ": errN=" << errN << " errV=" << errV << std::endl;
            ok = false;
        }
    }
    std::cout << "[Plucker consistency] endpoint-transform == direct-Plucker-transform : "
              << (ok ? "PASS" : "FAIL") << std::endl;
    return ok;
}

static bool testSim3LineEdgeJacobian()
{
    const double fx = 520.0, fy = 520.0, cx = 320.0, cy = 240.0;
    const double eps = 1e-6;

    double maxAbsPoint = 0.0, maxRelPoint = 0.0;
    double maxAbsSim3 = 0.0, maxRelSim3 = 0.0;

    for (int iter = 0; iter < 30; ++iter)
    {
        const Eigen::Vector3d X = randVec3(0.5, 3.0);
        const g2o::Sim3 S = randomSim3();

        // 2D observation line (normalized a*u + b*v + c = 0).
        const Eigen::Vector2d pA(randVec3(50.0, 500.0)(0), randVec3(50.0, 400.0)(1));
        const Eigen::Vector2d pB(randVec3(50.0, 500.0)(0), randVec3(50.0, 400.0)(1));
        Eigen::Vector2d dir2 = pB - pA;
        if (dir2.norm() < 1e-6) continue;
        const double nrm = dir2.norm();
        const double a = dir2(1) / nrm;
        const double b = -dir2(0) / nrm;
        const double c = -(a * pA(0) + b * pA(1));
        const Eigen::Vector3d line_abc(a, b, c);

        auto makeEdge = [&](const Eigen::Vector3d& Xv, const g2o::Sim3& Sv, bool fixPoint)
        {
            ORB_SLAM3::VertexSim3Expmap* vS = new ORB_SLAM3::VertexSim3Expmap();
            vS->setId(0);
            vS->setFixed(false);
            vS->_fix_scale = false;
            vS->setEstimate(Sv);
            vS->pCamera1 = nullptr;
            vS->pCamera2 = nullptr;

            g2o::VertexSBAPointXYZ* vP = new g2o::VertexSBAPointXYZ();
            vP->setId(1);
            vP->setFixed(fixPoint);
            vP->setEstimate(Xv);

            EdgeSim3ProjectPointToLine2D* e = new EdgeSim3ProjectPointToLine2D();
            e->setVertex(0, vP);
            e->setVertex(1, vS);
            e->setMeasurement(line_abc);
            e->SetCameraIntrinsics(fx, fy, cx, cy);
            e->setInformation(Eigen::Matrix<double, 1, 1>::Identity());

            return std::make_tuple(vS, vP, e);
        };

        // ---- 1. closed-form residual cross-check ----
        {
            auto [vS, vP, e] = makeEdge(X, S, true);
            e->computeError();
            const Eigen::Vector3d Xc = S.map(X);
            const double u = fx * Xc(0) / Xc(2) + cx;
            const double v = fy * Xc(1) / Xc(2) + cy;
            const double r_manual = a * u + b * v + c;
            if (std::abs(e->error()(0) - r_manual) > 1e-9)
            {
                std::cerr << "closed-form residual mismatch at iter " << iter << std::endl;
                return false;
            }
            delete e; delete vS; delete vP;
        }

        // ---- 2. g2o numerical Jacobian vs central FD (both vertices free) ----
        auto [vS, vP, e] = makeEdge(X, S, false);
        e->computeError();

        g2o::JacobianWorkspace ws;
        ws.updateSize(e);
        ws.allocate();
        e->linearizeOplus(ws);
        const Eigen::Matrix<double, 1, 3>& JgPoint = e->jacobianOplusXi();
        const Eigen::Matrix<double, 1, 7>& JgSim3 = e->jacobianOplusXj();

        Eigen::Matrix<double, 1, 3> JfdPoint;
        for (int d = 0; d < 3; ++d)
        {
            Eigen::Vector3d Xp = X, Xm = X;
            Xp(d) += eps;
            Xm(d) -= eps;
            auto [s1, p1, e1] = makeEdge(Xp, S, false);
            e1->computeError();
            const double rp = e1->error()(0);
            auto [s2, p2, e2] = makeEdge(Xm, S, false);
            e2->computeError();
            const double rm = e2->error()(0);
            JfdPoint(0, d) = (rp - rm) / (2.0 * eps);
            delete e1; delete e2; delete s1; delete p1; delete s2; delete p2;
        }

        Eigen::Matrix<double, 1, 7> JfdSim3;
        for (int d = 0; d < 7; ++d)
        {
            Eigen::Matrix<double, 7, 1> up = Eigen::Matrix<double, 7, 1>::Zero();
            Eigen::Matrix<double, 7, 1> um = Eigen::Matrix<double, 7, 1>::Zero();
            up(d) = eps;
            um(d) = -eps;
            g2o::Sim3 Sp = g2o::Sim3(up) * S;
            g2o::Sim3 Sm = g2o::Sim3(um) * S;
            auto [s1, p1, e1] = makeEdge(X, Sp, false);
            e1->computeError();
            const double rp = e1->error()(0);
            auto [s2, p2, e2] = makeEdge(X, Sm, false);
            e2->computeError();
            const double rm = e2->error()(0);
            JfdSim3(0, d) = (rp - rm) / (2.0 * eps);
            delete e1; delete e2; delete s1; delete p1; delete s2; delete p2;
        }

        const double errPoint = (JgPoint - JfdPoint).cwiseAbs().maxCoeff();
        const double relPoint = errPoint / std::max(1.0, JfdPoint.cwiseAbs().maxCoeff());
        const double errSim3 = (JgSim3 - JfdSim3).cwiseAbs().maxCoeff();
        const double relSim3 = errSim3 / std::max(1.0, JfdSim3.cwiseAbs().maxCoeff());
        maxAbsPoint = std::max(maxAbsPoint, errPoint);
        maxRelPoint = std::max(maxRelPoint, relPoint);
        maxAbsSim3 = std::max(maxAbsSim3, errSim3);
        maxRelSim3 = std::max(maxRelSim3, relSim3);

        delete e; delete vS; delete vP;
    }

    const double passRel = 1e-3;
    const bool okPoint = maxRelPoint < passRel;
    const bool okSim3 = maxRelSim3 < passRel;
    std::cout << std::scientific << std::setprecision(3);
    std::cout << "[Sim3 line edge] jacobian wrt point maxAbs=" << maxAbsPoint
              << " maxRel=" << maxRelPoint << " : " << (okPoint ? "PASS" : "FAIL") << std::endl;
    std::cout << "[Sim3 line edge] jacobian wrt Sim3  maxAbs=" << maxAbsSim3
              << " maxRel=" << maxRelSim3 << " : " << (okSim3 ? "PASS" : "FAIL") << std::endl;
    return okPoint && okSim3;
}

int main()
{
    bool ok = true;
    ok &= testPluckerConsistency();
    ok &= testSim3LineEdgeJacobian();
    std::cout << (ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return ok ? 0 : 1;
}
