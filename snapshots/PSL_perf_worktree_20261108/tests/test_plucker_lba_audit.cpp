// Minimal audit test for the Plucker line edge used by
// LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg:
//   EdgeSE3ProjectLine4D  (pose vertex + VertexLine4D, 2D residual)
//
// Coverage (per diagnosis requirement):
//   1. analytic pose Jacobian (6) and line Jacobian (4) vs central finite
//      difference, under pure-translation, pure-rotation and general poses.
//   2. degenerate line (zero direction v / zero normal n): VertexLine4D::oplus
//      is a safe no-op and computeError stays finite (no NaN).
//   3. a small g2o optimization (4 fixed poses + 1 free line) must reduce the
//      reprojection error and stay finite.
//
// No ORB-SLAM runtime required; header-only edge/vertex + g2o symbols from
// libORB_SLAM3.so.

#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>
#include <vector>

#include <Eigen/Dense>

#include "OptimizableTypes.h"
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/core/sparse_optimizer.h"

using namespace ORB_SLAM3;

static const double eps = 1e-6;

static Eigen::Vector2d projectPt(const g2o::SE3Quat& Tcw, const Eigen::Vector3d& Xw,
                                 double fx, double fy, double cx, double cy)
{
    Eigen::Vector3d Xc = Tcw.map(Xw);
    return Eigen::Vector2d(fx * Xc(0) / Xc(2) + cx, fy * Xc(1) / Xc(2) + cy);
}

// Central finite difference of a pose vertex (6 DoF, SE3 exp).
static Eigen::MatrixXd fdPose(g2o::VertexSE3Expmap& v, const std::function<Eigen::VectorXd()>& f, int m)
{
    Eigen::VectorXd e0 = f();
    Eigen::MatrixXd J(m, 6);
    for (int d = 0; d < 6; ++d)
    {
        Eigen::VectorXd dp = Eigen::VectorXd::Zero(6), dm = Eigen::VectorXd::Zero(6);
        dp(d) = eps; dm(d) = -eps;
        auto saved = v.estimate();
        v.oplusImpl(dp.data()); Eigen::VectorXd ep = f(); v.setEstimate(saved);
        v.oplusImpl(dm.data()); Eigen::VectorXd em = f(); v.setEstimate(saved);
        J.col(d) = (ep - em) / (2.0 * eps);
    }
    return J;
}

// Central finite difference of the 4-DoF orthonormal line vertex.
static Eigen::MatrixXd fdLine(VertexLine4D& v, const std::function<Eigen::VectorXd()>& f, int m)
{
    Eigen::VectorXd e0 = f();
    Eigen::MatrixXd J(m, 4);
    for (int d = 0; d < 4; ++d)
    {
        Eigen::VectorXd dp = Eigen::VectorXd::Zero(4), dm = Eigen::VectorXd::Zero(4);
        dp(d) = eps; dm(d) = -eps;
        auto saved = v.estimate();
        v.oplusImpl(dp.data()); Eigen::VectorXd ep = f(); v.setEstimate(saved);
        v.oplusImpl(dm.data()); Eigen::VectorXd em = f(); v.setEstimate(saved);
        J.col(d) = (ep - em) / (2.0 * eps);
    }
    return J;
}

struct Case { const char* name; g2o::SE3Quat pose; };

static bool checkJac(const char* tag, const g2o::SE3Quat& pose,
                     double& maxAbsP, double& maxRelP, double& maxAbsL, double& maxRelL)
{
    const double fx = 525.0, fy = 525.0, cx = 320.0, cy = 240.0;
    Eigen::Vector3d X1(-0.4, 0.3, 3.0), X2(0.5, -0.2, 3.5);
    Eigen::Vector3d dir = (X2 - X1).normalized();
    Eigen::Vector3d n = X1.cross(dir);   // moment
    Eigen::Vector3d v = dir;             // direction
    Eigen::Matrix<double,6,1> Lw; Lw << n, v;

    g2o::VertexSE3Expmap vPose; vPose.setEstimate(pose);
    VertexLine4D vLine; vLine.setEstimate(Lw);

    EdgeSE3ProjectLine4D edge;
    edge.setVertex(0, &vPose);
    edge.setVertex(1, &vLine);
    edge.SetCameraIntrinsics(fx, fy, cx, cy);
    Eigen::Vector2d u1 = projectPt(pose, X1, fx, fy, cx, cy);
    Eigen::Vector2d u2 = projectPt(pose, X2, fx, fy, cx, cy);
    edge.setMeasurement(std::make_pair(u1, u2));

    g2o::JacobianWorkspace ws; ws.updateSize(&edge); ws.allocate();
    edge.g2o::BaseBinaryEdge<2, std::pair<Eigen::Vector2d,Eigen::Vector2d>,
                             g2o::VertexSE3Expmap, VertexLine4D>::linearizeOplus(ws);
    Eigen::MatrixXd aP = edge.jacobianOplusXi().cast<double>();
    Eigen::MatrixXd aL = edge.jacobianOplusXj().cast<double>();

    auto errF = [&]() { edge.computeError(); return edge.error().cast<double>(); };
    Eigen::MatrixXd nP = fdPose(vPose, errF, 2);
    Eigen::MatrixXd nL = fdLine(vLine, errF, 2);

    maxAbsP = maxRelP = maxAbsL = maxRelL = 0.0;
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 6; ++c) {
            double a = std::abs(aP(r,c) - nP(r,c));
            double rel = a / std::max(1e-12, std::abs(nP(r,c)));
            maxAbsP = std::max(maxAbsP, a); maxRelP = std::max(maxRelP, rel);
        }
    for (int r = 0; r < 2; ++r)
        for (int c = 0; c < 4; ++c) {
            double a = std::abs(aL(r,c) - nL(r,c));
            double rel = a / std::max(1e-12, std::abs(nL(r,c)));
            maxAbsL = std::max(maxAbsL, a); maxRelL = std::max(maxRelL, rel);
        }

    // Correctness gate: finite and absolute error < 1e-5 for every entry.
    // (Relative error is reported but is ill-defined for near-zero Jacobian
    //  entries, e.g. the v-momentum column becomes unobservable when t = 0 in
    //  pure rotation, where the numeric Jacobian entry is ~5e-8.)
    bool okP = std::isfinite(maxAbsP) && std::isfinite(maxRelP) && maxAbsP < 1e-5;
    bool okL = std::isfinite(maxAbsL) && std::isfinite(maxRelL) && maxAbsL < 1e-5;
    std::cout << std::left << std::setw(34) << tag
              << " | pose maxAbs=" << std::scientific << std::setprecision(2) << maxAbsP
              << " maxRel=" << maxRelP << (okP ? " PASS" : " FAIL")
              << " | line maxAbs=" << maxAbsL << " maxRel=" << maxRelL << (okL ? " PASS" : " FAIL")
              << std::endl;
    return okP && okL;
}

// Residual-units test: for a horizontal image line, a 1-pixel perpendicular
// endpoint shift must change the residual by ~1 pixel, for two different
// resolutions / intrinsic sets.
static bool checkPixelUnits()
{
    bool ok = true;
    struct Intrinsics { double fx, fy, cx, cy; };
    const Intrinsics sets[2] = {{525.0, 525.0, 320.0, 240.0},
                                {1050.0, 1050.0, 640.0, 480.0}};
    for (const auto& K : sets)
    {
        // 3D line parallel to image x-axis at depth 3 (image line is horizontal).
        Eigen::Vector3d X1(0.0, 0.0, 3.0), X2(1.0, 0.0, 3.0);
        Eigen::Vector3d dir = (X2 - X1).normalized();
        Eigen::Vector3d n = X1.cross(dir);
        Eigen::Vector3d v = dir;
        Eigen::Matrix<double,6,1> Lw; Lw << n, v;

        g2o::VertexSE3Expmap vPose; vPose.setEstimate(g2o::SE3Quat());
        VertexLine4D vLine; vLine.setEstimate(Lw);
        EdgeSE3ProjectLine4D edge;
        edge.setVertex(0, &vPose);
        edge.setVertex(1, &vLine);
        edge.SetCameraIntrinsics(K.fx, K.fy, K.cx, K.cy);

        Eigen::Vector2d u1 = projectPt(g2o::SE3Quat(), X1, K.fx, K.fy, K.cx, K.cy);
        Eigen::Vector2d u2 = projectPt(g2o::SE3Quat(), X2, K.fx, K.fy, K.cx, K.cy);
        edge.setMeasurement(std::make_pair(u1, u2));
        edge.computeError();
        double e0 = edge.error().norm();

        // 1-pixel perpendicular shift of endpoint 1 (along +y, perpendicular to
        // the horizontal line): residual must become ~1.
        edge.setMeasurement(std::make_pair(Eigen::Vector2d(u1(0), u1(1) + 1.0), u2));
        edge.computeError();
        double eperp = std::abs(edge.error()(0));

        // 1-pixel shift along the line (+x, parallel): perpendicular residual must
        // stay ~0.
        edge.setMeasurement(std::make_pair(Eigen::Vector2d(u1(0) + 1.0, u1(1)), u2));
        edge.computeError();
        double epar = edge.error().norm();

        bool pass = (e0 < 1e-6) && std::abs(eperp - 1.0) < 1e-6 && (epar < 1e-6);
        std::cout << "pixel-units fx=" << K.fx << " cx=" << K.cx
                  << " | e0=" << e0 << " e(1px perp)=" << eperp
                  << " e(1px par)=" << epar << (pass ? " PASS" : " FAIL") << std::endl;
        ok &= pass;
    }
    return ok;
}

int main()
{
    bool ok = true;
    std::cout << "===== Plucker LBA edge audit =====" << std::endl;

    // --- 1. Jacobian under pure translation / pure rotation / general ---
    {
        double aP, rP, aL, rL;
        ok &= checkJac("pure translation",
            g2o::SE3Quat(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.1, -0.05, 0.2)), aP, rP, aL, rL);
        ok &= checkJac("pure rotation",
            g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(0.35, Eigen::Vector3d(0.3, -0.2, 0.9).normalized())),
                         Eigen::Vector3d::Zero()), aP, rP, aL, rL);
        ok &= checkJac("general pose",
            g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(0.25, Eigen::Vector3d(0.5, 0.6, -0.3).normalized())),
                         Eigen::Vector3d(0.12, 0.07, -0.11)), aP, rP, aL, rL);
    }

    // --- 2. residual units (pixels) ---
    ok &= checkPixelUnits();

    // --- 3. degenerate line safety ---
    {
        const double fx = 525.0, fy = 525.0, cx = 320.0, cy = 240.0;
        g2o::VertexSE3Expmap vPose; vPose.setEstimate(g2o::SE3Quat());
        EdgeSE3ProjectLine4D edge;
        edge.setVertex(0, &vPose);
        edge.SetCameraIntrinsics(fx, fy, cx, cy);
        edge.setMeasurement(std::make_pair(Eigen::Vector2d(10, 10), Eigen::Vector2d(50, 50)));

        // zero direction v
        {
            VertexLine4D vL;
            Eigen::Matrix<double,6,1> L; L << 1.0, 2.0, 3.0, 0.0, 0.0, 0.0;
            vL.setEstimate(L);
            edge.setVertex(1, &vL);
            Eigen::Vector4d upd(0.1, -0.2, 0.3, 0.4);
            auto before = vL.estimate();
            vL.oplusImpl(upd.data());
            bool noChange = vL.estimate().isApprox(before, 1e-12);
            edge.computeError();
            bool finite = edge.error().allFinite();
            std::cout << "zero-v: oplus no-op=" << (noChange ? "PASS" : "FAIL")
                      << " error finite=" << (finite ? "PASS" : "FAIL") << std::endl;
            ok &= noChange && finite;
        }
        // zero normal n
        {
            VertexLine4D vL;
            Eigen::Matrix<double,6,1> L; L << 0.0, 0.0, 0.0, 1.0, 2.0, 3.0;
            vL.setEstimate(L);
            edge.setVertex(1, &vL);
            Eigen::Vector4d upd(0.1, -0.2, 0.3, 0.4);
            auto before = vL.estimate();
            vL.oplusImpl(upd.data());
            bool noChange = vL.estimate().isApprox(before, 1e-12);
            edge.computeError();
            bool finite = edge.error().allFinite();
            std::cout << "zero-n: oplus no-op=" << (noChange ? "PASS" : "FAIL")
                      << " error finite=" << (finite ? "PASS" : "FAIL") << std::endl;
            ok &= noChange && finite;
        }
    }

    // --- 4. optimization must reduce reprojection error ---
    {
        const double fx = 525.0, fy = 525.0, cx = 320.0, cy = 240.0;
        Eigen::Vector3d dir(1.0, 0.2, 0.1); dir.normalize();
        Eigen::Vector3d p0(-0.2, 0.1, 3.0);
        Eigen::Vector3d p1 = p0 + 2.0 * dir;

        // 4 fixed poses around the line
        std::vector<g2o::SE3Quat> poses = {
            g2o::SE3Quat(),
            g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(0.4, Eigen::Vector3d(0,1,0))), Eigen::Vector3d(0.5, 0, 0)),
            g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(-0.5, Eigen::Vector3d(0,1,0))), Eigen::Vector3d(-0.4, 0.1, 0)),
            g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(0.3, Eigen::Vector3d(1,0,0))), Eigen::Vector3d(0, -0.3, 0.2)),
        };

        using BlockSolverX = g2o::BlockSolver<g2o::BlockSolverTraits<-1,-1>>;
        using LinearSolverX = g2o::LinearSolverEigen<BlockSolverX::PoseMatrixType>;
        g2o::SparseOptimizer opt;
        auto* solver = new g2o::OptimizationAlgorithmLevenberg(new BlockSolverX(new LinearSolverX()));
        opt.setAlgorithm(solver);
        opt.setVerbose(false);

        // true line (n = moment, v = direction)
        Eigen::Vector3d n0 = p0.cross(dir);
        Eigen::Vector3d v0 = dir;
        // perturbed initial line estimate
        Eigen::Vector3d n_init = n0 + Eigen::Vector3d(0.02, -0.01, 0.015);
        Eigen::Vector3d v_init = (dir + Eigen::Vector3d(0.05, -0.04, 0.03)).normalized();

        // vertices
        std::vector<g2o::VertexSE3Expmap*> vposes;
        for (size_t i = 0; i < poses.size(); ++i) {
            auto* vp = new g2o::VertexSE3Expmap();
            vp->setEstimate(poses[i]);
            vp->setId((int)i);
            vp->setFixed(true);
            opt.addVertex(vp);
            vposes.push_back(vp);
        }
        auto* vl = new VertexLine4D();
        Eigen::Matrix<double,6,1> Linit; Linit << n_init, v_init;
        vl->setEstimate(Linit);
        vl->setId((int)poses.size());
        opt.addVertex(vl);

        double before = 0.0;
        std::vector<EdgeSE3ProjectLine4D*> vEdges;
        for (size_t i = 0; i < poses.size(); ++i) {
            auto* e = new EdgeSE3ProjectLine4D();
            e->setVertex(0, vposes[i]);
            e->setVertex(1, vl);
            e->SetCameraIntrinsics(fx, fy, cx, cy);
            Eigen::Vector2d u1 = projectPt(poses[i], p0, fx, fy, cx, cy);
            Eigen::Vector2d u2 = projectPt(poses[i], p1, fx, fy, cx, cy);
            e->setMeasurement(std::make_pair(u1, u2));
            e->setInformation(Eigen::Matrix2d::Identity());
            opt.addEdge(e);
            vEdges.push_back(e);
            e->computeError();
            before += e->chi2();
        }
        opt.initializeOptimization();
        opt.optimize(10);
        double after = 0.0;
        for (auto* e : vEdges) {
            e->computeError();
            after += e->chi2();
        }
        bool decr = after < before;
        bool finite = std::isfinite(after) && std::isfinite(before);
        std::cout << std::scientific << std::setprecision(4)
                  << "optimize: before chi2=" << before << " after chi2=" << after
                  << " decrease=" << (decr ? "PASS" : "FAIL")
                  << " finite=" << (finite ? "PASS" : "FAIL") << std::endl;
        ok &= decr && finite;
    }

    std::cout << "===== " << (ok ? "ALL PASS" : "SOME FAIL") << " =====" << std::endl;
    return ok ? 0 : 1;
}
