/**
 * Deterministic analytic-Jacobian vs finite-difference check for the line edges
 * used by the Stereo + Line pipeline:
 *
 *   1. EdgeSE3ProjectLineXYZOnlyPose_PointToLine   (pose-only, 2D residual)
 *        - used as the monocular/fallback line edge in PoseOptimizationWithLine.
 *   2. EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine (pose-only, 4D residual)
 *        - used as the stereo line edge in PoseOptimizationWithLine.
 *   3. EdgeSE3ProjectLine4D (pose + line vertex, 2D residual)
 *        - Plucker line edge used in LocalBundleAdjustmentWithLine_Optimization_Plucker_Reg.
 *
 * No ORB-SLAM runtime is required; only the header-only edge definitions and
 * g2o symbols (linked via libORB_SLAM3.so) are used.
 */

#include <iostream>
#include <iomanip>
#include <random>
#include <cmath>

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>

#include "OptimizableTypes.h"

using namespace ORB_SLAM3;

static std::mt19937 rng(12345);

static Eigen::Vector3d randVec3(double lo, double hi)
{
    std::uniform_real_distribution<double> dist(lo, hi);
    return Eigen::Vector3d(dist(rng), dist(rng), dist(rng));
}

static g2o::SE3Quat randomPose()
{
    // Small but non-trivial perturbation around identity.
    std::uniform_real_distribution<double> r(-0.2, 0.2);
    std::uniform_real_distribution<double> t(-0.15, 0.15);
    Eigen::Vector3d axis(r(rng), r(rng), r(rng));
    double angle = 0.1 + 0.2 * std::abs(r(rng));
    if (axis.norm() < 1e-9) axis = Eigen::Vector3d(1, 0, 0);
    axis.normalize();
    Eigen::Quaterniond q(Eigen::AngleAxisd(angle, axis));
    return g2o::SE3Quat(q, Eigen::Vector3d(t(rng), t(rng), t(rng)));
}

static Eigen::Vector3d projectPoint(const g2o::SE3Quat& Tcw, const Eigen::Vector3d& Xw,
                                    double fx, double fy, double cx, double cy)
{
    Eigen::Vector3d Xc = Tcw.map(Xw);
    return Eigen::Vector3d(fx * Xc(0) / Xc(2) + cx,
                           fy * Xc(1) / Xc(2) + cy,
                           Xc(2));
}

struct JacReport
{
    std::string name;
    double maxAbs = 0.0;
    double maxRel = 0.0;
    int dim = 0;
};

static void accumulate(JacReport& rep, const Eigen::MatrixXd& analytic, const Eigen::MatrixXd& numeric)
{
    rep.dim = (int)analytic.size();
    for (int r = 0; r < analytic.rows(); ++r)
        for (int c = 0; c < analytic.cols(); ++c)
        {
            double a = analytic(r, c);
            double n = numeric(r, c);
            double absErr = std::abs(a - n);
            double relErr = absErr / std::max(1e-8, std::abs(n));
            rep.maxAbs = std::max(rep.maxAbs, absErr);
            rep.maxRel = std::max(rep.maxRel, relErr);
        }
}

// Finite-difference jacobian of an arbitrary error function f: R^d -> R^m,
// using the vertex's own oplusImpl for each perturbation (so the perturbation
// lives in the correct Lie-algebra / orthonormal tangent space).
template <typename Vertex, typename ErrFunc>
static Eigen::MatrixXd finiteDifference(Vertex& vertex, int dof, int m, ErrFunc&& errFunc, double eps)
{
    Eigen::VectorXd e0 = errFunc();
    Eigen::MatrixXd J(m, dof);
    for (int d = 0; d < dof; ++d)
    {
        Eigen::VectorXd deltaP = Eigen::VectorXd::Zero(dof);
        deltaP(d) = eps;
        Eigen::VectorXd deltaM = Eigen::VectorXd::Zero(dof);
        deltaM(d) = -eps;

        auto saved = vertex.estimate();

        vertex.oplusImpl(deltaP.data());
        Eigen::VectorXd eP = errFunc();
        vertex.setEstimate(saved);

        vertex.oplusImpl(deltaM.data());
        Eigen::VectorXd eM = errFunc();
        vertex.setEstimate(saved);

        J.col(d) = (eP - eM) / (2.0 * eps);
    }
    return J;
}

// Independent geometric ground-truth checks for the Plucker convention and the
// Plucker <-> endpoint SE3 transform equivalence. Uses its own line and its own
// projection, not the edge's implementation, to produce the expected values.
static bool geometricConsistency()
{
    const double fx = 525.0, fy = 525.0, cx = 320.0, cy = 240.0;

    struct PoseCase { const char* name; g2o::SE3Quat T; };
    const PoseCase cases[3] = {
        {"pure translation",
         g2o::SE3Quat(Eigen::Quaterniond::Identity(), Eigen::Vector3d(0.1, -0.05, 0.2))},
        {"pure rotation",
         g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(0.35, Eigen::Vector3d(0.3, -0.2, 0.9).normalized())),
                      Eigen::Vector3d::Zero())},
        {"general pose",
         g2o::SE3Quat(Eigen::Quaterniond(Eigen::AngleAxisd(0.25, Eigen::Vector3d(0.5, 0.6, -0.3).normalized())),
                      Eigen::Vector3d(0.12, 0.07, -0.11))},
    };

    bool ok = true;
    for (const auto& c : cases)
    {
        // Ground-truth 3D line in front of the camera.
        const Eigen::Vector3d X1(-0.4, 0.3, 3.0), X2(0.5, -0.2, 3.5);
        const Eigen::Vector3d dir = (X2 - X1).normalized();
        const Eigen::Vector3d n = X1.cross(dir);   // moment
        const Eigen::Vector3d v = dir;             // direction (unit)

        // 1. endpoints -> Plucker: X x v == n for every point X on the line.
        const bool endpointsToPlucker =
            (X1.cross(v) - n).norm() < 1e-9 && (X2.cross(v) - n).norm() < 1e-9;

        // 2. Plucker transform == endpoint SE3 transform.
        const Eigen::Matrix3d R = c.T.rotation().toRotationMatrix();
        const Eigen::Vector3d t = c.T.translation();
        const Eigen::Vector3d nT = R * n + t.cross(R * v);   // transformed moment
        const Eigen::Vector3d vT = R * v;                    // transformed direction
        const Eigen::Vector3d X1T = R * X1 + t, X2T = R * X2 + t;
        const Eigen::Vector3d dirT = (X2T - X1T).normalized();
        const Eigen::Vector3d nFromPts = X1T.cross(dirT);
        const bool transformEquiv =
            (nT - nFromPts).norm() < 1e-9 && (vT - dirT).norm() < 1e-9;

        // 3. transformed endpoints still on the transformed line.
        const bool onLineAfterTransform =
            (X1T.cross(vT) - nT).norm() < 1e-9 && (X2T.cross(vT) - nT).norm() < 1e-9;

        // 4. the projected line passes through the true projected endpoints
        //    (independent ground-truth projection -> edge residual must be ~0).
        g2o::VertexSE3Expmap vPose; vPose.setEstimate(c.T);
        VertexLine4D vLine;
        Eigen::Matrix<double,6,1> Lw; Lw << n, v;
        vLine.setEstimate(Lw);
        EdgeSE3ProjectLine4D edge;
        edge.setVertex(0, &vPose);
        edge.setVertex(1, &vLine);
        edge.SetCameraIntrinsics(fx, fy, cx, cy);
        const Eigen::Vector3d u1 = projectPoint(c.T, X1, fx, fy, cx, cy);
        const Eigen::Vector3d u2 = projectPoint(c.T, X2, fx, fy, cx, cy);
        edge.setMeasurement(std::make_pair(Eigen::Vector2d(u1(0), u1(1)),
                                           Eigen::Vector2d(u2(0), u2(1))));
        edge.computeError();
        const bool projThroughEndpoints = edge.error().norm() < 1e-9;

        const bool pass = endpointsToPlucker && transformEquiv &&
                          onLineAfterTransform && projThroughEndpoints;
        std::cout << std::left << std::setw(20) << c.name
                  << " | endpoints->plucker=" << (endpointsToPlucker ? "PASS" : "FAIL")
                  << " | transform-equiv=" << (transformEquiv ? "PASS" : "FAIL")
                  << " | on-line-after=" << (onLineAfterTransform ? "PASS" : "FAIL")
                  << " | proj-through-endpoints=" << (projThroughEndpoints ? "PASS" : "FAIL")
                  << std::endl;
        ok &= pass;
    }
    return ok;
}

int main()
{
    const double eps = 1e-5;
    const int trials = 200;
    const double fx = 525.0, fy = 525.0, cx = 320.0, cy = 240.0;
    const double bf = 40.0; // baseline * fx

    JacReport repMono{"EdgeSE3ProjectLineXYZOnlyPose_PointToLine"};
    JacReport repStereo{"EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine"};
    JacReport repPlucker{"EdgeSE3ProjectLine4D (pose jacobian)"};
    JacReport repPluckerLine{"EdgeSE3ProjectLine4D (line jacobian)"};

    for (int it = 0; it < trials; ++it)
    {
        g2o::SE3Quat pose = randomPose();
        g2o::VertexSE3Expmap vPose;
        vPose.setEstimate(pose);

        // Random 3D line in front of the camera (valid depth).
        Eigen::Vector3d X1 = randVec3(-1.0, 1.0);
        Eigen::Vector3d X2 = randVec3(-1.0, 1.0);
        X1(2) = 2.0 + std::abs(X1(2));
        X2(2) = 2.0 + std::abs(X2(2));
        if ((X2 - X1).norm() < 0.1) X2 += Eigen::Vector3d(0.2, 0.0, 0.0);

        // -----------------------------------------------------------------
        // 1. Monocular pose-only point-to-line edge (2D residual)
        // -----------------------------------------------------------------
        {
            EdgeSE3ProjectLineXYZOnlyPose_PointToLine edge;
            edge.setVertex(0, &vPose);
            edge.SetXw(X1, X2);
            edge.SetCameraIntrinsics(fx, fy, cx, cy);

            Eigen::Vector3d u1 = projectPoint(pose, X1, fx, fy, cx, cy);
            Eigen::Vector3d u2 = projectPoint(pose, X2, fx, fy, cx, cy);
            edge.SetObservedLineByEndpoints(u1(0), u1(1), u2(0), u2(1));

            g2o::JacobianWorkspace ws;
            ws.updateSize(&edge);
            ws.allocate();
            edge.g2o::BaseUnaryEdge<2, Eigen::Matrix<double,2,1>, g2o::VertexSE3Expmap>::linearizeOplus(ws);
            Eigen::MatrixXd analytic = edge.jacobianOplusXi().cast<double>();

            auto errFunc = [&]() {
                edge.computeError();
                return edge.error().cast<double>();
            };
            Eigen::MatrixXd numeric = finiteDifference(vPose, 6, 2, errFunc, eps);
            accumulate(repMono, analytic, numeric);
        }

        // -----------------------------------------------------------------
        // 2. Stereo pose-only point-to-line edge (4D residual)
        // -----------------------------------------------------------------
        {
            cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
            EdgeStereoSE3ProjectLineXYZOnlyPose_PointToLine edge(X1, X2, K, bf);
            edge.setVertex(0, &vPose);

            Eigen::Vector3d uL1 = projectPoint(pose, X1, fx, fy, cx, cy);
            Eigen::Vector3d uL2 = projectPoint(pose, X2, fx, fy, cx, cy);

            // Rectified horizontal stereo: uR = uL - bf/z, vR = vL.
            double z1 = pose.map(X1)(2);
            double z2 = pose.map(X2)(2);
            Eigen::Vector2d uR1(uL1(0) - bf / z1, uL1(1));
            Eigen::Vector2d uR2(uL2(0) - bf / z2, uL2(1));

            auto lineEq = [](double u1, double v1, double u2, double v2) -> Eigen::Vector3d {
                double dx = u2 - u1, dy = v2 - v1;
                double na = dy, nb = -dx, norm = std::sqrt(na * na + nb * nb);
                if (norm < 1e-12) return Eigen::Vector3d::Zero();
                double a = na / norm, b = nb / norm;
                return Eigen::Vector3d(a, b, -(a * u1 + b * v1));
            };
            edge.SetObservedLines(lineEq(uL1(0), uL1(1), uL2(0), uL2(1)),
                                  lineEq(uR1(0), uR1(1), uR2(0), uR2(1)));

            g2o::JacobianWorkspace ws;
            ws.updateSize(&edge);
            ws.allocate();
            edge.g2o::BaseUnaryEdge<4, Eigen::Matrix<double,4,1>, g2o::VertexSE3Expmap>::linearizeOplus(ws);
            Eigen::MatrixXd analytic = edge.jacobianOplusXi().cast<double>();

            auto errFunc = [&]() {
                edge.computeError();
                return edge.error().cast<double>();
            };
            Eigen::MatrixXd numeric = finiteDifference(vPose, 6, 4, errFunc, eps);
            accumulate(repStereo, analytic, numeric);
        }

        // -----------------------------------------------------------------
        // 3. Plucker line edge used in Local BA (2D residual, pose + line)
        // -----------------------------------------------------------------
        {
            EdgeSE3ProjectLine4D edge;
            edge.setVertex(0, &vPose);

            // Plucker convention (matches Converter::LineSegmentToPlucker):
            //   n = moment = X1 x dir,  v = direction (unit).
            Eigen::Vector3d dir = (X2 - X1).normalized();
            Eigen::Vector3d n = X1.cross(dir);
            Eigen::Vector3d v = dir;
            Eigen::Matrix<double, 6, 1> Lw;
            Lw << n, v;
            VertexLine4D vLine;
            vLine.setEstimate(Lw);
            edge.setVertex(1, &vLine);

            Eigen::Vector3d u1 = projectPoint(pose, X1, fx, fy, cx, cy);
            Eigen::Vector3d u2 = projectPoint(pose, X2, fx, fy, cx, cy);
            edge.setMeasurement(std::make_pair(Eigen::Vector2d(u1(0), u1(1)),
                                               Eigen::Vector2d(u2(0), u2(1))));
            edge.SetCameraIntrinsics(fx, fy, cx, cy);

            g2o::JacobianWorkspace ws;
            ws.updateSize(&edge);
            ws.allocate();
            edge.g2o::BaseBinaryEdge<2, std::pair<Eigen::Vector2d, Eigen::Vector2d>, g2o::VertexSE3Expmap, VertexLine4D>::linearizeOplus(ws);
            Eigen::MatrixXd analyticPose = edge.jacobianOplusXi().cast<double>();
            Eigen::MatrixXd analyticLine = edge.jacobianOplusXj().cast<double>();

            auto errFunc = [&]() {
                edge.computeError();
                return edge.error().cast<double>();
            };
            Eigen::MatrixXd numericPose = finiteDifference(vPose, 6, 2, errFunc, eps);
            Eigen::MatrixXd numericLine = finiteDifference(vLine, 4, 2, errFunc, eps);
            accumulate(repPlucker, analyticPose, numericPose);
            accumulate(repPluckerLine, analyticLine, numericLine);
        }
    }

    const double passAbs = 1e-4;
    const double passRel = 1e-5;

    auto print = [&](const JacReport& r, double thisEps, double thisPassAbs, double thisPassRel) {
        bool pass = (r.maxAbs < thisPassAbs) && (r.maxRel < thisPassRel);
        std::cout << std::left << std::setw(52) << r.name
                  << " | trials=" << std::setw(4) << trials
                  << " | eps=" << thisEps
                  << " | maxAbs=" << std::scientific << std::setprecision(3) << r.maxAbs
                  << " | maxRel=" << std::scientific << std::setprecision(3) << r.maxRel
                  << " | " << (pass ? "PASS" : "FAIL")
                  << " (thresholds: abs<" << thisPassAbs << ", rel<" << thisPassRel << ")" << std::endl;
        return pass;
    };

    std::cout << "===== Stereo line edge analytic-Jacobian vs central finite difference =====" << std::endl;
    bool ok = true;
    ok &= print(repMono, eps, passAbs, passRel);
    ok &= print(repStereo, eps, passAbs, passRel);
    ok &= print(repPlucker, eps, passAbs, passRel);
    ok &= print(repPluckerLine, eps, passAbs, passRel);
    std::cout << "=============================================================================" << std::endl;

    std::cout << "===== Plucker geometric consistency (independent ground truth) =====" << std::endl;
    ok &= geometricConsistency();
    std::cout << "=============================================================================" << std::endl;
    return ok ? 0 : 1;
}
