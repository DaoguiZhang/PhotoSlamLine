/**
* CPU test: does PoseOptimizationWithLine reproduce PoseOptimization's POINT
* outlier bookkeeping when there are ZERO line observations?
*
* STATIC FINDING (2026-10-30), source evidence:
*   Optimizer::PoseOptimization()          -> e->computeError() at 3 sites
*                                             (mono / mono_FHR / stereo)
*   Optimizer::PoseOptimizationWithLine()  -> 0 sites
*
* In g2o an edge demoted with setLevel(1) is NOT updated during optimize(), so
* chi2() returns a STALE error. ORB-SLAM3 re-computes it explicitly for exactly
* that reason. Omitting the call can keep an edge classified as an outlier
* after it has become an inlier, perturbing the inlier count that
* TrackReferenceKeyFrame(WithLine) compares against the >=10 threshold.
*
* This test reproduces the mechanism with the REAL edge class and the REAL
* projection: no SLAM, no threads, no Atlas.
*
* Build:
*   g++ -std=c++17 -O2 -w -I . -I include -I include/CameraModels \
*       -I Thirdparty/g2o -I Thirdparty/Sophus -I /usr/include/eigen3 \
*       -isystem /usr/local/include/opencv4 tests/test_pose_opt_outlier_reeval.cpp \
*       -o /tmp/test_pose_opt -L lib -lORB_SLAM3 -L Thirdparty/g2o/lib -lg2o \
*       -Wl,-rpath,<abs lib>:<abs Thirdparty/g2o/lib>
*/

#include "OptimizableTypes.h"
#include "CameraModels/Pinhole.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/linear_solver_dense.h>
#include <g2o/types/types_six_dof_expmap.h>

#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
int g_checks = 0;
int g_fails = 0;
void Check(bool cond, const std::string& what)
{
    ++g_checks;
    if (!cond) { ++g_fails; std::printf("  [FAIL] %s\n", what.c_str()); }
}
} // namespace

int main()
{
    const float fx = 535.4f, fy = 539.2f, cx = 320.1f, cy = 247.6f;
    ORB_SLAM3::Pinhole cam(std::vector<float>{fx, fy, cx, cy});

    g2o::SparseOptimizer optimizer;
    auto* linearSolver = new g2o::LinearSolverDense<g2o::BlockSolver_6_3::PoseMatrixType>();
    auto* solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    auto* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);

    auto* vSE3 = new g2o::VertexSE3Expmap();
    vSE3->setEstimate(g2o::SE3Quat(Eigen::Quaterniond::Identity(), Eigen::Vector3d::Zero()));
    vSE3->setId(0);
    vSE3->setFixed(false);
    optimizer.addVertex(vSE3);

    const int N = 6;
    std::vector<ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose*> edges;
    for (int i = 0; i < N; ++i)
    {
        auto* e = new ORB_SLAM3::EdgeSE3ProjectXYZOnlyPose();
        e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(0)));
        e->setInformation(Eigen::Matrix2d::Identity());
        e->pCamera = &cam;
        e->Xw << 0.30 + 0.05 * i, -0.20 + 0.03 * i, 3.0 + 0.1 * i;
        Eigen::Vector2d z(fx * e->Xw[0] / e->Xw[2] + cx,
                          fy * e->Xw[1] / e->Xw[2] + cy);
        if (i == 0) z += Eigen::Vector2d(120.0, 90.0);   // edge 0 is a bad match
        else        z += Eigen::Vector2d(0.6 * i, -0.4 * i);   // small residual
        e->setMeasurement(z);
        e->setId(i);
        optimizer.addEdge(e);
        edges.push_back(e);
    }

    std::printf("== iteration 1: edge0 carries a large residual ==\n");
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);
    std::printf("     edge0 chi2 = %.6f (level %d)\n", edges[0]->chi2(), edges[0]->level());
    Check(edges[0]->chi2() > 5.991,
          "edge0 is an OUTLIER at iteration 1 (chi2 > chi2Mono)");

    // ORB-SLAM3 demotes it exactly here. From now on the edge is excluded from
    // the optimisation, so nothing refreshes its cached error. Then its
    // measurement is corrected: the observation is now a good one.
    edges[0]->setLevel(1);
    {
        Eigen::Vector2d z(fx * edges[0]->Xw[0] / edges[0]->Xw[2] + cx,
                          fy * edges[0]->Xw[1] / edges[0]->Xw[2] + cy);
        edges[0]->setMeasurement(z);
    }

    std::printf("== iteration 2: identical optimisation in both variants ==\n");
    optimizer.initializeOptimization(0);
    optimizer.optimize(10);

    // variant B (PoseOptimizationWithLine semantics): read chi2 directly
    const double chi2Stale = edges[0]->chi2();
    // variant A (PoseOptimization semantics): refresh first, then read
    edges[0]->computeError();
    const double chi2Fresh = edges[0]->chi2();

    std::printf("     chi2 WITHOUT computeError() : %.6f\n", chi2Stale);
    std::printf("     chi2 WITH    computeError() : %.6f\n", chi2Fresh);
    const double diff = std::fabs(chi2Stale - chi2Fresh);
    std::printf("     |difference|                = %.6f\n", diff);

    Check(std::isfinite(chi2Stale) && std::isfinite(chi2Fresh), "both reads finite");
    Check(diff > 1e-3,
          "MECHANISM: without computeError() the chi2 is stale and differs");
    Check(chi2Stale >= 5.991 && chi2Fresh < 5.991,
          "DECISION DRIFT: stale keeps edge0 OUTLIER, refreshed makes it INLIER");
    std::printf("     => the point optimiser re-admits the edge (inlier count +1);\n"
                "        the line optimiser leaves it out (inlier count unchanged)\n");

    std::printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
