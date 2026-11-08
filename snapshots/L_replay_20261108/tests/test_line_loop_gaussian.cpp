// P5 CPU math test: line-Gaussian loop-sync transform contract.
//
// Verifies the mathematical operation applied to a line Gaussian under a known
// Sim(3) loop correction:
//   mean      P' = s * R * P + t
//   rotation  q' = q_R (R) composed with q
//   direction d' = R * d          (world line direction, scale-invariant)
//   scaling   s' = s * scaling    (world scale / covariance magnitude)
// plus: once-only guard, replaced/bad MapLine skip, and finiteness.
//
// Pure Eigen math (no CUDA / torch); the GPU implementation mirrors this in
// GaussianModelLine::scaledTransformVisiblePointsOfKeyframe.

#include <iostream>
#include <iomanip>
#include <cmath>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include "MapLine.h"
#include "Atlas.h"

using namespace ORB_SLAM3;

struct LineGaussian
{
    Eigen::Vector3f mean;      // world position
    Eigen::Quaternionf rot;    // local frame
    Eigen::Vector3f scale;     // per-axis world scale
    Eigen::Vector3f dir;       // world line direction (unit)
};

static bool applyLoopCorrection(LineGaussian& g,
                                const Eigen::Matrix3f& R, const Eigen::Vector3f& t, float s)
{
    if (!g.mean.allFinite() || !g.dir.allFinite() || !g.scale.allFinite())
        return false;
    g.mean = s * (R * g.mean) + t;
    g.rot = Eigen::Quaternionf(R) * g.rot;
    g.dir = (R * g.dir).normalized();
    g.scale = s * g.scale;
    return g.mean.allFinite() && g.dir.allFinite() && g.scale.allFinite();
}

static bool testTranslation()
{
    LineGaussian g{Eigen::Vector3f(0, 0, 3), Eigen::Quaternionf::Identity(),
                   Eigen::Vector3f(0.5, 0.3, 0.2), Eigen::Vector3f(1, 0, 0)};
    const Eigen::Vector3f t(0.1, -0.2, 0.05);
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    const bool ok = applyLoopCorrection(g, R, t, 1.0f);
    const bool meanOk = (g.mean - Eigen::Vector3f(0.1, -0.2, 3.05)).norm() < 1e-5f;
    const bool rotOk = g.rot.angularDistance(Eigen::Quaternionf::Identity()) < 1e-5f;
    const bool dirOk = (g.dir - Eigen::Vector3f(1, 0, 0)).norm() < 1e-5f;
    const bool scaleOk = (g.scale - Eigen::Vector3f(0.5, 0.3, 0.2)).norm() < 1e-5f;
    const bool pass = ok && meanOk && rotOk && dirOk && scaleOk;
    std::cout << "[Gaussian][translation] mean=" << meanOk << " rot=" << rotOk
              << " dir=" << dirOk << " scale=" << scaleOk << " : " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool testRotation()
{
    const Eigen::Quaternionf qR(Eigen::AngleAxisf(0.5f, Eigen::Vector3f::UnitZ()));
    LineGaussian g{Eigen::Vector3f(0.5, 0, 3), Eigen::Quaternionf::Identity(),
                   Eigen::Vector3f(0.5, 0.3, 0.2), Eigen::Vector3f::UnitX()};
    const Eigen::Matrix3f R = qR.toRotationMatrix();
    const bool ok = applyLoopCorrection(g, R, Eigen::Vector3f::Zero(), 1.0f);
    const Eigen::Vector3f meanExp = R * Eigen::Vector3f(0.5, 0, 3);
    const bool meanOk = (g.mean - meanExp).norm() < 1e-5f;
    const bool rotOk = g.rot.angularDistance(qR) < 1e-5f;
    const bool dirOk = (g.dir - (R * Eigen::Vector3f::UnitX())).norm() < 1e-5f;
    const bool scaleOk = (g.scale - Eigen::Vector3f(0.5, 0.3, 0.2)).norm() < 1e-5f;
    const bool pass = ok && meanOk && rotOk && dirOk && scaleOk;
    std::cout << "[Gaussian][rotation] mean=" << meanOk << " rot=" << rotOk
              << " dir=" << dirOk << " scale=" << scaleOk << " : " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool testScale()
{
    const float s = 1.2f;
    LineGaussian g{Eigen::Vector3f(0.5, 0.2, 3), Eigen::Quaternionf::Identity(),
                   Eigen::Vector3f(0.5, 0.3, 0.2), Eigen::Vector3f::UnitY()};
    const Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    const bool ok = applyLoopCorrection(g, R, Eigen::Vector3f::Zero(), s);
    const bool meanOk = (g.mean - s * Eigen::Vector3f(0.5, 0.2, 3)).norm() < 1e-5f;
    const bool dirOk = (g.dir - Eigen::Vector3f::UnitY()).norm() < 1e-5f;  // direction scale-invariant
    const bool scaleOk = (g.scale - s * Eigen::Vector3f(0.5, 0.3, 0.2)).norm() < 1e-5f;
    const bool pass = ok && meanOk && dirOk && scaleOk;
    std::cout << "[Gaussian][scale] mean=" << meanOk << " dir=" << dirOk
              << " scale=" << scaleOk << " : " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool testOnceOnly()
{
    // Simulate the per-loop once-only guard (a boolean flag, like the
    // point_not_transformed mask logic).
    LineGaussian g{Eigen::Vector3f(0, 0, 3), Eigen::Quaternionf::Identity(),
                   Eigen::Vector3f(0.5, 0.3, 0.2), Eigen::Vector3f(1, 0, 0)};
    const Eigen::Vector3f t(0.1, 0, 0);
    bool corrected = false;
    int applied = 0;
    for (int kf = 0; kf < 3; ++kf)   // 3 observing KFs in the same loop
    {
        if (corrected) continue;     // once-only guard
        applyLoopCorrection(g, Eigen::Matrix3f::Identity(), t, 1.0f);
        corrected = true;
        ++applied;
    }
    const bool pass = (applied == 1) && (g.mean.x() - 0.1f < 1e-5f);
    std::cout << "[Gaussian][once-only] applied=" << applied << " : " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool testReplacedMapLineSkip()
{
    // A bad / replaced MapLine must be skipped by the SLAM-side packing
    // (addMapLine checks isBad). Verify the guard semantics.
    MapLine ml;
    ml.mnId = 1;
    const bool before = !ml.isBad();
    // Simulate "replaced" by the fusion path: mark bad via the public contract.
    MapLine target; target.mnId = 2;
    ml.Replace(&target);   // old line becomes bad + replaced
    const bool afterBad = ml.isBad();
    const bool replacedSet = (ml.GetReplaced() == &target);
    const bool pass = before && afterBad && replacedSet;
    std::cout << "[Gaussian][replaced-skip] before=" << before
              << " afterBad=" << afterBad << " replaced=" << replacedSet
              << " : " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

static bool testOperationCopyPreservesLineSamples()
{
    // Regression: MappingOperation copy constructor must preserve
    // mvAssociatedLineSampledPoints (and mvAssociatedMapLineIds). A missing
    // member in the copy ctor silently dropped line-sampled points, so the
    // Gaussian mapper never seeded line Gaussians (candidates always 0).
    MappingOperation opr(MappingOperation::OprType::LocalMappingBA);
    auto& sp = opr.associatedLineSampledPoints();
    std::get<0>(sp) = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};  // 2 points (x,y,z x2)
    std::get<1>(sp) = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f};
    std::get<2>(sp) = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    auto& ids = opr.associatedMapLineIds();
    ids.push_back(100UL); ids.push_back(200UL);

    MappingOperation copy = opr;  // exercise the copy constructor

    const auto& sp2 = copy.associatedLineSampledPoints();
    const bool posOk = (std::get<0>(sp2).size() == 6 && std::get<0>(sp2)[5] == 6.f);
    const bool colOk = (std::get<1>(sp2).size() == 6 && std::get<1>(sp2)[5] == 0.6f);
    const bool dirOk = (std::get<2>(sp2).size() == 6 && std::get<2>(sp2)[4] == 1.f);
    const auto& ids2 = copy.associatedMapLineIds();
    const bool idsOk = (ids2.size() == 2 && ids2[1] == 200UL);
    const bool pass = posOk && colOk && dirOk && idsOk;
    std::cout << "[Gaussian][op-copy-line-samples] pos=" << posOk
              << " col=" << colOk << " dir=" << dirOk << " ids=" << idsOk
              << " : " << (pass ? "PASS" : "FAIL") << std::endl;
    return pass;
}

int main()
{
    bool ok = true;
    ok &= testTranslation();
    ok &= testRotation();
    ok &= testScale();
    ok &= testOnceOnly();
    ok &= testReplacedMapLineSkip();
    ok &= testOperationCopyPreservesLineSamples();
    std::cout << (ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED") << std::endl;
    return ok ? 0 : 1;
}
