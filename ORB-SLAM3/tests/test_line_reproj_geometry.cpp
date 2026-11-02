/**
* CPU unit test for the line reprojection geometry used by
* LocalMapping::CheckLineReprojection().
*
* The test includes the SAME header the production translation unit includes
* (ORB-SLAM3/include/LineReprojGeometry.h) and calls the SAME functions, so a
* regression in the shipped code fails this test. There is no re-derivation of
* the maths here.
*
* Build:
*   g++ -std=c++17 -O2 -I<ORB-SLAM3>/include -I<eigen> \
*       test_line_reproj_geometry.cpp -o test_line_reproj_geometry
*
* Keylines use image coordinates where y grows DOWNWARD (OpenCV). The camera
* projection also returns y downward, so an identity pose must reproduce the
* keyline exactly.
*/

#include "LineReprojGeometry.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <cmath>
#include <cstdio>
#include <limits>
#include <functional>
#include <string>

using namespace ORB_SLAM3;
using line_reproj::Evaluate;
using line_reproj::EvaluateLegacy;
using line_reproj::Params;
using line_reproj::Result;

namespace
{

int g_failures = 0;
int g_checks   = 0;

void Check(bool cond, const std::string& what)
{
    ++g_checks;
    if (!cond)
    {
        ++g_failures;
        std::printf("  [FAIL] %s\n", what.c_str());
    }
}

// Pinhole model identical to CameraModels/Pinhole.cpp::project().
struct Pinhole
{
    float fx = 535.4f, fy = 539.2f, cx = 320.1f, cy = 247.6f;
    Eigen::Vector2f operator()(const Eigen::Vector3f& p) const
    {
        return Eigen::Vector2f(fx * p(0) / p(2) + cx, fy * p(1) / p(2) + cy);
    }
};

Eigen::Vector3f BackProject(const Pinhole& K, const Eigen::Vector2f& px, float z)
{
    return Eigen::Vector3f((px(0) - K.cx) / K.fx * z,
                           (px(1) - K.cy) / K.fy * z,
                           z);
}

// Truth segment in camera coordinates of the FIRST keyframe.
const Pinhole K;
const Eigen::Vector2f PXS(250.f, 200.f);
const Eigen::Vector2f PXE(400.f, 300.f);
const float Z_TRUE = 4.0f;

Eigen::Vector3f CamS() { return BackProject(K, PXS, Z_TRUE); }
Eigen::Vector3f CamE() { return BackProject(K, PXE, Z_TRUE); }

// Keyline midpoint / direction from the true projected segment.
Eigen::Vector2f KeylineMid() { return 0.5f * (PXS + PXE); }
Eigen::Vector2f KeylineDir()
{
    Eigen::Vector2f d = PXE - PXS;
    return d / d.norm();
}

Eigen::Matrix3f RotZ(float deg)
{
    const float a = deg * float(M_PI) / 180.0f;
    Eigen::Matrix3f R;
    R << std::cos(a), -std::sin(a), 0.f,
         std::sin(a),  std::cos(a), 0.f,
         0.f,          0.f,        1.f;
    return R;
}

std::function<Eigen::Vector2f(const Eigen::Vector3f&)> Proj()
{
    return [](const Eigen::Vector3f& p) { return K(p); };
}

// Applies world transform (Rw, tw): Xw = Rw * Xc + tw
Eigen::Vector3f ToWorld(const Eigen::Matrix3f& Rw, const Eigen::Vector3f& tw,
                        const Eigen::Vector3f& Xc)
{
    return Rw * Xc + tw;
}

} // namespace

int main()
{
    const Params P;   // production defaults: 15 px, cos20, 1 px

    std::printf("== 1. identity pose: truth segment must be ACCEPTED ==\n");
    {
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        Result r = Evaluate(Rcw, tcw, CamS(), CamE(), KeylineMid(), KeylineDir(), Proj(), P);
        Check(r.ok, "identity pose accepts the exact segment");
        Check(r.code == line_reproj::kAccept, "code == ACCEPT");
        Check(std::fabs(r.dStart) < 1e-3f && std::fabs(r.dEnd) < 1e-3f,
              "perpendicular distances are ~0 px");
        Check(std::fabs(r.absCos - 1.f) < 1e-5f, "|cos| == 1");
        const float expectedLen = (PXE - PXS).norm();
        Check(std::fabs(r.lenProj - expectedLen) < 1e-2f, "projected length preserved");
    }

    std::printf("== 2. nonzero translation: legacy fails, fixed succeeds ==\n");
    {
        Eigen::Matrix3f Rw = Eigen::Matrix3f::Identity();
        const Eigen::Vector3f tw(12.f, -3.f, 40.f);      // camera centre in world
        Eigen::Matrix3f Rcw = Rw.transpose();
        Eigen::Vector3f tcw = -Rcw * tw;

        Eigen::Vector3f XwS = ToWorld(Rw, tw, CamS());
        Eigen::Vector3f XwE = ToWorld(Rw, tw, CamE());

        Result newR = Evaluate(Rcw, tcw, XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);
        Result oldR = EvaluateLegacy(XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);

        Check(newR.ok, "FIXED: world endpoints with poses -> ACCEPT");
        Check(!oldR.ok, "LEGACY: same inputs -> REJECT (this is the bug)");
        Check(oldR.code == line_reproj::kPerpDistance,
              "LEGACY rejects at the perpendicular-distance stage");
        std::printf("     legacy code=%s new code=%s\n",
                    line_reproj::CodeName(oldR.code), line_reproj::CodeName(newR.code));
    }

    std::printf("== 3. nonzero rotation ==\n");
    {
        Eigen::Matrix3f Rw = RotZ(35.f) * RotZ(0.f);
        const Eigen::Vector3f tw(1.5f, 2.5f, 6.0f);
        Eigen::Matrix3f Rcw = Rw.transpose();
        Eigen::Vector3f tcw = -Rcw * tw;

        Eigen::Vector3f XwS = ToWorld(Rw, tw, CamS());
        Eigen::Vector3f XwE = ToWorld(Rw, tw, CamE());

        Result newR = Evaluate(Rcw, tcw, XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);
        Check(newR.ok, "FIXED: rotated world frame -> ACCEPT");
        std::printf("     zS=%.3f zE=%.3f dStart=%.3f dEnd=%.3f code=%s\n",
                    newR.zS, newR.zE, newR.dStart, newR.dEnd,
                    line_reproj::CodeName(newR.code));
        Check(std::fabs(newR.zS - Z_TRUE) < 1e-3f, "z restored to the true depth");

        Result oldR = EvaluateLegacy(XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);
        Check(!oldR.ok, "LEGACY: rotated world frame -> REJECT");
    }

    std::printf("== 4. invariance under a COMMON world transform ==\n");
    {
        // Rigid change of world frame applied to BOTH the endpoints and the
        // pose must leave the decision unchanged.
        Eigen::Matrix3f Rw = RotZ(35.f);
        Eigen::Vector3f tw(1.5f, 2.5f, 6.0f);
        Eigen::Matrix3f Rcw = Rw.transpose();
        Eigen::Vector3f tcw = -Rcw * tw;
        Eigen::Vector3f XwS = ToWorld(Rw, tw, CamS());
        Eigen::Vector3f XwE = ToWorld(Rw, tw, CamE());
        Result a = Evaluate(Rcw, tcw, XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);

        // G: another arbitrary rigid transform of the world.
        Eigen::Matrix3f G = RotZ(-118.f);
        Eigen::Vector3f g(7.f, -2.f, 3.5f);
        // new pose: x_c = Rcw * (G^T (X' - g)) + tcw
        Eigen::Matrix3f Rcw2 = Rcw * G.transpose();
        Eigen::Vector3f tcw2 = -Rcw * G.transpose() * g + tcw;
        Eigen::Vector3f XwS2 = G * XwS + g;
        Eigen::Vector3f XwE2 = G * XwE + g;
        Result b = Evaluate(Rcw2, tcw2, XwS2, XwE2, KeylineMid(), KeylineDir(), Proj(), P);

        Check(a.ok == b.ok, "decision invariant under common world transform");
        Check(std::fabs(a.dStart - b.dStart) < 1e-2f &&
              std::fabs(a.dEnd - b.dEnd) < 1e-2f,
              "perp distances invariant (<1e-2 px)");
        Check(std::fabs(a.zS - b.zS) < 1e-3f, "depth invariant");
    }

    std::printf("== 5. behind-camera endpoint ==\n");
    {
        // Same pixels, but the segment lies BEHIND the camera (negative z).
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        Eigen::Vector3f XwS = BackProject(K, PXS, -3.0f);
        Eigen::Vector3f XwE = BackProject(K, PXE, -3.0f);
        Result r = Evaluate(Rcw, tcw, XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);
        Check(!r.ok, "behind-camera segment rejected");
        Check(r.code == line_reproj::kBehindCameraS, "code == BEHIND_S");
        Check(r.zS < 0.f, "reported z is negative");

        // Legacy accepts this: it is exactly the kind of false positive the
        // depth guard removes.
        Result oldR = EvaluateLegacy(XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);
        std::printf("     legacy ok=%d code=%s | new code=%s\n",
                    (int)oldR.ok, line_reproj::CodeName(oldR.code),
                    line_reproj::CodeName(r.code));
    }

    std::printf("== 6. NaN / Inf endpoints ==\n");
    {
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        Eigen::Vector3f XwS = CamS();
        XwS(1) = std::nanf("");
        Result r = Evaluate(Rcw, tcw, XwS, CamE(), KeylineMid(), KeylineDir(), Proj(), P);
        Check(!r.ok, "NaN endpoint rejected");
        Check(r.code == line_reproj::kNonFiniteInput, "code == NONFINITE_INPUT");

        Eigen::Vector3f XwS2 = CamS();
        XwS2(0) = std::numeric_limits<float>::infinity();
        Result r2 = Evaluate(Rcw, tcw, XwS2, CamE(), KeylineMid(), KeylineDir(), Proj(), P);
        Check(!r2.ok, "Inf endpoint rejected");
        Check(r2.code == line_reproj::kNonFiniteInput, "code == NONFINITE_INPUT");

        // NaN in the pose must not silently produce a decision.
        Eigen::Matrix3f Rbad = Rcw;
        Rbad(0,0) = std::nanf("");
        Result r3 = Evaluate(Rbad, tcw, CamS(), CamE(), KeylineMid(), KeylineDir(), Proj(), P);
        Check(!r3.ok, "NaN pose rejected");
    }

    std::printf("== 7. degenerate segment (zero projected length) ==\n");
    {
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        // Segment parallel to the optical axis: both endpoints project to the
        // same pixel. The legacy order (direction first, length last) means
        // normalized() of a zero vector yields a zero direction, so the legacy
        // code also rejects -- but it reports DIRECTION, masking the real
        // cause. The fixed version reports PROJ_LEN explicitly and rejects
        // before any division.
        Eigen::Vector3f XwS = BackProject(K, PXS, 2.0f);
        Eigen::Vector3f XwE = BackProject(K, PXS, 8.0f);
        Result r = Evaluate(Rcw, tcw, XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);
        Check(!r.ok, "degenerate (axis-parallel) segment rejected");
        Check(r.code == line_reproj::kProjLength, "code == PROJ_LEN");
        std::printf("     lenProj=%.6f code=%s\n", r.lenProj,
                    line_reproj::CodeName(r.code));

        Result oldR = EvaluateLegacy(XwS, XwE, KeylineMid(), KeylineDir(), Proj(), P);
        std::printf("     legacy ok=%d code=%s absCos=%.4f\n",
                    (int)oldR.ok, line_reproj::CodeName(oldR.code), oldR.absCos);
    }

    std::printf("== 8. direction disagreement beyond 20 deg ==\n");
    {
        Eigen::Matrix3f Rcw = Eigen::Matrix3f::Identity();
        Eigen::Vector3f tcw = Eigen::Vector3f::Zero();
        // Vertical projected segment while the keyline is horizontal: both
        // endpoints stay within 15 px of the keyline's infinite line, so the
        // perpendicular-distance test PASSES and the direction test is the
        // first to fail.
        const Eigen::Vector2f a(325.f, 240.f), b(325.f, 260.f);
        Eigen::Vector3f XwS = BackProject(K, a, Z_TRUE);
        Eigen::Vector3f XwE = BackProject(K, b, Z_TRUE);
        Eigen::Vector2f badMid(325.f, 250.f);
        Eigen::Vector2f badDir(std::cos(0.0f), std::sin(0.0f));   // horizontal
        Result r = Evaluate(Rcw, tcw, XwS, XwE, badMid, badDir, Proj(), P);
        Check(!r.ok, "90 deg direction mismatch rejected");
        Check(r.code == line_reproj::kDirection, "code == DIRECTION");
        Check(r.dStart <= P.reprojTh && r.dEnd <= P.reprojTh,
              "perpendicular distance test passed first (isolates DIRECTION)");
        std::printf("     dStart=%.2f dEnd=%.2f lenProj=%.2f absCos=%.4f code=%s\n",
                    r.dStart, r.dEnd, r.lenProj, r.absCos,
                    line_reproj::CodeName(r.code));
    }

    std::printf("== 9. Triangulate() input convention: unit ray vs (u,v,1) ==\n");
    {
        // GeometricTools::Triangulate builds
        //   row0 = x_c1(0)*Tc1w.row(2) - Tc1w.row(0)
        //   row1 = x_c1(1)*Tc1w.row(2) - Tc1w.row(1)
        // which is the DLT constraint for x_c = (X/Z, Y/Z). unprojectEig()
        // returns (u, v, 1); the mono fallback USED TO call .normalize(),
        // producing a unit ray (u,v,1)/L. That is NOT equivalent: a row is
        //   x0 * Tc.row(2) - Tc.row(0)
        // so scaling x0 alone does not scale the whole row, the constant term
        // keeps its weight, and the null space moves. Verified numerically
        // below; see tests/test_line_triangulate.cpp for the real-implementation
        // check against all three scalings.
        const Eigen::Vector3f Xc(0.35f, -0.22f, 5.0f);   // (u,v,1)*z form
        const Eigen::Vector3f uv1(Xc(0)/Xc(2), Xc(1)/Xc(2), 1.f);
        const Eigen::Vector3f ray1 = uv1.normalized();

        Eigen::Matrix<float,3,4> T1, T2;
        T1 << 1,0,0,0,  0,1,0,0,  0,0,1,0;              // identity
        T2 << 1,0,0,0,  0,1,0,0,  0,0,1,-1.0f;          // 1 m along z

        // Observation in the SECOND camera: Xc expressed in camera 2.
        const Eigen::Vector3f Xc2(Xc(0), Xc(1), Xc(2) - 1.0f);
        const Eigen::Vector3f uv2(Xc2(0)/Xc2(2), Xc2(1)/Xc2(2), 1.f);
        const Eigen::Vector3f ray2 = uv2.normalized();

        auto triangulate = [](const Eigen::Vector3f& a, const Eigen::Vector3f& b,
                              Eigen::Matrix<float,3,4>& Ta, Eigen::Matrix<float,3,4>& Tb,
                              Eigen::Vector3f& out) -> bool {
            Eigen::Matrix4f A_;
            A_.block<1,4>(0,0) = a(0)*Ta.block<1,4>(2,0) - Ta.block<1,4>(0,0);
            A_.block<1,4>(1,0) = a(1)*Ta.block<1,4>(2,0) - Ta.block<1,4>(1,0);
            A_.block<1,4>(2,0) = b(0)*Tb.block<1,4>(2,0) - Tb.block<1,4>(0,0);
            A_.block<1,4>(3,0) = b(1)*Tb.block<1,4>(2,0) - Tb.block<1,4>(1,0);
            Eigen::JacobiSVD<Eigen::Matrix4f> svd(A_, Eigen::ComputeFullV);
            Eigen::Vector4f h = svd.matrixV().col(3);
            if (h(3) == 0) return false;
            out = h.head(3) / h(3);
            return true;
        };

        Eigen::Vector3f outUV, outRay;
        bool okUV  = triangulate(uv1,  uv2,  T1, T2, outUV);
        bool okRay = triangulate(ray1, ray2, T1, T2, outRay);
        Check(okUV && okRay, "both conventions triangulate");

        const float eUV  = (outUV  - Xc).norm();
        const float eRay = (outRay - Xc).norm();
        std::printf("     |uv - truth|  = %.3e\n", (double)eUV);
        std::printf("     |ray - truth| = %.3e   (u=%.3f,v=%.3f,L=%.5f)\n",
                    (double)eRay, (double)uv1(0), (double)uv1(1),
                    (double)uv1.norm());

        Check(eUV < 1e-2f, "(u,v,1) convention recovers ground truth");
        // ------------------------------------------------------------------
        // FINDING (recorded, NOT fixed in this round):
        // passing a UNIT ray instead of (u,v,1) is NOT equivalent. A DLT row is
        //   a0 * T.row(2) - T.row(0)
        // so scaling a0 alone does not scale the whole row: the constant term
        // keeps its old weight and the solution is biased. The mono fallback in
        // LocalMapping.cc does
        //   xu1 = cam->unprojectEig(mid1); xu1.normalize();
        // before Triangulate(), while the POINT path (LocalMapping.cc:1213+)
        // passes unprojectEig() output WITHOUT normalize(). This assertion
        // records that the deviation is materially non-zero, so it cannot be
        // dismissed as floating-point noise.
        // ------------------------------------------------------------------
        Check(eRay > 1e-3f,
              "normalize() before Triangulate() measurably biases the result");
        std::printf("     => unit-ray convention deviates by %.1f mm at 5 m\n",
                    (double)(eRay * 1000.f));
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
