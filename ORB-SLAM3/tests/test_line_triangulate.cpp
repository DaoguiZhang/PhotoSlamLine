/**
* CPU test for GeometricTools::Triangulate — the REAL shipped implementation.
*
* This test links the actual shared library and calls
* ORB_SLAM3::GeometricTools::Triangulate(). It does not re-derive the DLT.
*
* Build (see tests/run_cpu_tests.sh):
*   g++ -std=c++17 -O2 -I include -I include/CameraModels \
*       -I Thirdparty/Sophus -I /usr/include/eigen3 \
*       -isystem /usr/local/include/opencv4 \
*       tests/test_line_triangulate.cpp -o /tmp/test_line_triangulate \
*       -L lib -lORB_SLAM3 -Wl,-rpath,<abs lib dir>
*
* INTERFACE FACT UNDER TEST
* -------------------------
* Triangulate() reads ONLY x_c(0) and x_c(1) and builds
*     row = x_c(0) * Tc.row(2) - Tc.row(0)
* i.e. the DLT constraint for u = X/Z, v = Y/Z. The third component is never
* read. Therefore the contract is NORMALIZED PLANE coordinates (u, v, *):
*   - a unit ray (u,v,1)/|(u,v,1)| is WRONG  (only the first term is scaled)
*   - an arbitrary homogeneous scale s*(u,v,1) is WRONG for s != 1
* The mono line fallback used to pass a unit ray; the point path never did.
*/

#include "GeometricTools.h"

#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

using ORB_SLAM3::GeometricTools;

namespace
{

int g_checks = 0;
int g_fails  = 0;

void Check(bool cond, const std::string& what)
{
    ++g_checks;
    if (!cond) { ++g_fails; std::printf("  [FAIL] %s\n", what.c_str()); }
}

Eigen::Matrix3f RotAxisAngle(float ax, float ay, float az, float deg)
{
    Eigen::Vector3f axis(ax, ay, az);
    axis.normalize();
    const float a = deg * float(M_PI) / 180.0f;
    return Eigen::AngleAxisf(a, axis).toRotationMatrix();
}

// camera pose: x_c = Rcw * Xw + tcw
struct Pose
{
    Eigen::Matrix3f R;
    Eigen::Vector3f t;
    Eigen::Matrix<float,3,4> mat() const
    {
        Eigen::Matrix<float,3,4> m;
        m.block<3,3>(0,0) = R;
        m.block<3,1>(0,3) = t;
        return m;
    }
    Eigen::Vector3f toCam(const Eigen::Vector3f& Xw) const { return R * Xw + t; }
};

} // namespace

int main()
{
    // ---------------------------------------------------------------------
    // HOW TO READ THIS TEST
    //
    // Check() passing always means "the OBSERVED behaviour is what the label
    // says" -- it never means "the function is safe".
    //
    // Some checks are labelled RECORDED-ABSENCE. Those pass precisely because
    // GeometricTools::Triangulate() does NOT reject the input: no depth guard,
    // no degeneracy guard, no finiteness guard. A green run therefore documents
    // the ABSENCE of protection inside Triangulate and must never be quoted as
    // evidence that Triangulate has any.
    //
    // The only protective statements in this file are about the CALLER
    // (CheckLineReprojection rejecting non-finite / z<=0 endpoints) and about
    // the numerical contract (which scaling is valid).
    // ---------------------------------------------------------------------
    std::printf("== 1. known 3D point, two views, real Triangulate ==\n");
    const Eigen::Vector3f XwTrue(0.42f, -0.31f, 3.7f);
    {
        Pose A, B;
        A.R = Eigen::Matrix3f::Identity();  A.t = Eigen::Vector3f::Zero();
        B.R = Eigen::Matrix3f::Identity();  B.t = Eigen::Vector3f(0.f, 0.f, -0.8f);

        Eigen::Vector3f xc1 = A.toCam(XwTrue);
        Eigen::Vector3f xc2 = B.toCam(XwTrue);
        Eigen::Vector3f uv1(xc1(0)/xc1(2), xc1(1)/xc1(2), 1.f);
        Eigen::Vector3f uv2(xc2(0)/xc2(2), xc2(1)/xc2(2), 1.f);

        Eigen::Matrix<float,3,4> T1 = A.mat(), T2 = B.mat();
        Eigen::Vector3f out;
        bool ok = GeometricTools::Triangulate(uv1, uv2, T1, T2, out);
        Check(ok, "returns true");
        Check((out - XwTrue).norm() < 1e-3f, "recovers the true world point");
        std::printf("     |err| = %.3e\n", (double)(out - XwTrue).norm());
    }

    std::printf("== 2. nonzero translation AND rotation (both cameras) ==\n");
    {
        Pose A, B;
        A.R = RotAxisAngle(0.2f, 1.f, 0.1f, 12.f);  A.t = Eigen::Vector3f(1.0f, -0.5f, 2.0f);
        B.R = RotAxisAngle(-0.3f, 1.f, 0.4f, -25.f); B.t = Eigen::Vector3f(1.7f, -0.2f, 1.4f);

        Eigen::Vector3f xc1 = A.toCam(XwTrue);
        Eigen::Vector3f xc2 = B.toCam(XwTrue);
        Eigen::Vector3f uv1(xc1(0)/xc1(2), xc1(1)/xc1(2), 1.f);
        Eigen::Vector3f uv2(xc2(0)/xc2(2), xc2(1)/xc2(2), 1.f);

        Eigen::Matrix<float,3,4> T1 = A.mat(), T2 = B.mat();
        Eigen::Vector3f out;
        bool ok = GeometricTools::Triangulate(uv1, uv2, T1, T2, out);
        Check(ok, "returns true");
        Check((out - XwTrue).norm() < 1e-2f, "recovers the true point under rotation");
        std::printf("     |err| = %.3e\n", (double)(out - XwTrue).norm());
    }

    std::printf("== 3. same ray at different positive scales ==\n");
    {
        Pose A, B;
        A.R = Eigen::Matrix3f::Identity(); A.t = Eigen::Vector3f::Zero();
        B.R = Eigen::Matrix3f::Identity(); B.t = Eigen::Vector3f(0.f, 0.f, -0.8f);

        Eigen::Vector3f xc1 = A.toCam(XwTrue);
        Eigen::Vector3f xc2 = B.toCam(XwTrue);
        Eigen::Vector3f uv1(xc1(0)/xc1(2), xc1(1)/xc1(2), 1.f);
        Eigen::Vector3f uv2(xc2(0)/xc2(2), xc2(1)/xc2(2), 1.f);

        const float L = uv1.norm();
        const float scales[4] = {1.0f, 1.0f/L, 0.5f, 3.0f};
        const char* names[4] = {"x1 (plane coords, CORRECT)", "x(1/L) unit ray (WRONG)", "x0.5 (WRONG)", "x3 (WRONG)"};
        float errs[4];
        for (int k = 0; k < 4; ++k)
        {
            Eigen::Vector3f a = uv1 * scales[k];
            Eigen::Vector3f b = uv2 * scales[k];
            Eigen::Matrix<float,3,4> T1 = A.mat(), T2 = B.mat();
            Eigen::Vector3f out;
            bool ok = GeometricTools::Triangulate(a, b, T1, T2, out);
            errs[k] = ok ? (out - XwTrue).norm() : std::numeric_limits<float>::quiet_NaN();
            std::printf("     scale %-26s err = %.3e\n", names[k], (double)errs[k]);
        }
        Check(errs[0] < 1e-3f, "scale 1 reproduces the point");
        Check(errs[1] > 1e-3f, "unit ray is NOT equivalent (documented contract)");
        Check(errs[2] > 1e-3f && errs[3] > 1e-3f,
              "arbitrary homogeneous scale is NOT equivalent");
        std::printf("     => the function is NOT scale invariant; only (u,v,*) is valid\n");
    }

    std::printf("== 4. non-positive depth (endpoint behind a camera) ==\n");
    {
        Pose A, B;
        A.R = Eigen::Matrix3f::Identity(); A.t = Eigen::Vector3f::Zero();
        B.R = Eigen::Matrix3f::Identity(); B.t = Eigen::Vector3f(0.f, 0.f, -0.8f);

        // Put the point BEHIND camera A.
        const Eigen::Vector3f XwBehind(0.3f, -0.2f, -2.0f);
        Eigen::Vector3f xc1 = A.toCam(XwBehind);
        Eigen::Vector3f xc2 = B.toCam(XwBehind);
        Eigen::Vector3f uv1(xc1(0)/xc1(2), xc1(1)/xc1(2), 1.f);
        Eigen::Vector3f uv2(xc2(0)/xc2(2), xc2(1)/xc2(2), 1.f);
        Eigen::Matrix<float,3,4> T1 = A.mat(), T2 = B.mat();
        Eigen::Vector3f out;
        bool ok = GeometricTools::Triangulate(uv1, uv2, T1, T2, out);
        // The function has NO depth guard: it happily returns a point with z<0.
        std::printf("     ok=%d recovered=(%.3f, %.3f, %.3f)\n",
                    (int)ok, out(0), out(1), out(2));
        Check(ok, "RECORDED-ABSENCE: no depth guard inside Triangulate");
        Check(out(2) < 0.f, "recovered point really is behind the camera");
        std::printf("     => depth validity is the CALLER's responsibility"
                    " (CreateNewMapLines checks z1/z2 afterwards)\n");
    }

    std::printf("== 5. invalid coordinates (NaN / Inf) ==\n");
    {
        Pose A, B;
        A.R = Eigen::Matrix3f::Identity(); A.t = Eigen::Vector3f::Zero();
        B.R = Eigen::Matrix3f::Identity(); B.t = Eigen::Vector3f(0.f, 0.f, -0.8f);
        Eigen::Matrix<float,3,4> T1 = A.mat(), T2 = B.mat();

        Eigen::Vector3f xc1 = A.toCam(XwTrue);
        Eigen::Vector3f xc2 = B.toCam(XwTrue);
        Eigen::Vector3f good1(xc1(0)/xc1(2), xc1(1)/xc1(2), 1.f);
        Eigen::Vector3f good2(xc2(0)/xc2(2), xc2(1)/xc2(2), 1.f);

        // (a) NaN in the first observation
        Eigen::Vector3f bad1 = good1; bad1(0) = std::numeric_limits<float>::quiet_NaN();
        Eigen::Vector3f outA(0.f, 0.f, 0.f);
        bool okA = GeometricTools::Triangulate(bad1, good2, T1, T2, outA);

        // (b) Inf in the second observation
        Eigen::Vector3f bad2 = good2; bad2(1) = std::numeric_limits<float>::infinity();
        Eigen::Vector3f outB(0.f, 0.f, 0.f);
        bool okB = GeometricTools::Triangulate(good1, bad2, T1, T2, outB);

        std::printf("     NaN -> ok=%d out=(%g,%g,%g)\n", (int)okA, outA(0),outA(1),outA(2));
        std::printf("     Inf -> ok=%d out=(%g,%g,%g)\n", (int)okB, outB(0),outB(1),outB(2));

        // The stable, build-independent statement is about CORRECTNESS, not about
        // finiteness: the function has no validity check, so the numeric output
        // for invalid input is an artefact of the SVD on a degenerate matrix and
        // is not reproducible across builds. Only "does not return the true
        // point" can be asserted.
        Check(!(okA && (outA - XwTrue).norm() < 1e-2f),
              "RECORDED-ABSENCE: NaN input is not rejected, only unusable");
        Check(!(okB && (outB - XwTrue).norm() < 1e-2f),
              "RECORDED-ABSENCE: Inf input is not rejected, only unusable");
        std::printf("     => no validity and no finiteness check inside Triangulate;\n"
                    "        the output for invalid input is not reproducible across\n"
                    "        builds. Validity must be enforced by the caller.\n");
    }

    std::printf("== 6. near-degenerate (tiny baseline) ==\n");
    {
        Pose A, B;
        A.R = Eigen::Matrix3f::Identity(); A.t = Eigen::Vector3f::Zero();
        B.R = Eigen::Matrix3f::Identity(); B.t = Eigen::Vector3f(0.f, 0.f, -1e-4f);

        Eigen::Vector3f xc1 = A.toCam(XwTrue);
        Eigen::Vector3f xc2 = B.toCam(XwTrue);
        Eigen::Vector3f uv1(xc1(0)/xc1(2), xc1(1)/xc1(2), 1.f);
        Eigen::Vector3f uv2(xc2(0)/xc2(2), xc2(1)/xc2(2), 1.f);
        Eigen::Matrix<float,3,4> T1 = A.mat(), T2 = B.mat();
        Eigen::Vector3f out;
        bool ok = GeometricTools::Triangulate(uv1, uv2, T1, T2, out);
        const float err = ok ? (out - XwTrue).norm() : -1.f;
        std::printf("     baseline=1e-4 ok=%d err=%.3e\n", (int)ok, (double)err);
        Check(ok, "RECORDED-ABSENCE: no degeneracy guard inside Triangulate");
        std::printf("     => Triangulate returns success for degenerate geometry;"
                    " geometric validity must be judged downstream\n");
    }

    std::printf("\n%d checks, %d failures\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}
