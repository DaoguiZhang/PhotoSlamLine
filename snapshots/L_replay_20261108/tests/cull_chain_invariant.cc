/**
* Minimal constructed cases for the culled-keyframe chain invariant
* (Photo-SLAM-L). Companion to KeyFrame::SetBadFlag / CarryCulledDependents and to
* the exporter's reconstruction in System::SaveTrajectoryTUM.
*
* MODEL (what the code does)
*   * A keyframe holds a pose P = Tcw (world -> camera).
*   * KeyFrame::SetBadFlag freezes  mTcp = P_child * P_parent^-1  - exactly the
*     relative pose, evaluated in the metric of that instant.
*   * The exporter rebuilds a culled keyframe's pose as
*         TrwKF = mTcp_child * mTcp_parent * ... * P_live
*     walking the mTcp links upward until it reaches a live keyframe.
*   * Every pose-replacement site in the tree assigns a new pose PER KEYFRAME
*     (local BA `Tiw`, pose graph `Tiw`, GBA `mTcwGBA`, merge `mTcwMerge`, loop
*     correction, Map::ApplyScaledRotation, initialization), so the transform
*     that applies to keyframe k is  D_k = P'_k * P_k^-1.  There is no site that
*     multiplies every keyframe by one common rigid factor on the right;
*     Map::ApplyScaledRotation rescales the TRANSLATION of each keyframe's
*     INVERSE pose, which is not such a factor.
*
* INVARIANT
*   A culled child must land where it would have landed had it stayed LIVE and
*   received the same per-keyframe rule:   chain(child) = D_parent * P_child.
*   The carry implements it as  mTcp <- D * mTcp * D^-1.
*
* Cases 2..8 have an exact reference (the sibling receives the same D), so the
* carried chain must be ~0 while the "no carry" column shows the defect size.
* Case 1 (ordinary BA) has no exact reference - the sibling is refined on its own
* path - so its third column is |carry - no carry|, the size of the choice itself.
* Case 3's third column is the negative control: conjugating by the Sim3 (the
* map-point rule) instead of the pose-level D.
*
* Build (no CMake change needed):
*   g++ -O2 -std=c++17 -I/usr/include/eigen3 -IORB-SLAM3/Thirdparty/Sophus \
*       tests/cull_chain_invariant.cc -o /tmp/cull_chain_invariant
*/

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

using Sophus::SE3f;
using Eigen::Vector3f;
using Eigen::Quaternionf;

static double PosErr(const SE3f &a, const SE3f &b)
{
    return (a.translation() - b.translation()).norm();
}

// Exactly KeyFrame::ConjugateCulledTcp.
static SE3f Carry(const SE3f &mTcp, const SE3f &D)
{
    return D * mTcp * D.inverse();
}

// The transform a merge leaves on a KEYFRAME: the merge writes
// correctedTiw(rotation, translation / s), i.e. the SE(3) part of the Sim3.
static SE3f PoseRule(const Quaternionf &R_s, const Vector3f &t_s, double s)
{
    return SE3f(R_s.cast<double>().normalized().cast<float>(),
                (t_s / static_cast<float>(s)).cast<float>());
}

static Quaternionf RandAxis(std::mt19937 &rng, float deg, const Vector3f &ax)
{
    return Quaternionf(Eigen::AngleAxisf(deg * static_cast<float>(M_PI) / 180.0f, ax.normalized()));
}

struct Result
{
    const char *name;
    double carry_err;
    double nocarry_err;
    double control_err;
    bool bControl;
    Result(const char *n) : name(n), carry_err(0), nocarry_err(0), control_err(0), bControl(false) {}
};

int main()
{
    std::mt19937 rng(20261015);
    std::vector<Result> vRes;
    const Vector3f axGen(0.3f, 0.6f, 0.7416f);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);

    // 1. ordinary BA: parent and a would-be-live sibling are each refined by their
    //    OWN small update; the culled child cannot be refined at all.
    {
        Result r("1 BA x20 (0.5deg / 25mm steps), no exact reference");
        const int nTrial = 200, nStep = 20;
        for(int t = 0; t < nTrial; ++t)
        {
            SE3f Pp(PoseRule(Quaternionf::Identity(), Vector3f(1.0f, 0.2f, -0.4f), 1.0));
            SE3f Pc(PoseRule(Quaternionf::Identity(), Vector3f(1.8f, 0.1f, -0.2f), 1.0));
            const SE3f mTcpFrozen = Pc * Pp.inverse();
            SE3f mTcpCarried = mTcpFrozen;
            SE3f PcLive = Pc;
            for(int s = 0; s < nStep; ++s)
            {
                const SE3f Dp(RandAxis(rng, 0.5f * u(rng), Vector3f(u(rng), u(rng), u(rng) + 0.01f)),
                              0.025f * Vector3f(u(rng), u(rng), u(rng)));
                mTcpCarried = Carry(mTcpCarried, Dp);
                Pp = Dp * Pp;
                const SE3f Dc(RandAxis(rng, 0.5f * u(rng), Vector3f(u(rng), u(rng), u(rng) + 0.01f)),
                              0.025f * Vector3f(u(rng), u(rng), u(rng)));
                PcLive = Dc * PcLive;
            }
            r.carry_err += PosErr(mTcpCarried * Pp, PcLive);
            r.nocarry_err += PosErr(mTcpFrozen * Pp, PcLive);
            r.control_err += PosErr(mTcpCarried * Pp, mTcpFrozen * Pp);
            r.bControl = true;
        }
        r.carry_err /= nTrial; r.nocarry_err /= nTrial; r.control_err /= nTrial;
        vRes.push_back(r);
    }

    auto SingleStep = [&](const char *name, const Quaternionf &R_s, const Vector3f &t_s, double s,
                          bool bSim3Control) {
        Result r(name);
        const int nTrial = 200;
        for(int t = 0; t < nTrial; ++t)
        {
            SE3f Pp(PoseRule(Quaternionf::Identity(), Vector3f(1.0f, 0.2f, -0.4f), 1.0));
            SE3f Pc(PoseRule(RandAxis(rng, 30.0f, axGen), Vector3f(1.8f, 0.1f, -0.2f), 1.0));
            const SE3f mTcpFrozen = Pc * Pp.inverse();
            const SE3f D = PoseRule(R_s, t_s, s);
            const SE3f mTcpCarried = Carry(mTcpFrozen, D);
            const SE3f PpNew = D * Pp;
            const SE3f PcLiveNew = D * Pc;    // exact reference: same D for every moved KF
            r.carry_err += PosErr(mTcpCarried * PpNew, PcLiveNew);
            r.nocarry_err += PosErr(mTcpFrozen * PpNew, PcLiveNew);
            if(bSim3Control)
            {
                const SE3f S(R_s.cast<double>().normalized().cast<float>(), t_s);
                r.control_err += PosErr(Carry(mTcpFrozen, S) * PpNew, PcLiveNew);
                r.bControl = true;
            }
        }
        r.carry_err /= nTrial; r.nocarry_err /= nTrial;
        if(bSim3Control) r.control_err /= nTrial;
        vRes.push_back(r);
    };

    SingleStep("2 rigid merge: R=40deg |t|=1.8m s=1", RandAxis(rng, 40.0f, Vector3f(0.2f, -0.5f, 0.84f)),
               Vector3f(1.6f, -0.6f, 0.5f), 1.0, false);
    SingleStep("3 Sim3 merge: R=75deg |t|=1.5m s=0.233926 (measured case)",
               RandAxis(rng, 75.0f, axGen), Vector3f(0.4f, -1.2f, 0.7f), 0.233926, true);
    SingleStep("4 pure scale: R=I t=0 s=0.2 (pose rule = identity)",
               Quaternionf::Identity(), Vector3f(0, 0, 0), 0.2, false);
    SingleStep("5 rotation only: R=40deg t=0 s=1", RandAxis(rng, 40.0f, Vector3f(0.2f, -0.5f, 0.84f)),
               Vector3f(0, 0, 0), 1.0, false);
    SingleStep("6 translation only: R=I |t|=1.8m s=1", Quaternionf::Identity(),
               Vector3f(1.6f, -0.6f, 0.5f), 1.0, false);

    // 7. two-level frozen chain (P <- C1 <- C2) with a Sim3 merge: both links carried.
    {
        Result r("7 two-level chain (P<-C1<-C2) + Sim3 merge");
        const int nTrial = 200;
        for(int t = 0; t < nTrial; ++t)
        {
            SE3f Pp(PoseRule(Quaternionf::Identity(), Vector3f(1.0f, 0.1f, 0.3f), 1.0));
            SE3f P1(PoseRule(RandAxis(rng, 20.0f, Vector3f::UnitX()), Vector3f(1.2f, 0.4f, 0.1f), 1.0));
            SE3f P2(PoseRule(RandAxis(rng, -12.0f, Vector3f::UnitY()), Vector3f(1.5f, 0.7f, 0.2f), 1.0));
            const SE3f m1 = P1 * Pp.inverse();
            const SE3f m2 = P2 * P1.inverse();
            const SE3f D = PoseRule(RandAxis(rng, 75.0f, axGen), Vector3f(0.4f, -1.2f, 0.7f), 0.233926);
            const SE3f PpNew = D * Pp, P2Live = D * P2;
            r.carry_err += PosErr(Carry(m2, D) * Carry(m1, D) * PpNew, P2Live);
            r.nocarry_err += PosErr(m2 * m1 * PpNew, P2Live);
        }
        r.carry_err /= nTrial; r.nocarry_err /= nTrial;
        vRes.push_back(r);
    }

    // 8. sequence: 50 BA steps, one Sim3 merge, 50 more BA steps. The carry has to
    //    compose exactly (G2*(G1*M*G1^-1)*G2^-1 == (G2*G1)*M*(G2*G1)^-1).
    {
        Result r("8 sequence BAx50 -> Sim3 merge -> BAx50 (composition)");
        const int nTrial = 100;
        for(int t = 0; t < nTrial; ++t)
        {
            SE3f Pp(PoseRule(Quaternionf::Identity(), Vector3f(0.5f, -0.2f, 0.9f), 1.0));
            SE3f Pc(PoseRule(RandAxis(rng, 35.0f, Vector3f::UnitZ()), Vector3f(1.1f, 0.3f, 0.4f), 1.0));
            SE3f m = Pc * Pp.inverse();
            SE3f PcRef = Pc;
            for(int phase = 0; phase < 3; ++phase)
            {
                const int n = (phase == 1) ? 1 : 50;
                for(int i = 0; i < n; ++i)
                {
                    const SE3f Dp = (phase == 1)
                        ? PoseRule(RandAxis(rng, 30.0f, Vector3f(0.1f, 0.9f, 0.4f)), Vector3f(1.3f, 0.4f, -0.8f), 0.5)
                        : SE3f(RandAxis(rng, 0.3f * u(rng), Vector3f(u(rng), u(rng), u(rng) + 0.01f)),
                               0.015f * Vector3f(u(rng), u(rng), u(rng)));
                    m = Carry(m, Dp);
                    Pp = Dp * Pp;
                    PcRef = Dp * PcRef;
                }
            }
            r.carry_err += PosErr(m * Pp, PcRef);
            r.nocarry_err += PosErr((Pc * Pp.inverse()) * Pp, PcRef);
        }
        r.carry_err /= nTrial; r.nocarry_err /= nTrial;
        vRes.push_back(r);
    }

    // 9. Map::ApplyScaledRotation: Twc = P^-1; Twc.translation() *= s; P' = (T*Twc)^-1.
    //    This is NOT a rigid or similarity factor common to all keyframes, so the
    //    per-keyframe rule D = P'*P^-1 is the only meaningful reference.
    {
        Result r("9 ApplyScaledRotation s=0.2 (per-KF rule, not a rigid factor)");
        const int nTrial = 200;
        const Quaternionf Ryw = RandAxis(rng, 20.0f, Vector3f(0.1f, 0.2f, 0.97f));
        const Vector3f tyw(0.7f, -1.1f, 0.3f);
        const float s = 0.2f;
        for(int t = 0; t < nTrial; ++t)
        {
            SE3f Pp(PoseRule(Quaternionf::Identity(), Vector3f(1.0f, 0.2f, -0.4f), 1.0));
            SE3f Pc(PoseRule(RandAxis(rng, 30.0f, axGen), Vector3f(1.8f, 0.1f, -0.2f), 1.0));
            const SE3f mTcpFrozen = Pc * Pp.inverse();
            auto ApplyScaled = [&](const SE3f &P) {
                SE3f Twc = P.inverse();
                Twc.translation() *= s;
                return SE3f(Ryw, tyw) * Twc;
            };
            const SE3f PpNew = ApplyScaled(Pp).inverse();
            const SE3f PcLiveNew = ApplyScaled(Pc).inverse();
            const SE3f D = PpNew * Pp.inverse();
            r.carry_err += PosErr(Carry(mTcpFrozen, D) * PpNew, PcLiveNew);
            r.nocarry_err += PosErr(mTcpFrozen * PpNew, PcLiveNew);
        }
        r.carry_err /= nTrial; r.nocarry_err /= nTrial;
        vRes.push_back(r);
    }

    // 11. ApplyScaledRotation handled with the SAME rule instead of conjugation: the
    //     rule is applied to the culled child directly (as
    //     KeyFrame::CarryCulledDependentsScaled does), which is exact by
    //     construction; case 9 shows what conjugation would give instead.
    {
        Result r("11 ApplyScaledRotation + same-rule carry (the implemented fix)");
        const int nTrial = 200;
        const Quaternionf Ryw = RandAxis(rng, 20.0f, Vector3f(0.1f, 0.2f, 0.97f));
        const Vector3f tyw(0.7f, -1.1f, 0.3f);
        const float s = 0.2f;
        for(int t = 0; t < nTrial; ++t)
        {
            SE3f Pp(PoseRule(Quaternionf::Identity(), Vector3f(1.0f, 0.2f, -0.4f), 1.0));
            SE3f Pc(PoseRule(RandAxis(rng, 30.0f, axGen), Vector3f(1.8f, 0.1f, -0.2f), 1.0));
            const SE3f mTcpFrozen = Pc * Pp.inverse();
            auto ApplyScaled = [&](const SE3f &P) {
                SE3f Twc = P.inverse();
                Twc.translation() *= s;
                return (SE3f(Ryw, tyw) * Twc).inverse();
            };
            const SE3f PpNew = ApplyScaled(Pp);
            const SE3f PcLiveNew = ApplyScaled(Pc);
            const SE3f mTcpSameRule = PcLiveNew * PpNew.inverse();      // the fix
            r.carry_err += PosErr(mTcpSameRule * PpNew, PcLiveNew);
            r.nocarry_err += PosErr(mTcpFrozen * PpNew, PcLiveNew);
            r.control_err += PosErr(Carry(mTcpFrozen, PpNew * Pp.inverse()) * PpNew, PcLiveNew);
            r.bControl = true;
        }
        r.carry_err /= nTrial; r.nocarry_err /= nTrial; r.control_err /= nTrial;
        vRes.push_back(r);
    }

    // 10. Replay of the measured hop from the triggered mono run (f3, hop 321->318):
    //     the CARRY row recorded the applied transform as G_t=1.36553 m,
    //     G_r=67.8096 deg, and the per-hop audit measured the stored-vs-implied
    //     mismatch as |dT|=1.383 m, dA=68.23 deg. The model says the implied link is
    //     mTcp * G^-1, whose rotation differs from mTcp by exactly G's rotation
    //     angle, and whose translation differs by a term of the order of G's
    //     translation - so the model must reproduce those measured numbers.
    {
        Result r("10 replay of the measured hop (G_r=67.81deg, G_t=1.3655m)");
        const Quaternionf R_g = RandAxis(rng, 67.8096f, axGen);
        const Vector3f t_g = 1.36553f * Vector3f(0.3f, -0.8f, 0.5f).normalized();
        const SE3f G(R_g, t_g);
        const SE3f mTcpMeasured(RandAxis(rng, 5.0f, Vector3f::UnitZ()), Vector3f(0.0478f, 0.0220f, 0.0979f));
        const SE3f implied = mTcpMeasured * G.inverse();   // P_child * P_parent_new^-1
        const Quaternionf dq = (mTcpMeasured.unit_quaternion().conjugate() * implied.unit_quaternion()).normalized();
        const double dA = 2.0 * std::atan2(dq.vec().norm(), std::fabs(dq.w())) * 180.0 / M_PI;
        printf("\n  model replay: dA(model)=%.4f deg vs measured 68.23 deg ;"
               " |dT|(model)=%.4f m vs measured 1.383 m\n",
               dA, (implied.translation() - mTcpMeasured.translation()).norm());
        (void)r;
    }

    printf("\n  %-56s %11s %11s %12s\n", "case", "carry [m]", "no carry [m]", "control [m]");
    printf("  %-56s %11s %11s %12s\n", "", "", "(defect size)", "");
    for(const Result &r : vRes)
    {
        printf("  %-56s %11.6f %11.6f", r.name, r.carry_err, r.nocarry_err);
        if(!r.bControl)
            printf(" %12s\n", "-");
        else
            printf(" %12.6f\n", r.control_err);
    }
    printf("\n  case 1 control  = |carry - no carry| (no exact reference exists);\n"
           "  case 3 control  = conjugation by the Sim3 instead of the pose rule.\n\n");
    return 0;
}
