/**
* Offline replay of a captured pose-optimisation state pack.
*
* Reads a file produced by PoCaptureWrite() (real frame, real geometry, real
* observations, real incoming outlier flags), rebuilds an equivalent Frame and
* MapPoint set, and runs the REAL optimiser functions on it:
*
*     Optimizer::PoseOptimization       (native point-only reference)
*     Optimizer::PoseOptimizationWithLine
*
* The binary is built twice, once against the "old" library (neither
* re-evaluation compiled in) and once against the "new" library (both), so the
* SAME captured state is replayed through both implementations. Only final
* state is compared; per-iteration edge levels live inside the optimiser and
* are not exposed without changing it, which is deliberately not done.
*
* IMPORTANT SCOPE LIMIT
*---------------------
* The Tracking-level acceptance value is
*     nmatchesMap = #{i : !mvbOutlier[i] && mvpMapPoints[i]->Observations() > 0}
* Observations() lives in the KeyFrame map of the MapPoint and cannot be
* reconstructed offline without a KeyFrame. The captured `obs` field is used as
* an IMMUTABLE input to recompute that predicate, which is valid because the
* optimiser never changes Observations(). Any other Tracking-level quantity is
* NOT reconstructed here.
*
* Usage: replay_po_state <capture.txt> <tag: point|line> <label>
*/

#include "Frame.h"
#include "MapPoint.h"
#include "Optimizer.h"
#include "CameraModels/Pinhole.h"

#include <Eigen/Core>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using ORB_SLAM3::Frame;
using ORB_SLAM3::MapPoint;

namespace
{
struct Rec
{
    int idx = 0;
    bool has = false;
    float Xw[3] = {0, 0, 0};
    long obs = 0;
    int bad = 0;
    float u = 0, v = 0;
    int octave = 0;
    float right = -1.f;
    int outlier = 0;
};

struct Pack
{
    std::string sourceFn;
    long long frameId = 0;
    double ts = 0;
    int N = 0, NL = 0, Nleft = 0, NLleft = 0;
    int hasCam2 = 0;
    float q[4] = {0, 0, 0, 1};
    float t[3] = {0, 0, 0};
    std::vector<float> sigma2;
    std::vector<Rec> pts;
};
} // namespace

static bool ReadPack(const char* path, Pack& p)
{
    std::ifstream in(path);
    if (!in.good()) return false;
    std::string line;
    while (std::getline(in, line))
    {
        std::istringstream is(line);
        std::string tag;
        is >> tag;
        if (tag == "source_fn") is >> p.sourceFn;
        else if (tag == "frame_id") is >> p.frameId;
        else if (tag == "timestamp") is >> p.ts;
        else if (tag == "N") { is >> p.N >> tag >> p.NL >> tag >> p.Nleft >> tag >> p.NLleft; }
        else if (tag == "has_camera2") is >> p.hasCam2;
        else if (tag == "pose") is >> p.q[0] >> p.q[1] >> p.q[2] >> p.q[3] >> p.t[0] >> p.t[1] >> p.t[2];
        else if (tag == "sigma2") { float f; while (is >> f) p.sigma2.push_back(f); }
        else if (tag == "pt")
        {
            Rec r;
            is >> r.idx >> r.has;
            is >> r.Xw[0] >> r.Xw[1] >> r.Xw[2] >> r.obs >> r.bad;
            is >> r.u >> r.v >> r.octave >> r.right >> r.outlier;
            p.pts.push_back(r);
        }
        else if (tag == "end") break;
    }
    return p.N > 0;
}

int main(int argc, char** argv)
{
    if (argc < 4) { std::printf("usage: %s <capture.txt> <point|line> <label>\n", argv[0]); return 2; }
    Pack p;
    if (!ReadPack(argv[1], p)) { std::printf("cannot read %s\n", argv[1]); return 2; }
    const std::string tag = argv[2], label = argv[3];

    ORB_SLAM3::Pinhole cam(std::vector<float>{535.4f, 539.2f, 320.1f, 247.6f});

    Frame f;
    f.N = p.N; f.NL = p.NL; f.Nleft = p.Nleft; f.NLleft = p.NLleft;
    f.mpCamera = &cam; f.mpCamera2 = nullptr;
    f.mvKeysUn.assign(p.N, cv::KeyPoint());
    f.mvpMapPoints.assign(p.N, nullptr);
    f.mvuRight.assign(p.N, -1.f);
    f.mvbOutlier.assign(p.N, false);
    f.mvInvLevelSigma2 = p.sigma2.empty() ? std::vector<float>(8, 1.0f) : p.sigma2;
    f.mvpMapLines.assign(p.NL, nullptr);
    f.mvbOutlierLines.assign(p.NL, false);
    f.mvbLineOutlier.assign(p.NL, false);
    f.SetPose(Sophus::SE3f(Eigen::Quaternionf(p.q[3], p.q[0], p.q[1], p.q[2]),
                           Eigen::Vector3f(p.t[0], p.t[1], p.t[2])));

    std::vector<MapPoint*> owned;
    std::vector<long> obs;
    for (size_t k = 0; k < p.pts.size(); ++k)
    {
        const Rec& r = p.pts[k];
        const int i = r.idx;
        if (i < 0 || i >= p.N) continue;
        f.mvKeysUn[i].pt.x = r.u; f.mvKeysUn[i].pt.y = r.v;
        f.mvKeysUn[i].octave = r.octave;
        f.mvuRight[i] = r.right;
        f.mvbOutlier[i] = (r.outlier != 0);
        obs.push_back(r.obs);
        if (r.has)
        {
            MapPoint* mp = new MapPoint();
            mp->SetWorldPos(Eigen::Vector3f(r.Xw[0], r.Xw[1], r.Xw[2]));
            owned.push_back(mp);
            f.mvpMapPoints[i] = mp;
        }
    }

    int ret = 0;
    if (tag == "point") ret = ORB_SLAM3::Optimizer::PoseOptimization(&f);
    else                ret = ORB_SLAM3::Optimizer::PoseOptimizationWithLine(&f);

    // ------------------------------------------------------------------
    // Four SEPARATE quantities, never equated:
    //   assocPt   points that actually carry a MapPoint association (an edge
    //             is only built for those; unassociated points are NOT edges)
    //   edgesPt   point edges handed to g2o (mono: equals assocPt, since the
    //             mono branch builds one edge per association with mvuRight<0)
    //   obsInl    unique observations that are inliers: associated AND
    //             !outlier AND Observations()>0  (the value Tracking's
    //             nmatchesMap is built from). A default-false outlier flag on
    //             an UNASSOCIATED point never counts here.
    //   ret       the optimiser's OWN return value =
    //             nInitialCorrespondences - nBad(last iteration); it is a
    //             correspondence count, not an observation-inlier count.
    // ------------------------------------------------------------------
    int assocPt = 0, edgesPt = 0, obsInl = 0;
    for (int i = 0; i < f.N; ++i)
    {
        if (!f.mvpMapPoints[i]) continue;
        ++assocPt;
        if (i < (int)f.mvuRight.size() && f.mvuRight[i] < 0) ++edgesPt;
        if (!f.mvbOutlier[i] && obs[i] > 0) ++obsInl;
    }
    const Sophus::SE3f T = f.GetPose();
    const Eigen::Vector3f t = T.translation();

    std::printf("%-6s %-22s N=%2d NL=%2d assocPt=%2d edgesPt=%2d obsInl=%2d ret=%2d"
                " pose=(%.6f,%.6f,%.6f)\n",
                label.c_str(), p.sourceFn.c_str(), p.N, p.NL, assocPt, edgesPt,
                obsInl, ret, t.x(), t.y(), t.z());

    for (auto* m : owned) delete m;
    return 0;
}
