// Unified SLAM-only evaluation entry for cross-dataset front-end evaluation.
//
// Why this file exists: examples/tum_mono.cpp and examples/tum_rgbd.cpp both
// pull in the Gaussian mapper / ImGui viewer, which drags torch + CUDA into the
// link and makes the evaluation binary depend on the learned backend. Front-end
// evaluation must depend only on ORB-SLAM3, so this entry instantiates
// ORB_SLAM3::System exactly like the production entries do with
// PHOTO_SLAM_SLAM_ONLY=1, and additionally drains the mapping-operation queue
// that the Gaussian mapper would otherwise consume.
//
// Tracking calls are identical to production:
//   mono : System::TrackMonocularWithLine(im, ts, {}, filename)
//   rgbd : System::TrackRGBDWithLine(imRGB, imD, ts, {}, filename)
// No threshold, initialisation, covisibility or acceptance logic is modified.
//
// Usage:
//   eval_frontend <vocab> <slam_cfg> <gauss_cfg_ignored> <seq_dir> <out_dir> \
//                 <mono|rgbd> [assoc_file]
//
// Outputs in <out_dir>:
//   eval_entry.txt            banner + effective configuration snapshot
//   final_state.csv           per-frame tracking state after each Track* call
//   CameraTrajectory_TUM.txt  / KeyFrameTrajectory_TUM.txt
//   TrackingTime.txt

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/include/MapEventLog.h"

namespace fs = std::filesystem;

static const char *StateName(int s)
{
    switch (s)
    {
    case -1: return "SYSTEM_NOT_READY";
    case 0:  return "NO_IMAGES_YET";
    case 1:  return "NOT_INITIALIZED";
    case 2:  return "OK";
    case 3:  return "RECENTLY_LOST";
    case 4:  return "LOST";
    default: return "UNKNOWN";
    }
}

static std::string EnvOr(const char *k, const char *d)
{
    const char *v = std::getenv(k);
    return v ? std::string(v) : std::string(d);
}

static double Ms(std::chrono::steady_clock::time_point a,
                 std::chrono::steady_clock::time_point b)
{
    return std::chrono::duration_cast<std::chrono::duration<double>>(b - a).count() * 1e3;
}

// association file: "ts rgb/xxx.png ts depth/xxx.png"
static void LoadAssoc(const std::string &f,
                      std::vector<std::string> &rgb,
                      std::vector<std::string> &dep,
                      std::vector<double> &ts)
{
    std::ifstream in(f);
    if (!in)
    {
        std::cerr << "Cannot open association file: " << f << std::endl;
        std::exit(2);
    }
    std::string line;
    while (std::getline(in, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        double t1, t2;
        std::string r, d;
        if (!(ss >> t1 >> r >> t2 >> d)) continue;
        ts.push_back(t1);
        rgb.push_back(r);
        dep.push_back(d);
    }
}

// timestamp list file: one timestamp per line (mono)
static void LoadStamps(const std::string &f, std::vector<double> &ts)
{
    std::ifstream in(f);
    if (!in)
    {
        std::cerr << "Cannot open timestamp file: " << f << std::endl;
        std::exit(2);
    }
    double t;
    while (in >> t) ts.push_back(t);
}

int main(int argc, char **argv)
{
    if (argc < 7 || argc > 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <vocab> <slam_cfg> <gauss_cfg_unused> <seq_dir> <out_dir>"
                  << " <mono|rgbd> [assoc_file]\n";
        return 1;
    }

    const std::string vocab = argv[1];
    const std::string slamCfg = argv[2];
    const std::string seqDir = argv[4];
    const fs::path outDir(argv[5]);
    const std::string modeArg = argv[6];

    // "replica-mono"/"replica-rgbd" reuse the whole mono/rgbd path; only the
    // frame list differs (Replica ships numbered images, no timestamp file).
    const bool replica = (modeArg.rfind("replica-", 0) == 0);
    const std::string mode = (modeArg == "replica-rgbd") ? "rgbd"
                             : (modeArg == "replica-mono") ? "mono"
                                                            : modeArg;
    if (mode != "mono" && mode != "rgbd")
    {
        std::cerr << "mode must be mono|rgbd|replica-mono|replica-rgbd\n";
        return 1;
    }

    // ---- explicit arm dispatch -------------------------------------------
    // The arm is taken from EVAL_ARM, NOT inferred from PHOTO_SLAM_LINE_MODE.
    // T0 must call the pure-point tracking API (no line code path is entered at
    // all); P1/P2 call the WithLine API and differ only by the line edges that
    // PHOTO_SLAM_LINE_MODE enables inside the optimizers.
    const std::string arm = EnvOr("EVAL_ARM", "");
    if (arm != "T0" && arm != "P1" && arm != "P2")
    {
        std::cerr << "EVAL_ARM must be T0|P1|P2 (got '" << arm << "')\n";
        return 1;
    }
    const std::string lineMode = EnvOr("PHOTO_SLAM_LINE_MODE", "2");
    const std::string want = (arm == "T0" ? "0" : (arm == "P1" ? "1" : "2"));
    if (lineMode != want)
    {
        std::cerr << "arm/mode mismatch: EVAL_ARM=" << arm << " wants "
                  << "PHOTO_SLAM_LINE_MODE=" << want << " but got " << lineMode
                  << "\n";
        return 1;
    }
    const bool useLineApi = (arm != "T0");

    fs::create_directories(outDir);
    std::ofstream banner(outDir / "eval_entry.txt");
    banner << "entry=eval_frontend.cpp (SLAM-only, no Gaussian mapper)\n";
    banner << "mode=" << mode << "\n";
    banner << "arm=" << arm << "\n";
    banner << "line_api=" << (useLineApi ? "WithLine" : "pure-point") << "\n";
    banner << "api=" << (mode == "rgbd"
                            ? (useLineApi ? "TrackRGBDWithLine" : "TrackRGBD")
                            : (useLineApi ? "TrackMonocularWithLine" : "TrackMonocular"))
           << "\n";
    banner << "vocab=" << vocab << "\n";
    banner << "slam_cfg=" << slamCfg << "\n";
    banner << "gauss_cfg_ignored=" << argv[3] << "\n";
    banner << "seq_dir=" << seqDir << "\n";
    banner << "out_dir=" << outDir.string() << "\n";
    banner << "assoc=" << (argc == 8 ? argv[7] : std::string("<none>")) << "\n";
    banner << "env_LINE_MODE=" << EnvOr("LINE_MODE", "(cfg)") << "\n";
    banner << "env_PHOTO_SLAM_SLAM_ONLY=" << EnvOr("PHOTO_SLAM_SLAM_ONLY", "(unset)")
           << "\n";
    banner << "env_PHOTO_SLAM_NO_PACING=" << EnvOr("PHOTO_SLAM_NO_PACING", "(unset)")
           << "\n";
    banner << "env_ORB_SLAM3_LINE_MODE=" << EnvOr("ORB_SLAM3_LINE_MODE", "(unset)")
           << "\n";
    banner.flush();

    // ---- [MEASUREMENT] complete pose products -----------------------------
    // 1. Map lifecycle events. The library writes them to a CSV when
    //    PHOTO_SLAM_MAP_EVENTS_CSV is set; point it at this run's directory
    //    unless the caller already chose a path (setenv(..., 0) never
    //    overrides). Read-only observer: no estimation path reads it.
    // 2. Per-input-frame online pose goes to online_poses.csv, written in the
    //    tracking loop below; final_state.csv keeps its original 4 columns so
    //    existing consumers are unaffected.
    // 3. Every map's final keyframe trajectory goes to traj_map<id>.txt via
    //    System::SaveAllMapsTrajectoryTUM, because SaveTrajectoryTUM exports
    //    only the single map chosen by SelectExportMap.
    const std::string mapEventsCsv = (outDir / "map_events.csv").string();
    setenv("PHOTO_SLAM_MAP_EVENTS_CSV", mapEventsCsv.c_str(), 0);
    {
        std::ofstream b2(outDir / "eval_entry.txt", std::ios::app);
        b2 << "map_events_csv=" << mapEventsCsv << "\n";
        b2 << "map_events_gate="
           << (EnvOr("PHOTO_SLAM_MAP_EVENTS_CSV", "") == mapEventsCsv ?
                   "set_by_entry" : "set_by_caller")
           << "\n";
        b2 << "online_poses_csv=" << (outDir / "online_poses.csv").string()
           << "\n";
        b2 << "all_maps_export_prefix=" << (outDir / "").string() << "\n";
    }

    std::vector<std::string> vRGB, vD;
    std::vector<double> vTs;
    if (replica)
    {
        // Replica: <seq>/results/frame%06d.jpg and depth%06d.png, no timestamps.
        // A synthetic time base of 1/30 s per index is recorded in eval_entry.txt;
        // the values are indices, not measurements.
        const std::string res = seqDir + "/results";
        char buf[64];
        for (int i = 0;; i++)
        {
            snprintf(buf, sizeof(buf), "/frame%06d.jpg", i);
            if (!fs::exists(res + buf)) break;
            vRGB.push_back(std::string("results") + buf);
            snprintf(buf, sizeof(buf), "results/depth%06d.png", i);
            vD.push_back(buf);
            vTs.push_back(i / 30.0);
        }
        if (vRGB.empty())
        {
            std::cerr << "no Replica frames under " << res << "/frameNNNNNN.jpg\n";
            return 1;
        }
        if (mode == "rgbd")
        {
            for (size_t k = 0; k < vD.size(); k++)
                if (!fs::exists(seqDir + "/" + vD[k]))
                {
                    std::cerr << "missing depth for frame " << k << ": " << vD[k] << "\n";
                    return 1;
                }
        }
    }
    else if (mode == "rgbd")
    {
        if (argc != 8) { std::cerr << "rgbd needs assoc_file\n"; return 1; }
        LoadAssoc(argv[7], vRGB, vD, vTs);
        // Reject a non-monotonic association list instead of silently feeding
        // frames out of order (this exact defect invalidated an earlier batch).
        for (size_t k = 1; k < vTs.size(); k++)
            if (vTs[k] <= vTs[k - 1])
            {
                std::cerr << "association file is not time-ordered at line " << k
                          << " (" << vTs[k - 1] << " then " << vTs[k] << ")\n";
                return 1;
            }
    }
    else if (mode == "mono")
    {
        fs::path tf = fs::path(seqDir) / "timestamps.txt";
        if (!fs::exists(tf)) tf = fs::path(seqDir) / "rgb.txt";
        if (fs::exists(tf))
        {
            // rgb.txt style: "ts rgb/xxx.png" -> reuse assoc loader with dep==""
            std::ifstream in(tf.string());
            std::string line;
            while (std::getline(in, line))
            {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream ss(line);
                double t;
                std::string r;
                if (!(ss >> t >> r)) continue;
                vTs.push_back(t);
                vRGB.push_back(r);
            }
        }
        else
        {
            std::cerr << "mono needs " << tf.string() << " or rgb.txt\n";
            return 1;
        }
    }
    else
    {
        std::cerr << "mode must be mono|rgbd\n";
        return 1;
    }

    const int nImages = (int)vTs.size();
    if (nImages == 0) { std::cerr << "no frames\n"; return 1; }

    std::cerr << "[EVAL-ENTRY] mode=" << mode << " frames=" << nImages
              << " seq=" << seqDir << " out=" << outDir.string() << std::endl;
    std::cerr << "[EVAL-DISPATCH] arm=" << arm << " line_mode=" << lineMode
              << " use_line_api=" << (useLineApi ? 1 : 0)
              << " api="
              << (mode == "rgbd"
                      ? (useLineApi ? "TrackRGBDWithLine" : "TrackRGBD")
                      : (useLineApi ? "TrackMonocularWithLine" : "TrackMonocular"))
              << std::endl;

    auto pSLAM = std::make_shared<ORB_SLAM3::System>(
        vocab, slamCfg,
        mode == "rgbd" ? ORB_SLAM3::System::RGBD : ORB_SLAM3::System::MONOCULAR);
    const float imageScale = pSLAM->GetImageScale();

    std::ofstream fsCsv(outDir / "final_state.csv");
    fsCsv << "frame,timestamp,state,state_name\n";

    // Per-input-frame online pose. pose_valid=1 means the state was OK and the
    // transform is finite; only those rows are poses of the frame at the time
    // it was tracked. map_id/ref_kf_id identify which map produced the row, so
    // rows of different world frames are never stitched together silently.
    std::ofstream fsPose(outDir / "online_poses.csv");
    fsPose << "frame,timestamp,state,state_name,pose_valid,map_id,ref_kf_id,"
              "frame_id,tx,ty,tz,qx,qy,qz,qw\n";

    std::vector<float> vTimesTrack(nImages, 0.f);
    std::vector<double> vState(nImages, -2.0);
    // Entry-level stage timing (wall clock, non-overlapping buckets).
    double t_read = 0, t_cvt = 0, t_depth = 0, t_track = 0, t_drain = 0, t_pace = 0;

    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown()) break;

        auto c0 = std::chrono::steady_clock::now();
        cv::Mat im = cv::imread(seqDir + "/" + vRGB[ni], cv::IMREAD_UNCHANGED);
        auto c1 = std::chrono::steady_clock::now();
        t_read += Ms(c0, c1);
        if (im.empty())
        {
            std::cerr << "Failed to load image: " << seqDir << "/" << vRGB[ni]
                      << std::endl;
            return 1;
        }
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);
        auto c2 = std::chrono::steady_clock::now();
        t_cvt += Ms(c1, c2);

        cv::Mat imD;
        if (mode == "rgbd")
        {
            auto d0 = std::chrono::steady_clock::now();
            imD = cv::imread(seqDir + "/" + vD[ni], cv::IMREAD_UNCHANGED);
            auto d1 = std::chrono::steady_clock::now();
            t_depth += Ms(d0, d1);
            if (imD.empty())
            {
                std::cerr << "Failed to load depth: " << seqDir << "/" << vD[ni]
                          << std::endl;
                return 1;
            }
        }

        if (imageScale != 1.f)
        {
            const cv::Size sz((int)(im.cols * imageScale), (int)(im.rows * imageScale));
            cv::resize(im, im, sz);
            if (mode == "rgbd") cv::resize(imD, imD, sz);
        }

        const double tframe = vTs[ni];
        // Publish the input frame index before tracking so any map lifecycle
        // event emitted during this call is bound to this frame (read-only).
        {
            ORB_SLAM3::map_events::SetFrameIndex(ni);
        }
        auto t1 = std::chrono::steady_clock::now();

        if (mode == "rgbd")
        {
            if (useLineApi)
                pSLAM->TrackRGBDWithLine(im, imD, tframe,
                                         std::vector<ORB_SLAM3::IMU::Point>(), vRGB[ni]);
            else
                pSLAM->TrackRGBD(im, imD, tframe,
                                 std::vector<ORB_SLAM3::IMU::Point>(), vRGB[ni]);
        }
        else
        {
            if (useLineApi)
                pSLAM->TrackMonocularWithLine(im, tframe,
                                              std::vector<ORB_SLAM3::IMU::Point>(), vRGB[ni]);
            else
                pSLAM->TrackMonocular(im, tframe,
                                      std::vector<ORB_SLAM3::IMU::Point>(), vRGB[ni]);
        }

        auto t2 = std::chrono::steady_clock::now();
        t_track += Ms(t1, t2);

        // SLAM-only: nothing consumes the mapping-operation queue.
        if (pSLAM->getAtlas() != nullptr) pSLAM->getAtlas()->clearMappingOperation();
        auto t3 = std::chrono::steady_clock::now();
        t_drain += Ms(t2, t3);

        const double ttrack = Ms(t1, t2) * 1e-3;
        vTimesTrack[ni] = (float)ttrack;

        const int st = pSLAM->GetTrackingState();
        vState[ni] = st;
        fsCsv << ni << "," << std::fixed << tframe << "," << st << ","
              << StateName(st) << "\n";
        fsCsv.flush();

        // Online pose of THIS frame, read back from the tracker right after the
        // Track* call returned (single-threaded here). Nothing is corrected or
        // re-derived: this is the pose the system had for the frame at the time.
        {
            long mapId = -1, refKfId = -1;
            unsigned long frameId = 0;
            Sophus::SE3f Twc;
            bool poseValid = false;
            ORB_SLAM3::Tracking* pTr = pSLAM->getTracker();
            if (pTr)
            {
                const ORB_SLAM3::Frame& fr = pTr->mCurrentFrame;
                frameId = fr.mnId;
                Twc = fr.GetPose().inverse();
                poseValid = (st == 2) && Twc.translation().allFinite();
                if (fr.mpReferenceKF)
                {
                    refKfId = (long)fr.mpReferenceKF->mnId;
                    if (fr.mpReferenceKF->GetMap())
                        mapId = (long)fr.mpReferenceKF->GetMap()->GetId();
                }
            }
            const Eigen::Vector3f t = Twc.translation();
            const Eigen::Quaternionf q = Twc.unit_quaternion();
            fsPose << ni << "," << std::fixed << tframe << "," << st << ","
                   << StateName(st) << "," << (poseValid ? 1 : 0) << ","
                   << mapId << "," << refKfId << "," << frameId << ","
                   << t(0) << "," << t(1) << "," << t(2) << ","
                   << q.x() << "," << q.y() << "," << q.z() << ","
                   << q.w() << "\n";
            fsPose.flush();
        }

        double T = 0;
        if (ni < nImages - 1) T = vTs[ni + 1] - tframe;
        else if (ni > 0) T = tframe - vTs[ni - 1];

        const bool skipPacing = (EnvOr("PHOTO_SLAM_NO_PACING", "0") == "1");
        if (!skipPacing && ttrack < T)
        {
            const auto p0 = std::chrono::steady_clock::now();
            usleep((T - ttrack) * 1e6);
            t_pace += Ms(p0, std::chrono::steady_clock::now());
        }
    }

    fsCsv.close();
    pSLAM->Shutdown();

    {
        std::ofstream st(outDir / "TrackingTime.txt");
        double sum = 0;
        for (int i = 0; i < nImages; i++)
        {
            st << vTimesTrack[i] << "\n";
            sum += vTimesTrack[i];
        }
        st << "mean=" << (nImages ? sum / nImages : 0.0) << "\n";
    }

    pSLAM->SaveTrajectoryTUM((outDir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((outDir / "KeyFrameTrajectory_TUM.txt").string());
    // Every map, not only the one SelectExportMap picked above.
    pSLAM->SaveAllMapsTrajectoryTUM((outDir / "").string());

    int nOk = 0, nLost = 0, nOther = 0;
    for (double s : vState)
    {
        if ((int)s == 2) nOk++;
        else if ((int)s == 4) nLost++;
        else nOther++;
    }
    const double tot = t_read + t_cvt + t_depth + t_track + t_drain + t_pace;
    std::cerr << "[EVAL-TIMING] frames=" << nImages
              << " read_rgb_ms=" << t_read << " cvt_ms=" << t_cvt
              << " read_depth_ms=" << t_depth << " track_ms=" << t_track
              << " drain_ms=" << t_drain << " pace_ms=" << t_pace
              << " sum_ms=" << tot << std::endl;
    std::cerr << "[EVAL-ENTRY] done frames_processed=" << nImages
              << " final_state=" << (nImages ? (int)vState[nImages - 1] : -2)
              << " ok=" << nOk << " lost=" << nLost << " other=" << nOther << std::endl;
    return 0;
}
