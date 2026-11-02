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
    const std::string mode = argv[6];

    fs::create_directories(outDir);
    std::ofstream banner(outDir / "eval_entry.txt");
    banner << "entry=eval_frontend.cpp (SLAM-only, no Gaussian mapper)\n";
    banner << "mode=" << mode << "\n";
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

    std::vector<std::string> vRGB, vD;
    std::vector<double> vTs;
    if (mode == "rgbd")
    {
        if (argc != 8) { std::cerr << "rgbd needs assoc_file\n"; return 1; }
        LoadAssoc(argv[7], vRGB, vD, vTs);
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

    auto pSLAM = std::make_shared<ORB_SLAM3::System>(
        vocab, slamCfg,
        mode == "rgbd" ? ORB_SLAM3::System::RGBD : ORB_SLAM3::System::MONOCULAR);
    const float imageScale = pSLAM->GetImageScale();

    std::ofstream fsCsv(outDir / "final_state.csv");
    fsCsv << "frame,timestamp,state,state_name\n";

    std::vector<float> vTimesTrack(nImages, 0.f);
    std::vector<double> vState(nImages, -2.0);

    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown()) break;

        cv::Mat im = cv::imread(seqDir + "/" + vRGB[ni], cv::IMREAD_UNCHANGED);
        if (im.empty())
        {
            std::cerr << "Failed to load image: " << seqDir << "/" << vRGB[ni]
                      << std::endl;
            return 1;
        }
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB);

        cv::Mat imD;
        if (mode == "rgbd")
        {
            imD = cv::imread(seqDir + "/" + vD[ni], cv::IMREAD_UNCHANGED);
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
        auto t1 = std::chrono::steady_clock::now();

        if (mode == "rgbd")
            pSLAM->TrackRGBDWithLine(im, imD, tframe,
                                     std::vector<ORB_SLAM3::IMU::Point>(), vRGB[ni]);
        else
            pSLAM->TrackMonocularWithLine(im, tframe,
                                          std::vector<ORB_SLAM3::IMU::Point>(), vRGB[ni]);

        auto t2 = std::chrono::steady_clock::now();

        // SLAM-only: nothing consumes the mapping-operation queue.
        if (pSLAM->getAtlas() != nullptr) pSLAM->getAtlas()->clearMappingOperation();

        const double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = (float)ttrack;

        const int st = pSLAM->GetTrackingState();
        vState[ni] = st;
        fsCsv << ni << "," << std::fixed << tframe << "," << st << ","
              << StateName(st) << "\n";
        fsCsv.flush();

        double T = 0;
        if (ni < nImages - 1) T = vTs[ni + 1] - tframe;
        else if (ni > 0) T = tframe - vTs[ni - 1];

        const bool skipPacing = (EnvOr("PHOTO_SLAM_NO_PACING", "0") == "1");
        if (!skipPacing && ttrack < T) usleep((T - ttrack) * 1e6);
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

    int nOk = 0, nLost = 0, nOther = 0;
    for (double s : vState)
    {
        if ((int)s == 2) nOk++;
        else if ((int)s == 4) nLost++;
        else nOther++;
    }
    std::cerr << "[EVAL-ENTRY] done frames_processed=" << nImages
              << " final_state=" << (nImages ? (int)vState[nImages - 1] : -2)
              << " ok=" << nOk << " lost=" << nLost << " other=" << nOther << std::endl;
    return 0;
}
