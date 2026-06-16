/**
 * This file is part of Photo-SLAM (Customized for ROVER Dataset - Monocular Mode)
 */

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <memory>

#include <opencv2/core/core.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

// 🌟 修改函数声明
void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
                std::vector<double> &vTimestamps);
void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath);
void saveGpuPeakMemoryUsage(std::filesystem::path pathSave);

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"                   /*1*/
                  << " path_to_ORB_SLAM3_settings"           /*2*/
                  << " path_to_gaussian_mapping_settings"    /*3*/
                  << " path_to_sequence"                     /*4*/
                  << " path_to_trajectory_output_directory/" /*5*/
                  << " (optional)no_viewer"                  /*6*/
                  << std::endl;
        return 1;
    }
    bool use_viewer = true;
    if (argc == 7)
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);

    // Retrieve paths to images
    std::vector<std::string> vstrImageFilenamesRGB;
    std::vector<double> vTimestamps;
    std::string strFile = std::string(argv[4]) + "/rgb.txt";
    LoadImages(strFile, vstrImageFilenamesRGB, vTimestamps);

    // Check consistency in the number of images
    int nImages = vstrImageFilenamesRGB.size();
    if (vstrImageFilenamesRGB.empty())
    {
        std::cerr << std::endl << "❌ No images found in provided path: " << strFile << std::endl;
        return 1;
    }

    // Device
    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::MONOCULAR);
    float imageScale = pSLAM->GetImageScale();

    // Create GaussianMapper
    std::filesystem::path gaussian_cfg_path(argv[3]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    // Create Gaussian Viewer
    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    // Vector for tracking time statistics
    std::vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    std::cout << std::endl << "=======================================" << std::endl;
    std::cout << "🚀 Start processing ROVER Monocular sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << nImages << std::endl;
    std::cout << "=======================================" << std::endl << std::endl;
    
    // 记录整个系统启动的总起始时间
    std::chrono::steady_clock::time_point total_start_time = std::chrono::steady_clock::now();

    // Main loop
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        if (pSLAM->isShutDown())
            break;
            
        // 从 rgb.txt 拼接出绝对/相对路径读取
        im = cv::imread(std::string(argv[4]) + "/" + vstrImageFilenamesRGB[ni], cv::IMREAD_UNCHANGED);
        cv::cvtColor(im, im, cv::COLOR_BGR2RGB); // 升级为 OpenCV4 标准色彩转换宏
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            std::cerr << std::endl << "❌ Failed to load image at: "
                      << std::string(argv[4]) << "/" << vstrImageFilenamesRGB[ni] << std::endl;
            return 1;
        }

        if (imageScale != 1.f)
        {
            int width = im.cols * imageScale;
            int height = im.rows * imageScale;
            cv::resize(im, im, cv::Size(width, height));
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // 🌟 将图像送入单目前端
        pSLAM->TrackMonocular(im, tframe, std::vector<ORB_SLAM3::IMU::Point>(), vstrImageFilenamesRGB[ni]);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        vTimesTrack[ni] = ttrack;

        // 根据相邻帧时间戳计算真实的等待时间
        double T = 0;
        if (ni < nImages - 1)
            T = vTimestamps[ni + 1] - tframe;
        else if (ni > 0)
            T = tframe - vTimestamps[ni - 1];

        if (ttrack < T)
            usleep((T - ttrack) * 1e6);
    }

    // Stop all threads
    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer)
        viewer_thd.join();
        
    // 计算总耗时
    std::chrono::steady_clock::time_point total_end_time = std::chrono::steady_clock::now();
    double total_operation_time_sec = std::chrono::duration_cast<std::chrono::duration<double>>(total_end_time - total_start_time).count();
    double total_operation_time_min = total_operation_time_sec / 60.0;

    std::cout << "=====================================================" << std::endl;
    std::cout << "[ROVER System] Total Operation Time: " << total_operation_time_sec << " seconds ("
              << total_operation_time_min << " minutes)." << std::endl;
    std::cout << "=====================================================" << std::endl;

    std::cout << "All threads finished. Saving results..." << std::endl;

    // 自动将总运行时间保存到 txt 文件
    std::ofstream total_time_out((output_dir / "TotalOperationTime.txt").string());
    if (total_time_out.is_open()) {
        total_time_out << "Total Operation Time (seconds): " << total_operation_time_sec << std::endl;
        total_time_out << "Total Operation Time (minutes): " << total_operation_time_min << std::endl;
        total_time_out.close();
    }

    // GPU 内存数据保存
    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");

    // 前端追踪耗时数据保存
    saveTrackingTime(vTimesTrack, (output_dir / "TrackingTime.txt").string());

    // 保存多格式相机轨迹文件
    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    return 0;
}

// 🌟 [核心修改] 升级后的安全多格式图像加载函数
void LoadImages(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
                std::vector<double> &vTimestamps)
{
    std::ifstream f;
    f.open(strFile.c_str());
    if (!f.is_open())
    {
        std::cerr << "❌ Cannot open file: " << strFile << std::endl;
        return;
    }

    while (!f.eof())
    {
        std::string s;
        std::getline(f, s);
        if (!s.empty())
        {
            // 💡 如果行首是注释符号 '#'，则动态跳过，完美兼容任意格式的 rgb.txt 文件
            if (s[0] == '#')
                continue;

            std::stringstream ss;
            ss << s;
            double t;
            std::string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
    f.close();
}

void saveTrackingTime(std::vector<float> &vTimesTrack, const std::string &strSavePath)
{
    std::ofstream out;
    out.open(strSavePath.c_str());
    std::size_t nImages = vTimesTrack.size();
    for (int ni = 0; ni < nImages; ni++)
    {
        out << std::fixed << std::setprecision(4)
            << vTimesTrack[ni] << std::endl;
    }
    out.close();
}

void saveGpuPeakMemoryUsage(std::filesystem::path pathSave)
{
    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);
    
    c10::CachingAllocator::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10::CachingAllocator::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);
    
    c10::CachingAllocator::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10::CachingAllocator::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    std::ofstream out(pathSave);
    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
    out.close();
}