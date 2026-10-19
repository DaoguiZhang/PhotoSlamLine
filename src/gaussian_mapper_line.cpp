/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
 * Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include "include/gaussian_mapper_line.h"

#include <ATen/cuda/CUDAEvent.h>

#include <cstdlib>
#include <string>
#include <cmath>
#include <functional>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <sstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "LineMode.h"

// defined in cuda_rasterizer/backward.cu
void SetRasterBackend(int b);
// geom-sync-v1 contract self-test (gaussian_model_line.cpp)
bool runRigidCorrectionContractTest();
// raster-perf-v6 call-trace hooks (cuda_rasterizer/rasterizer_impl.cu)
void RasterCallTraceReset();
void RasterCallTraceGet(long long* out);
void RasterCallTraceDims(int* out);
int RasterBackendGet();

namespace {
// Line structural-loss is computed every N iterations. Default 1 (every step):
// skipping steps measurably lowered PSNR (~-0.2 dB at N=5), so keep the loss
// every step and instead make the loss itself cheaper (vectorized orientation).
int lineLossEvery() {
    static const int v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_LOSS_EVERY");
        return e ? std::atoi(e) : 1;
    }();
    return v;
}

// CUDA 事件级性能剖分（opt-in，PHOTO_SLAM_GPU_PROF=1）。
// 用 cudaEventDefault(0) 打开计时；默认构造的 CUDAEvent 是 DisableTiming。
// 事件记录在 torch 当前流上，每 kSyncEvery 次迭代才同步一次并采样，
// 避免给热循环引入每迭代同步。
//
// 事件对象故意泄漏（不析构）：CUDAEvent 析构会调用 cudaEventDestroy，
// 若发生在静态对象析构期（__cxa_finalize）会抛 "CUDA error: driver
// shutting down" 导致 SIGABRT(134)。
struct GpuSectionProfiler {
    at::cuda::CUDAEvent* ev_start = nullptr;
    at::cuda::CUDAEvent* ev_fwd = nullptr;
    at::cuda::CUDAEvent* ev_line = nullptr;
    at::cuda::CUDAEvent* ev_bwd = nullptr;
    at::cuda::CUDAEvent* ev_dens = nullptr;
    at::cuda::CUDAEvent* ev_opt = nullptr;

    bool enabled = false;
    bool inited = false;
    int since_sync = 0;
    static constexpr int kSyncEvery = 50;

    struct Acc {
        double fwd = 0, line = 0, bwd = 0, dens = 0, opt = 0, total = 0;
        long iters = 0;
    };
    Acc online, refine;

    void ensureInit() {
        if (inited) return;
        inited = true;
        const char* e = std::getenv("PHOTO_SLAM_GPU_PROF");
        enabled = e && std::atoi(e) != 0 && torch::cuda::is_available();
        if (!enabled) return;
        ev_start = new at::cuda::CUDAEvent(0u);
        ev_fwd   = new at::cuda::CUDAEvent(0u);
        ev_line  = new at::cuda::CUDAEvent(0u);
        ev_bwd   = new at::cuda::CUDAEvent(0u);
        ev_dens  = new at::cuda::CUDAEvent(0u);
        ev_opt   = new at::cuda::CUDAEvent(0u);
    }
    void markStart()        { ensureInit(); if (enabled) ev_start->record(); }
    void resetAcc(bool refine) { (refine ? this->refine : this->online) = Acc{}; }
    void markForwardEnd()   { if (enabled) ev_fwd->record(); }
    void markLineLossEnd()  { if (enabled) ev_line->record(); }
    void markBackwardEnd()  { if (enabled) ev_bwd->record(); }
    void markDensifyEnd()   { if (enabled) ev_dens->record(); }
    void markOptimizerEnd() { if (enabled) ev_opt->record(); }

    void endIteration(bool refine) {
        if (!enabled) return;
        if (++since_sync < kSyncEvery) return;
        since_sync = 0;
        torch::cuda::synchronize();
        Acc& a = refine ? this->refine : this->online;
        // earlier_event.elapsed_time(later_event) = later - earlier.
        a.fwd   += ev_start->elapsed_time(*ev_fwd);
        a.line  += ev_fwd->elapsed_time(*ev_line);
        a.bwd   += ev_line->elapsed_time(*ev_bwd);
        a.dens  += ev_bwd->elapsed_time(*ev_dens);
        a.opt   += ev_dens->elapsed_time(*ev_opt);
        a.total += ev_start->elapsed_time(*ev_opt);
        a.iters += 1;
    }

    void print() const {
        if (!enabled) return;
        auto dump = [](const char* name, const Acc& a) {
            const long n = a.iters > 0 ? a.iters : 1;
            const double fwd = a.fwd / n, line = a.line / n, bwd = a.bwd / n,
                         dens = a.dens / n, opt = a.opt / n, total = a.total / n;
            const double sum = fwd + line + bwd + dens + opt;
            std::cerr << "[GpuProfiler] " << name << " (n=" << a.iters << " iters)\n"
                      << "    forward(render):   " << fwd << " ms/iter\n"
                      << "    loss_fwd(L1+SSIM+aniso+line): " << line << " ms/iter\n"
                      << "    backward:          " << bwd << " ms/iter\n"
                      << "    dens_stats(maxR+gradAccum, 每步): " << dens << " ms/iter\n"
                      << "    optimizer:         " << opt << " ms/iter\n"
                      << "    ---- sum(parts)=" << sum << "  total(start->opt)=" << total << " ms/iter\n";
        };
        dump("online (incremental mapping)", online);
        dump("refine (final refinement)", refine);
    }
};
GpuSectionProfiler g_gpu_prof;
} // namespace

GaussianMapperLine::GaussianMapperLine(
    std::shared_ptr<ORB_SLAM3::System> pSLAM,
    std::filesystem::path gaussian_config_file_path,
    std::filesystem::path result_dir,
    int seed,
    torch::DeviceType device_type)
    : pSLAM_(pSLAM),
      initial_mapped_(false),
      interrupt_training_(false),
      stopped_(false),
      iteration_(0),
      ema_loss_for_log_(0.0f),
      SLAM_ended_(false),
      loop_closure_iteration_(false),
      min_num_initial_map_kfs_(15UL),
      large_rot_th_(1e-1f),
      large_trans_th_(1e-2f),
      training_report_interval_(0)
{
    // Random seed. PHOTO_SLAM_SEED (if set) overrides the caller-provided seed so
    // an experiment can be made reproducible/controllable from the shell. When
    // unset, behaviour is exactly as before (random_seed_ stays -1).
    if (const char* seed_env = std::getenv("PHOTO_SLAM_SEED")) {
        if (*seed_env != '\0')
            random_seed_ = std::atoi(seed_env);
    }
    if (random_seed_ >= 0)
        seed = random_seed_;
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cerr << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cerr << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(result_dir)
    config_file_path_ = gaussian_config_file_path;
    readConfigFromFile(gaussian_config_file_path);

    std::vector<float> bg_color;
    if (model_params_.white_background_)
        bg_color = {1.0f, 1.0f, 1.0f};
    else
        bg_color = {0.0f, 0.0f, 0.0f};
    background_ = torch::tensor(bg_color,
                    torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    override_color_ = torch::empty(0, torch::TensorOptions().device(device_type_));

    // Initialize scene and model
    gaussians_ = std::make_shared<GaussianModelLine>(model_params_);
    scene_ = std::make_shared<GaussianSceneLine>(model_params_);

    // Mode
    if (!pSLAM) {
        // NO SLAM
        return;
    }

    // Sensors
    switch (pSLAM->getSensorType())
    {
    case ORB_SLAM3::System::MONOCULAR:
    case ORB_SLAM3::System::IMU_MONOCULAR:
    {
        this->sensor_type_ = MONOCULARLINE;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREOLINE;
        this->stereo_baseline_length_ = pSLAM->getSettings()->b();
        this->stereo_cv_sgm_ = cv::cuda::createStereoSGM(
            this->stereo_min_disparity_,
            this->stereo_num_disparity_);
        this->stereo_Q_ = pSLAM->getSettings()->Q().clone();
        stereo_Q_.convertTo(stereo_Q_, CV_32FC3, 1.0);
    }
    break;
    case ORB_SLAM3::System::RGBD:
    case ORB_SLAM3::System::IMU_RGBD:
    {
        this->sensor_type_ = RGBDLINE;
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    // Cameras
    // TODO: not only monocular
    auto settings = pSLAM->getSettings();
    cv::Size SLAM_im_size = settings->newImSize();
    UndistortParamsLine undistort_params(
        SLAM_im_size,
        settings->camera1DistortionCoef()
    );

    auto vpCameras = pSLAM->getAtlas()->GetAllCameras();
    for (auto& SLAM_camera : vpCameras) {
        Camera camera;
        camera.camera_id_ = SLAM_camera->GetId();
        if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_PINHOLE) {
            camera.setModelId(Camera::CameraModelType::PINHOLE);
            float SLAM_fx = SLAM_camera->getParameter(0);
            float SLAM_fy = SLAM_camera->getParameter(1);
            float SLAM_cx = SLAM_camera->getParameter(2);
            float SLAM_cy = SLAM_camera->getParameter(3);

            // Old K, i.e. K in SLAM
            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << SLAM_fx, 0.f, SLAM_cx,
                        0.f, SLAM_fy, SLAM_cy,
                        0.f, 0.f, 1.f
            );

            // camera.width_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.width
            //                                              : graphics_utils::roundToIntegerMultipleOf16(
            //                                                    undistort_params.old_size_.width);
            camera.width_ = undistort_params.old_size_.width;
            float x_ratio = static_cast<float>(camera.width_) / undistort_params.old_size_.width;

            // camera.height_ = this->sensor_type_ == STEREO ? undistort_params.old_size_.height
            //                                               : graphics_utils::roundToIntegerMultipleOf16(
            //                                                     undistort_params.old_size_.height);
            camera.height_ = undistort_params.old_size_.height;
            float y_ratio = static_cast<float>(camera.height_) / undistort_params.old_size_.height;

            camera.num_gaus_pyramid_sub_levels_ = num_gaus_pyramid_sub_levels_;
            camera.gaus_pyramid_width_.resize(num_gaus_pyramid_sub_levels_);
            camera.gaus_pyramid_height_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                camera.gaus_pyramid_width_[l] = camera.width_ * this->kf_gaus_pyramid_factors_[l];
                camera.gaus_pyramid_height_[l] = camera.height_ * this->kf_gaus_pyramid_factors_[l];
            }

            camera.params_[0]/*new fx*/= SLAM_fx * x_ratio;
            camera.params_[1]/*new fy*/= SLAM_fy * y_ratio;
            camera.params_[2]/*new cx*/= SLAM_cx * x_ratio;
            camera.params_[3]/*new cy*/= SLAM_cy * y_ratio;

            cv::Mat K_new = (
                cv::Mat_<float>(3, 3)
                    << camera.params_[0], 0.f, camera.params_[2],
                        0.f, camera.params_[1], camera.params_[3],
                        0.f, 0.f, 1.f
            );

            // Undistortion
            if (this->sensor_type_ == MONOCULARLINE || this->sensor_type_ == RGBDLINE)
                undistort_params.dist_coeff_.copyTo(camera.dist_coeff_);

            camera.initUndistortRectifyMapAndMask(K, SLAM_im_size, K_new, true);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_sub_undistort_mask;
            int viewer_image_height_ = camera.height_ * rendered_image_viewer_scale_;
            int viewer_image_width_ = camera.width_ * rendered_image_viewer_scale_;
            cv::resize(camera.undistort_mask, viewer_sub_undistort_mask,
                       cv::Size(viewer_image_width_, viewer_image_height_));
            viewer_sub_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_sub_undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

            if (this->sensor_type_ == STEREOLINE) {
                camera.stereo_bf_ = stereo_baseline_length_ * camera.params_[0];
                if (this->stereo_Q_.cols != 4) {
                    this->stereo_Q_ = cv::Mat(4, 4, CV_32FC1);
                    this->stereo_Q_.setTo(0.0f);
                    this->stereo_Q_.at<float>(0, 0) = 1.0f;
                    this->stereo_Q_.at<float>(0, 3) = -camera.params_[2];
                    this->stereo_Q_.at<float>(1, 1) = 1.0f;
                    this->stereo_Q_.at<float>(1, 3) = -camera.params_[3];
                    this->stereo_Q_.at<float>(2, 3) = camera.params_[0];
                    this->stereo_Q_.at<float>(3, 2) = 1.0f / stereo_baseline_length_;
                }
            }
        }
        else if (SLAM_camera->GetType() == ORB_SLAM3::GeometricCamera::CAM_FISHEYE) {
            camera.setModelId(Camera::CameraModelType::FISHEYE);
        }
        else {
            camera.setModelId(Camera::CameraModelType::INVALID);
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }
}

void GaussianMapperLine::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cerr << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
    std::unique_lock<std::mutex> lock(mutex_settings_);

    // Model parameters
    model_params_.sh_degree_ =
        settings_file["Model.sh_degree"].operator int();
    model_params_.resolution_ =
        settings_file["Model.resolution"].operator float();
    model_params_.white_background_ =
        (settings_file["Model.white_background"].operator int()) != 0;
    model_params_.eval_ =
        (settings_file["Model.eval"].operator int()) != 0;

    // Pipeline Parameters
    z_near_ =
        settings_file["Camera.z_near"].operator float();
    z_far_ =
        settings_file["Camera.z_far"].operator float();

    monocular_inactive_geo_densify_max_pixel_dist_ =
        settings_file["Monocular.inactive_geo_densify_max_pixel_dist"].operator float();
    stereo_min_disparity_ =
        settings_file["Stereo.min_disparity"].operator int();
    stereo_num_disparity_ =
        settings_file["Stereo.num_disparity"].operator int();
    RGBD_min_depth_ =
        settings_file["RGBD.min_depth"].operator float();
    RGBD_max_depth_ =
        settings_file["RGBD.max_depth"].operator float();

    inactive_geo_densify_ =
        (settings_file["Mapper.inactive_geo_densify"].operator int()) != 0;
    max_depth_cached_ =
        settings_file["Mapper.depth_cache"].operator int();
    min_num_initial_map_kfs_ = 
        static_cast<unsigned long>(settings_file["Mapper.min_num_initial_map_kfs"].operator int());
    new_keyframe_times_of_use_ = 
        settings_file["Mapper.new_keyframe_times_of_use"].operator int();
    local_BA_increased_times_of_use_ = 
        settings_file["Mapper.local_BA_increased_times_of_use"].operator int();
    loop_closure_increased_times_of_use_ = 
        settings_file["Mapper.loop_closure_increased_times_of_use_"].operator int();
    cull_keyframes_ =
        (settings_file["Mapper.cull_keyframes"].operator int()) != 0;
    large_rot_th_ =
        settings_file["Mapper.large_rotation_threshold"].operator float();
    large_trans_th_ =
        settings_file["Mapper.large_translation_threshold"].operator float();
    stable_num_iter_existence_ =
        settings_file["Mapper.stable_num_iter_existence"].operator int();

    pipe_params_.convert_SHs_ =
        (settings_file["Pipeline.convert_SHs"].operator int()) != 0;
    pipe_params_.compute_cov3D_ =
        (settings_file["Pipeline.compute_cov3D"].operator int()) != 0;

    do_gaus_pyramid_training_ =
        (settings_file["GausPyramid.do"].operator int()) != 0;
    num_gaus_pyramid_sub_levels_ =
        settings_file["GausPyramid.num_sub_levels"].operator int();
    int sub_level_times_of_use =
        settings_file["GausPyramid.sub_level_times_of_use"].operator int();
    kf_gaus_pyramid_times_of_use_.resize(num_gaus_pyramid_sub_levels_);
    kf_gaus_pyramid_factors_.resize(num_gaus_pyramid_sub_levels_);
    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
        kf_gaus_pyramid_times_of_use_[l] = sub_level_times_of_use;
        kf_gaus_pyramid_factors_[l] = std::pow(0.5f, num_gaus_pyramid_sub_levels_ - l);
    }

    keyframe_record_interval_ = 
        settings_file["Record.keyframe_record_interval"].operator int();
    all_keyframes_record_interval_ = 
        settings_file["Record.all_keyframes_record_interval"].operator int();
    record_rendered_image_ = 
        (settings_file["Record.record_rendered_image"].operator int()) != 0;
    record_ground_truth_image_ = 
        (settings_file["Record.record_ground_truth_image"].operator int()) != 0;
    record_loss_image_ = 
        (settings_file["Record.record_loss_image"].operator int()) != 0;
    training_report_interval_ = 
        settings_file["Record.training_report_interval"].operator int();
    record_loop_ply_ =
        (settings_file["Record.record_loop_ply"].operator int()) != 0;

    // Optimization Parameters
    opt_params_.iterations_ =
        settings_file["Optimization.max_num_iterations"].operator int();
    opt_params_.position_lr_init_ =
        settings_file["Optimization.position_lr_init"].operator float();
    opt_params_.position_lr_final_ =
        settings_file["Optimization.position_lr_final"].operator float();
    opt_params_.position_lr_delay_mult_ =
        settings_file["Optimization.position_lr_delay_mult"].operator float();
    opt_params_.position_lr_max_steps_ =
        settings_file["Optimization.position_lr_max_steps"].operator int();
    opt_params_.feature_lr_ =
        settings_file["Optimization.feature_lr"].operator float();
    opt_params_.opacity_lr_ =
        settings_file["Optimization.opacity_lr"].operator float();
    opt_params_.scaling_lr_ =
        settings_file["Optimization.scaling_lr"].operator float();
    opt_params_.rotation_lr_ =
        settings_file["Optimization.rotation_lr"].operator float();

    opt_params_.percent_dense_ =
        settings_file["Optimization.percent_dense"].operator float();
    opt_params_.lambda_dssim_ =
        settings_file["Optimization.lambda_dssim"].operator float();
    opt_params_.densification_interval_ =
        settings_file["Optimization.densification_interval"].operator int();
    opt_params_.opacity_reset_interval_ =
        settings_file["Optimization.opacity_reset_interval"].operator int();
    opt_params_.densify_from_iter_ =
        settings_file["Optimization.densify_from_iter_"].operator int();
    opt_params_.densify_until_iter_ =
        settings_file["Optimization.densify_until_iter"].operator int();
    opt_params_.densify_grad_threshold_ =
        settings_file["Optimization.densify_grad_threshold"].operator float();
    //Line Weight Setting
    opt_params_.weight_line_coherence_ =
        settings_file["Optimization.weight_line_coherence"].operator float();
    opt_params_.weight_line_shape_ori_ =
        settings_file["Optimization.weight_line_shape_ori"].operator float();
    opt_params_.weight_line_shape_ecc_ =
        settings_file["Optimization.weight_line_shape_ecc"].operator float();
    //added by zdg
    opt_params_.voxel_size_ = settings_file["Optimization.voxel_size"].isNone() ? 0.25f : settings_file["Optimization.voxel_size"].operator float();
    opt_params_.line_sample_step_ = settings_file["Optimization.line_sample_step"].isNone() ? 0.2f : settings_file["Optimization.line_sample_step"].operator float();
    opt_params_.line_view_weight_ = settings_file["Optimization.line_view_weight"].isNone() ? 2.0f : settings_file["Optimization.line_view_weight"].operator float();
    opt_params_.line_sigma_ = settings_file["Optimization.line_sigma"].isNone() ? 3.0f : settings_file["Optimization.line_sigma"].operator float();
    opt_params_.line_top_k_ = settings_file["Optimization.line_top_k"].isNone() ? 3 : settings_file["Optimization.line_top_k"].operator int();
    opt_params_.refined_gaussian_max_iter_num_ = settings_file["Optimization.refined_gaussian_max_iter_num"].isNone() ? 10000 : settings_file["Optimization.refined_gaussian_max_iter_num"].operator int();

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();

    // 建议在 Optimization Parameters 块中添加
    opt_params_.max_anisotropy_threshold_ = 
        settings_file["Optimization.max_anisotropy_threshold"].isNone() ? 600.0f : settings_file["Optimization.max_anisotropy_threshold"].operator float();
    opt_params_.lambda_anisotropy_ = 
        settings_file["Optimization.lambda_anisotropy"].isNone() ? 0.01f : settings_file["Optimization.lambda_anisotropy"].operator float();
    opt_params_.absolute_max_ratio_ = 
        settings_file["Optimization.absolute_max_ratio"].isNone() ? 1500.0f : settings_file["Optimization.absolute_max_ratio"].operator float();

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
}

void GaussianMapperLine::run()
{
    const auto mapper_start = std::chrono::steady_clock::now();
    // Initialize append/budget configuration BEFORE natural training so the
    // cumulative training-time boundary is active from the first real update.
    appendRefinementEnabled();
    // First loop: Initial gaussian mapping
    std::cerr << "[DEBUG] Thread: GaussianMapperLine thread started." << std::endl;
    while (!isStopped()) {
        
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions()) {
            //std::cerr << "[DEBUG] Checkpoint 1: Initial conditions met. Starting data collection..." << std::endl;
            pSLAM_->getAtlas()->clearMappingOperation();

            // Get initial sparse map
            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector<ORB_SLAM3::MapPoint*> vpMPs;
            std::vector<ORB_SLAM3::MapLine*> vpMPLs;

            // 快照阶段：只拷贝后续去畸变/建金字塔所需的最小数据（图像、位姿、
            // 相机 ID、关键点），把昂贵的去畸变与 Tensor 上传移到锁外，
            // 缩短 mMutexMapUpdate 持有时间（避免阻塞 LocalMapping 线程）。
            struct KfSnapshot {
                unsigned long fid;
                unsigned long camera_id;
                Sophus::SE3f pose;
                cv::Mat img_left;
                cv::Mat img_aux;
                std::string filename;
                std::vector<float> pixels;
                std::vector<float> points_local;
            };
            std::vector<KfSnapshot> kf_snapshots;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vpKFs = pMap->GetAllKeyFrames();
                vpMPs = pMap->GetAllMapPoints();
                vpMPLs = pMap->GetAllMapLines();
                for (const auto& pMP : vpMPs){
                    if (!pMP || pMP->isBad()) continue;
                    // 🌟 新增过滤逻辑 A：观测帧数过滤
                    // 刚初始化的地图点如果只被 2 帧看到，由于视差小，3D 位置极度不可靠
                    if (pMP->Observations() < 3) continue;

                    Point3D point3D;
                    auto pos = pMP->GetWorldPos();
                    point3D.xyz_(0) = pos.x();
                    point3D.xyz_(1) = pos.y();
                    point3D.xyz_(2) = pos.z();
                    auto color = pMP->GetColorRGB();
                    point3D.color_(0) = color(0);
                    point3D.color_(1) = color(1);
                    point3D.color_(2) = color(2);
                    // Deformation anchor: the MapPoint's reference keyframe.
                    ORB_SLAM3::KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();
                    point3D.ref_kf_id_ =
                        pRefKF ? static_cast<long long>(pRefKF->mnId) : -1LL;
                    scene_->cachePoint3D(pMP->mnId, point3D);
                }
                //Sample Point 3d from vpMPLs
                // for(size_t i = 0; i < vpMPLs.size(); ++i)
                // {
                //     float sample_step = 0.1f;        // 世界坐标采样步长 (e.g. 0.05f)
                //     float view_angle_power = 2.0f;   // 视角权重指数 (e.g. 2.0)
                //     float sigma_line_pixel = 3.0f;   // 图像线一致性 σ (e.g. 3.0 px)
                //     int   top_k = 3;               // Top-K 视角 (e.g. 3)
                //     vpMPLs[i]->SamplePointsAlongLine_MultiViewWeighted_Advanced(sample_step, view_angle_power, sigma_line_pixel, top_k);    //调用这个函数是否出现bug之类的。
                //     //std::vector<Eigen::Vector3f> sampled_line_pnts = 
                // }
                for (const auto& pML : vpMPLs)
                {
                    if(!pML) continue;
                    // 1. 获取线段端点并计算方向向量 (Crucial Step!)
                    auto endpoints = pML->GetLineWorldPos();
                    Eigen::Vector3f p1(endpoints.first[0], endpoints.first[1], endpoints.first[2]);
                    Eigen::Vector3f p2(endpoints.second[0], endpoints.second[1], endpoints.second[2]);
                    
                    Eigen::Vector3f line_vec = p2 - p1;
                    float line_len = line_vec.norm();
                    Eigen::Vector3f line_dir = line_vec.normalized(); // 归一化方向

                    // 2. 缓存 Line3D (用于可视化或调试)
                    Line3D line3D;
                    //line3D.start_ = scene_->getPoint3D(pML->GetStartPointId());
                    //line3D.end_ = scene_->getPoint3D(pML->GetEndPointId());
                    //line3D.color_ = pML->GetColorRGB();
                    line3D.p1_[0] = endpoints.first[0];
                    line3D.p1_[1] = endpoints.first[1];
                    line3D.p1_[2] = endpoints.first[2];
                    line3D.p2_[0] = endpoints.second[0];
                    line3D.p2_[1] = endpoints.second[1];
                    line3D.p2_[2] = endpoints.second[2];
                    auto line_colors = pML->GetLineColorRGB(); // 缓存 getter，避免 6 次重复调用
                    line3D.color1_[0] = line_colors.first[0];
                    line3D.color1_[1] = line_colors.first[1];
                    line3D.color1_[2] = line_colors.first[2];
                    line3D.color2_[0] = line_colors.second[0];
                    line3D.color2_[1] = line_colors.second[1];
                    line3D.color2_[2] = line_colors.second[2];
                    scene_->cacheLine3D(pML->mnId, line3D);

                    // 3. 执行采样 (Sample Points)
                    // 建议将这些参数放入配置文件 readConfigFromFile 中
                    float sample_step = opt_params_.line_sample_step_; 
                    float view_angle_power = opt_params_.line_view_weight_;
                    float sigma_line_pixel = opt_params_.line_sigma_;
                    int top_k = opt_params_.line_top_k_;

                    // 这一步会填充 mapLine 内部的 buffer
                    pML->SamplePointsAlongLine_MultiViewWeighted_Advanced(sample_step, view_angle_power, sigma_line_pixel, top_k);
                    
                    // 在MapLine类中采样3D线段并存储到Point3D中。需要再写一下MapLine类的函数
                    const std::vector<Eigen::Vector3f> sampledPoints3D = pML->GetLineSampledPoints3D();
                    const std::vector<cv::Vec3b> sampledColors = pML->GetLineSampledPntsColors();
                    // 4. 将采样点转为 Gaussian 初始点 (Point3D)
                    for (size_t j = 0; j < sampledPoints3D.size(); ++j)
                    {
                        Point3D point3D;
                        point3D.xyz_[0] = sampledPoints3D[j][0];
                        point3D.xyz_[1] = sampledPoints3D[j][1];
                        point3D.xyz_[2] = sampledPoints3D[j][2];
                        point3D.color_(0) = (float)(sampledColors[j][0]/255.0);
                        point3D.color_(1) = (float)(sampledColors[j][1]/255.0);
                        point3D.color_(2) = (float)(sampledColors[j][2]/255.0);

                        // 【核心修正】必须设置方向和采样步长等元数据
                        point3D.line_dir_ = line_dir;       // 传入方向，用于初始化 Rotation
                        point3D.sample_step_ = sample_step; // 传入步长，用于初始化 Scale_parallel
                        // Deformation anchor of the source MapLine.
                        ORB_SLAM3::KeyFrame* pRefKF = pML->GetReferenceKeyFrame();
                        point3D.ref_kf_id_ =
                            pRefKF ? static_cast<long long>(pRefKF->mnId) : -1LL;
                        // 如果 MapLine 能提供 ref_depth 或 ref_focal 更好，否则使用默认值
                        // point3D.ref_depth_z_ = ...;
                        // point3D.ref_focal_ = ...;

                        scene_->addPointToVoxel(sampledPoints3D[j], opt_params_.voxel_size_); // 先把点加入体素网格，后续优化时会更新位置
                        ///scene_->cachePoint3D(pML->mnId, point3D);
                        scene_->cacheLineSampledPnts3D(pML->mnId, point3D);
                    }
                }

                std::cerr << "[Gaussian Mapper] Voxel Grid Initialized. Total occupied cells: " 
                            << scene_->getVoxelCount() << std::endl;

                //debug by zdg
                //std::cerr <<"===============================================================" << std::endl;
                //std::cerr << "result_dir_: " << result_dir_ << std::endl;
                //saveDebugMapToObj("debug_initial_map.obj", vpMPs, vpMPLs);
                // ================= [DEBUG CALL] =================
                // 此时 scene_->cached_point_cloud_ 已经填满了点和采样点
                // scene_->cached_line3D_cloud_ 已经填满了线段
                //scene_->saveDebugSceneToObj("debug_scene_init.obj");
                // ================================================
                //std::cerr <<"===============================================================" << std::endl;
                //end debug

                kf_snapshots.reserve(vpKFs.size());
                for (const auto& pKF : vpKFs) {
                    KfSnapshot s;
                    s.fid = pKF->mnId;
                    s.camera_id = pKF->mpCamera->GetId();
                    s.pose = pKF->GetPose();
                    s.img_left = pKF->imgLeftRGB.clone();
                    s.img_aux = pKF->imgAuxiliary.clone();
                    s.filename = pKF->mNameFile;
                    pKF->GetKeypointInfo(s.pixels, s.points_local);
                    kf_snapshots.push_back(std::move(s));
                }
            }

            // 锁外处理：创建 GaussianKeyframe、去畸变、上传 Tensor（与原逻辑一致）
            for (auto& s : kf_snapshots) {
                std::shared_ptr<GaussianKeyframeLine> new_kf = std::make_shared<GaussianKeyframeLine>(s.fid, getIteration());
                new_kf->zfar_ = z_far_;
                new_kf->znear_ = z_near_;
                new_kf->setPose(
                    s.pose.unit_quaternion().cast<double>(),
                    s.pose.translation().cast<double>());
                cv::Mat imgRGB_undistorted, imgAux_undistorted;
                try {
                    Camera& camera = scene_->cameras_.at(s.camera_id);
                    new_kf->setCameraParams(camera);

                    if (this->sensor_type_ == STEREOLINE)
                        imgRGB_undistorted = s.img_left;
                    else
                        camera.undistortImage(s.img_left, imgRGB_undistorted);
                    if (this->sensor_type_ == RGBDLINE)
                        camera.undistortImage(s.img_aux, imgAux_undistorted);
                    else
                        imgAux_undistorted = s.img_aux;

                    new_kf->original_image_ =
                        tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                    new_kf->img_filename_ = s.filename;
                    new_kf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
                    new_kf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
                    new_kf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                }
                catch (std::out_of_range) {
                    throw std::runtime_error("[GaussianMapper::run]KeyFrame Camera not found!");
                }
                new_kf->computeTransformTensors();
                scene_->addKeyframe(new_kf, &kfid_shuffled_);

                increaseKeyframeTimesOfUse(new_kf, newKeyframeTimesOfUse());

                new_kf->kps_pixel_ = std::move(s.pixels);
                new_kf->kps_point_local_ = std::move(s.points_local);
                new_kf->img_undist_ = imgRGB_undistorted;
                new_kf->img_auxiliary_undist_ = imgAux_undistorted;
            }

            // Prepare multi resolution images for training
            for (auto& kfit : scene_->keyframes()) {
                auto pkf = kfit.second;
                if (device_type_ == torch::kCUDA) {
                    cv::cuda::GpuMat img_gpu;
                    img_gpu.upload(pkf->img_undist_);
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::cuda::GpuMat img_resized;
                        cv::cuda::resize(img_gpu, img_resized,
                                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
                    }
                }
                else {
                    pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
                    for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                        cv::Mat img_resized;
                        cv::resize(pkf->img_undist_, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                        pkf->gaus_pyramid_original_image_[l] =
                            tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
                    }
                }
            }

            // Prepare for training
            {
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
                gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
                std::unique_lock<std::mutex> lock(mutex_settings_);
                gaussians_->trainingSetup(opt_params_);
            }

            // Invoke training once
            //trainForOneIteration();

            std::cerr << "[DEBUG] Checkpoint 10: Setup done. Starting first training iteration..." << std::endl;

            try {
                // Invoke training once
                trainForOneIteration();
            } 
            catch (const c10::Error& e) {
                std::cerr << "Torch Error: " << e.msg() << std::endl;
                // 可以在这里保存进度，或者做清理工作
                return;
            }
            catch (const std::exception& e) {
                std::cerr << "Standard Error: " << e.what() << std::endl;
            }
            catch (...) {
                std::cerr << "Unknown Error occurred." << std::endl;
            }

            // Finish initial mapping loop
            initial_mapped_ = true;
            break;
        }
        else if (pSLAM_->isShutDown()) {
            std::cerr << "[DEBUG] SLAM Shutdown detected during wait." << std::endl;
            break;
        }
        else {
            // Initial conditions not satisfied
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    // Second loop: Incremental gaussian mapping
    int SLAM_stop_iter = 0;
    while (!isStopped()) {
        //std::cerr <<"============================start run GaussianMapperLine: LOCAL Mapping  ===================================" << std::endl;
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations_withLine();
            if (cull_keyframes_)
                cullKeyframes();
        }

        // Invoke training once
        //trainForOneIteration();

        try {
                //高斯优化代码 Invoke training once
                trainForOneIteration();
            } 
            catch (const c10::Error& e) {
                std::cerr << "Torch Error second loop: " << e.msg() << std::endl;
                // 可以在这里保存进度，或者做清理工作
                return;
            }
            catch (const std::exception& e) {
                std::cerr << "Standard Error second loop: " << e.what() << std::endl;
            }
            catch (...) {
                std::cerr << "Unknown Error occurred second loop." << std::endl;
            }

        if (pSLAM_->isShutDown()) {
            if (phase_slam_end_s_ == 0.0)
                phase_slam_end_s_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - mapper_start).count() / 1000.0;

            // 【修改点 1】循环清空所有积压的操作，确保最后一批 Line Gaussians 全部入场
            const auto drain_start = std::chrono::steady_clock::now();
            // geom-sync-v1: one-shot CPU contract self-test (diagnostic only).
    {
        static const bool contract_test =
            (std::getenv("PHOTO_SLAM_SYNC_CONTRACT_TEST") != nullptr);
        static bool contract_done = false;
        if (contract_test && !contract_done) {
            contract_done = true;
            runRigidCorrectionContractTest();
        }
    }
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
                combineMappingOperations_withLine();
            }
            phase_drain_ms_ += std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - drain_start).count();

            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_) {
            phase_mapper_end_s_ = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - mapper_start).count() / 1000.0;
            break;
        }
    }

    // // Third loop: Tail gaussian optimization
    // int densify_interval = densifyInterval();
    // int n_delay_iters = densify_interval * 3; // Delay more iterations to ensure stability after SLAM stops
    // while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) 
    // {
    //     //trainForOneIteration();
    //     try {
    //             //高斯优化代码 Invoke training once
    //             trainForOneIteration();
    //         } 
    //         catch (const c10::Error& e) {
    //             std::cerr << "Torch Error third loop: " << e.msg() << std::endl;
    //             // 可以在这里保存进度，或者做清理工作
    //             return;
    //         }
    //         catch (const std::exception& e) {
    //             std::cerr << "Standard Error third loop: " << e.what() << std::endl;
    //         }
    //         catch (...) {
    //             std::cerr << "Unknown Error occurred third loop." << std::endl;
    //         }
    //     densify_interval = densifyInterval();
    //     //n_delay_iters = densify_interval * 0.8;
    //     n_delay_iters = densify_interval * 3;
    // }

    // Third loop: Tail gaussian optimization (Enhanced for Photo-SLAM-L)
    //std::cerr << "\n[Gaussian Mapper] SLAM tracking ended! Starting final global refinement..." << std::endl;
// added by zdg
    // 当增量过程结束时，强制清空一次 SLAM 操作队列，确保最后的数据被加入
    //combineMappingOperations_withLine();

    // ------------------------------------------------------------------
    // Frozen-geometry mapper ablation: the SLAM-produced scene is now final
    // and complete. Freeze it once and replay every mapper condition on that
    // identical input instead of running a single global refinement.
    // ------------------------------------------------------------------
    if (frozenAblationEnabled()) {
        runFrozenAblation();
        signalStop();
        return;
    }

    // ====================================================================
    // 🌟 核心操作：重置所有关键帧的使用次数
    // ====================================================================
    std::cerr << "[Gaussian Mapper] SLAM ended. Resetting remaining_times_of_use for global refinement..." << std::endl;
    {
        // 建议加上互斥锁，防止此时还有其他线程访问 keyframes (虽然此时 SLAM 已关)
        // std::unique_lock<std::mutex> lock(mutex_keyframes_); 
        for (auto& kfit : scene_->keyframes()) {
            // 将所有帧的剩余使用次数设为一个很大的数（如 1000）
            // 这样 useOneRandomSlidingWindowKeyframe() 就会像随机采样一样公平对待每一帧
            kfit.second->remaining_times_of_use_ = 1000; 
        }
    }
//end by zdg

    // 【核心修改】强制增加全局优化次数，确保最后几帧完美收敛
    // 5000 次是一个非常安全的数值，大约需要多花几秒钟，但能挽救整个地图的边缘画质
    int final_iters = opt_params_.refined_gaussian_max_iter_num_; 
    int target_stop_iter = getIteration() + final_iters;

    prof_refine_phase_ = true;
    const auto refine_start = std::chrono::steady_clock::now();

    while (getIteration() < target_stop_iter || isKeepingTraining()) {
        // Guard: when SLAM produced no keyframes (short/sparse run), the scene is
        // empty and getIteration() can never advance -> the loop would spin
        // forever. Stop refinement in that case.
        if (scene_->keyframes().empty()) {
            std::cerr << "[Gaussian Mapper] No keyframes in scene; stopping final refinement." << std::endl;
            break;
        }
        try {
            // 高斯优化代码 Invoke training once
            //trainForOneIteration();
            trainForOneIterationErrorGuided();  //<--- 这里调用了新的训练函数

            // // 🌟 在精修到一半（比如 2500 次）时，执行一次终极清理
            // if (getIteration() == (target_stop_iter - 2500)) {
            //     std::cerr << "[Gaussian Mapper] Applying Ultimate Cleanup..." << std::endl;
            //     // 1. 重置透明度：让所有点重新竞争
            //     gaussians_->resetOpacity(); 
            //     // 2. 杀掉所有透明度低于 0.1 的点（非常暴力但有效）
            //     auto final_prune_mask = (gaussians_->getOpacityActivation() < 0.1f).squeeze();
            //     gaussians_->prunePointsWithLineAwareness(final_prune_mask);
            // }
            
            // 进度汇报，让您在终端里看到它在努力优化
            if (getIteration() % 200 == 0) {
                std::cerr << "[Gaussian Mapper] Final refinement iteration: " 
                          << getIteration() - (target_stop_iter - final_iters) 
                          << " / " << final_iters << std::endl;
            }
        } 
        catch (const c10::Error& e) {
            std::cerr << "Torch Error third loop: " << e.msg() << std::endl;
            return;
        }
        catch (const std::exception& e) {
            std::cerr << "Standard Error third loop: " << e.what() << std::endl;
        }
        catch (...) {
            std::cerr << "Unknown Error occurred third loop." << std::endl;
        }
    }
    std::cerr << "[Gaussian Mapper] Final refinement completed perfectly! Saving models..." << std::endl;
    phase_refine_s_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - refine_start).count() / 1000.0;

    // Append refinement: real extra optimizer updates on the EXISTING line
    // model. Inserted AFTER the original final refinement and BEFORE shutdown
    // export/signalStop(); the natural training and fixed refinement rules are
    // left untouched.
    if (appendRefinementEnabled()) {
        runAppendRefinement();
    }

    // Save and clear
    const auto export_start = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty()) {
        std::cerr << "[Gaussian Mapper] No keyframes in scene; skipping shutdown render/PLY export." << std::endl;
    } else {
        renderAndRecordAllKeyframesReadOnly("_shutdown");
        savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
        writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
    }
    shutdown_export_seconds_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - export_start).count() / 1000.0;
    if (!append_dir_.empty()) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(6) << shutdown_export_seconds_;
        append_refinement::writeKv(append_dir_ / "append_summary.txt",
                                   "shutdown_export_seconds", os.str());
        append_refinement::writeKv(append_dir_ / "append_export_timing.txt",
                                   "shutdown_export_seconds", os.str());
        append_refinement::appendLine(append_dir_ / "append_status.txt",
                                      "shutdown_export_seconds=" + os.str());
    }

    // ==========================================================
    // 🌟 [新增代码]：输出最终统计时间
    // ==========================================================
    double total_time_seconds = total_training_time_ms_ / 1000.0;
    double total_time_minutes = total_time_seconds / 60.0;
    
    std::cerr << "=====================================================" << std::endl;
    std::cerr << "[Gaussian Mapper] Total Mapping Time: " << total_time_seconds << " s (" 
              << total_time_minutes << " mins)" << std::endl;
    std::cerr << "[Gaussian Mapper] Total Iterations: " << getIteration() << std::endl;
    // Quad comparison telemetry: identical fields to the point build.
    long long quad_line_gaussians = 0;
    if (gaussians_ && gaussians_->is_line_.defined() && gaussians_->is_line_.numel() > 0)
        quad_line_gaussians = gaussians_->is_line_.to(torch::kCPU).sum().item<int64_t>();
    std::cerr << "[QuadMetrics] optimizer_steps=" << frozen_opt_steps_
              << " gaussians=" << (gaussians_ ? gaussians_->xyz_.size(0) : 0)
              << " line_gaussians=" << quad_line_gaussians
              << " kfs=" << scene_->keyframes().size()
              << " mapping_seconds=" << total_time_seconds << std::endl;
    {
        std::ofstream quad_csv(result_dir_ / "quad_metrics.txt");
        quad_csv << "optimizer_steps=" << frozen_opt_steps_ << "\n"
                 << "gaussians=" << (gaussians_ ? gaussians_->xyz_.size(0) : 0) << "\n"
                 << "line_gaussians=" << quad_line_gaussians << "\n"
                 << "kfs=" << scene_->keyframes().size() << "\n"
                 << "mapping_seconds=" << total_time_seconds << "\n";
    }
    std::cerr << "=====================================================" << std::endl;

    // (可选) 将时间写入 txt 文件，方便后续跑批量实验时写 Python 脚本自动提取
    std::ofstream time_log(result_dir_ / "total_mapping_time.txt");
    if (time_log.is_open()) {
        time_log << "Total_Time_Seconds: " << total_time_seconds << "\n";
        time_log << "Total_Time_Minutes: " << total_time_minutes << "\n";
        time_log << "Total_Iterations: " << getIteration() << "\n";
        time_log.close();
    }

    g_gpu_prof.print();
    printDensifyProfiler();
    std::cerr << "[PhaseTiming] SLAM_end=" << phase_slam_end_s_ << "s"
              << " mapper_end=" << phase_mapper_end_s_ << "s"
              << " queue_drain=" << phase_drain_ms_ << "ms"
              << " refinement=" << phase_refine_s_ << "s\n";

    signalStop();
}

namespace {
// Mean of the value column of a "keyframe_id value" metrics file written by
// renderAndRecordAllKeyframes() (comment lines start with '#'). Returns -1 if
// the file is missing or empty.
double frozenAblMeanValueColumn(const std::filesystem::path& path)
{
    std::ifstream in(path);
    if (!in.is_open())
        return -1.0;
    std::string line;
    double sum = 0.0;
    long n = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        const auto sp = line.find(' ');
        if (sp == std::string::npos)
            continue;
        const auto b = line.find_first_not_of(" \t", sp);
        if (b == std::string::npos)
            continue;
        try {
            sum += std::stod(line.substr(b));
            ++n;
        }
        catch (const std::exception&) {
            continue;
        }
    }
    return n > 0 ? sum / static_cast<double>(n) : -1.0;
}

// Fixed-precision float formatting for deterministic snapshot text.
std::string frozenAblF(double v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.9g", v);
    return std::string(buf);
}

// FNV-1a 64-bit over a byte string, rendered as hex.
std::string frozenAblHashHex(const std::string& s)
{
    std::uint64_t h = 1469598103934665603ULL;
    for (unsigned char ch : s) {
        h ^= static_cast<std::uint64_t>(ch);
        h *= 1099511628211ULL;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return std::string(buf);
}

// Serialise the frozen scene (poses, camera ids, image identity, sparse point
// cloud incl. line-sampled points and their directions) into deterministic
// text. This text IS the frozen input snapshot and its hash is the input hash.
std::string frozenAblSerializeScene(
    GaussianSceneLine& scene,
    const std::map<point3D_id_t, Point3D>& pcd,
    float extent,
    std::size_t n_pt,
    std::size_t n_ln)
{
    std::string s;
    s.reserve(1 << 20);
    s += "# photo-slam-l frozen geometry snapshot v1\n";
    s += "# kfs=" + std::to_string(scene.keyframes().size())
       + " map_points=" + std::to_string(n_pt)
       + " line_samples=" + std::to_string(n_ln)
       + " cameras_extent=" + frozenAblF(extent) + "\n";
    for (auto& kv : scene.keyframes()) {
        auto& pkf = kv.second;
        const Sophus::SE3f T = pkf->getPosef();
        const Eigen::Quaternionf q = T.unit_quaternion();
        const Eigen::Vector3f t = T.translation();
        s += "KF " + std::to_string(pkf->fid_) + ' ' + std::to_string(pkf->camera_id_);
        s += ' ' + frozenAblF(q.w()) + ' ' + frozenAblF(q.x()) + ' ' + frozenAblF(q.y()) + ' ' + frozenAblF(q.z());
        s += ' ' + frozenAblF(t.x()) + ' ' + frozenAblF(t.y()) + ' ' + frozenAblF(t.z());
        s += ' ' + std::to_string(pkf->image_width_) + 'x' + std::to_string(pkf->image_height_);
        s += " | " + pkf->img_filename_ + "\n";
    }
    for (const auto& kv : pcd) {
        const Point3D& p = kv.second;
        s += "P " + std::to_string(kv.first);
        s += ' ' + frozenAblF(p.xyz_[0]) + ' ' + frozenAblF(p.xyz_[1]) + ' ' + frozenAblF(p.xyz_[2]);
        s += ' ' + frozenAblF(p.color_[0]) + ' ' + frozenAblF(p.color_[1]) + ' ' + frozenAblF(p.color_[2]);
        s += ' ' + std::to_string(p.source_ == PointSourceType::LINE_SAMPLED ? 1 : 0);
        s += ' ' + frozenAblF(p.line_dir_[0]) + ' ' + frozenAblF(p.line_dir_[1]) + ' ' + frozenAblF(p.line_dir_[2]);
        s += ' ' + frozenAblF(p.sample_step_) + "\n";
    }
    return s;
}

// Identity/hash of the image set actually consumed (path + byte size per KF).
std::string frozenAblImageKey(GaussianSceneLine& scene)
{
    std::string s;
    for (auto& kv : scene.keyframes()) {
        const std::string& f = kv.second->img_filename_;
        std::error_code ec;
        const auto sz = std::filesystem::file_size(f, ec);
        s += std::to_string(kv.second->fid_) + ' ' + f + ' '
           + (ec ? std::string("?") : std::to_string(static_cast<unsigned long long>(sz))) + '\n';
    }
    return s;
}

std::string frozenAblJoinSeq(const std::vector<std::size_t>& seq)
{
    std::string s;
    s.reserve(seq.size() * 4 + 16);
    for (std::size_t v : seq) {
        s += std::to_string(v);
        s += ',';
    }
    return s;
}
} // namespace

bool GaussianMapperLine::frozenAblationEnabled() const
{
    const char* e = std::getenv("PHOTO_SLAM_FROZEN_ABLATION");
    return e != nullptr && e[0] != '\0';
}

// Pre-generate the keyframe sampling sequence ONCE per frozen scene. Every
// condition and every repeat replays this exact sequence, so the viewpoint
// order is not a source of difference between them (it is, by construction, the
// same 10000-step sequence for the whole scene x sensor grid).
void GaussianMapperLine::preGenerateFrozenKfSequence(int n_steps)
{
    frozen_kf_seq_.clear();
    if (n_steps <= 0 || scene_->keyframes().empty())
        return;

    auto resetUsage = [&]() {
        for (auto& kfit : scene_->keyframes())
            kfit.second->remaining_times_of_use_ = 1000;
        kfs_used_times_.clear();
        kfid_shuffle_idx_ = 0;
    };

    int seq_seed = 12345;
    if (const char* e = std::getenv("PHOTO_SLAM_FROZEN_SEQ_SEED"))
        seq_seed = std::atoi(e);

    resetUsage();
    std::srand(static_cast<unsigned>(seq_seed));
    frozen_kf_seq_.reserve(static_cast<std::size_t>(n_steps));
    // Evaluation views must never be sampled for training, otherwise the
    // reported quality would be partly training quality.
    auto isEvalView = [&](std::size_t idx) {
        return std::find(frozen_eval_kf_seq_.begin(), frozen_eval_kf_seq_.end(), idx)
               != frozen_eval_kf_seq_.end();
    };
    for (int i = 0; i < n_steps; ++i) {
        std::shared_ptr<GaussianKeyframeLine> sel;
        std::size_t idx = 0;
        bool found = false;
        for (int attempt = 0; attempt < 64; ++attempt) {
            sel = useOneTrainingErrorSlidingWindowKeyframe();
            if (!sel)
                break;
            idx = 0;
            for (auto it = scene_->keyframes().begin(); it != scene_->keyframes().end(); ++it, ++idx) {
                if (it->second == sel)
                    break;
            }
            if (!isEvalView(idx)) {
                found = true;
                break;
            }
        }
        if (!sel || !found)
            break;
        frozen_kf_seq_.push_back(idx);
    }
    resetUsage();
    frozen_kf_step_ = 0;
    frozen_kf_replay_ = false;
    std::cerr << "[FrozenABL] pre-generated kf sampling sequence: "
              << frozen_kf_seq_.size() << " steps (seed " << seq_seed << ")" << std::endl;
}

// ---------------------------------------------------------------------------
// Fixed-geometry convergence experiment helpers.
// ---------------------------------------------------------------------------
namespace {
// Persist a [3,H,W] float image in [0,1] as a PNG so metrics (LPIPS in
// particular) can be recomputed offline on exactly the same views.
void fgmSaveImage(const torch::Tensor& img, const std::filesystem::path& path)
{
    auto t = img.detach().to(torch::kCPU).clamp(0.0, 1.0).mul(255.0)
                 .to(torch::kUInt8).contiguous();
    if (t.dim() != 3 || t.size(0) != 3)
        return;
    const int H = static_cast<int>(t.size(1));
    const int W = static_cast<int>(t.size(2));
    std::vector<cv::Mat> ch(3);
    for (int c = 0; c < 3; ++c)
        ch[c] = cv::Mat(H, W, CV_8UC1, t[c].data_ptr<uint8_t>());
    cv::Mat merged;
    cv::merge(ch, merged);
    cv::cvtColor(merged, merged, cv::COLOR_RGB2BGR);
    cv::imwrite(path.string(), merged);
}
} // namespace

// Effective structural-loss implementation selector (see header).
bool GaussianMapperLine::useRefStructLoss() const
{
    if (frozen_use_ref_ == 1) return true;
    if (frozen_use_ref_ == 0) return false;
    static const bool env_ref = []() {
        const char* e = std::getenv("PHOTO_SLAM_FGM_USE_REF");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return env_ref;
}

// Hard guard: with the count frozen, ANY change to the Gaussian total or to the
// number of line Gaussians is a bug and must be reported, not silently averaged
// into the metrics.
void GaussianMapperLine::assertFrozenCounts(const char* where)
{
    if (!frozen_fixed_count_ || !gaussians_ || !gaussians_->xyz_.defined())
        return;
    const long long n_all = static_cast<long long>(gaussians_->xyz_.size(0));
    const long long n_line = (gaussians_->is_line_.defined() &&
                              gaussians_->is_line_.numel() > 0)
                                 ? gaussians_->is_line_.to(torch::kCPU).sum().item().toLong()
                                 : 0LL;
    if ((frozen_count_ref_all_ >= 0 && n_all != frozen_count_ref_all_) ||
        (frozen_count_ref_line_ >= 0 && n_line != frozen_count_ref_line_)) {
        std::cerr << "[FGM][COUNT-VIOLATION] " << where
                  << ": total=" << n_all << " (ref " << frozen_count_ref_all_ << ")"
                  << " line=" << n_line << " (ref " << frozen_count_ref_line_ << ")"
                  << std::endl;
        frozen_count_violation_ = 1;
    }
}

// ---------------------------------------------------------------------------
// Mechanism diagnostics (see §六 of the task). All reads; the gradient probe is
// env-gated and restores the gradient buffers so training is unaffected.
// ---------------------------------------------------------------------------
void GaussianMapperLine::diagAtFrozenStep(long long opt_steps, const std::string& tag)
{
    if (!gaussians_ || frozen_eval_kf_seq_.empty() || !gaussians_->xyz_.defined())
        return;
    if (!gaussians_->is_line_.defined() || gaussians_->is_line_.numel() == 0)
        return;

    const bool want_grad = (std::getenv("PHOTO_SLAM_FGM_GRAD_DIAG") != nullptr);
    const auto mask = gaussians_->is_line_;

    auto quant5 = [](const torch::Tensor& t, std::vector<double>& out) {
        auto x = t.detach().to(torch::kCPU).flatten();
        if (x.numel() == 0) {
            for (int i = 0; i < 7; ++i) out.push_back(std::nan(""));
            return;
        }
        auto q = torch::quantile(x, torch::tensor({0.0, 0.1, 0.5, 0.9, 1.0}));
        for (int i = 0; i < 5; ++i) out.push_back(q[i].item<double>());
        out.push_back(x.mean().item<double>());
        out.push_back(x.std().item<double>());
    };

    std::vector<double> v;
    {
        torch::NoGradGuard ng;
        // --- actual 3-axis scales and aspect ratio of the line Gaussians ---
        auto s = torch::exp(gaussians_->scaling_).index({mask});   // [Nl,3]
        auto s_par = s.index({torch::indexing::Slice(), 0});
        auto s_perp = s.index({torch::indexing::Slice(),
                               torch::indexing::Slice(1, 3)}).mean(1);
        quant5(s_par, v);                                   // 7
        quant5(s_perp, v);                                  // 7
        quant5(s_par / (s_perp + 1e-12), v);                // 7
        quant5(s.index({torch::indexing::Slice(), 1}), v);  // 7
        quant5(s.index({torch::indexing::Slice(), 2}), v);  // 7
        // --- line-Gaussian opacity (sigmoid) and contribution proxy ---
        auto a_line = torch::sigmoid(gaussians_->opacity_).index({mask}).flatten();
        auto a_all = torch::sigmoid(gaussians_->opacity_).flatten();
        quant5(a_line, v);                                  // 7
        v.push_back(a_line.sum().item<double>());
        v.push_back(a_all.sum().item<double>());
        v.push_back((a_line < 0.01f).sum().item<double>() / a_line.numel());
        v.push_back((a_line < 0.05f).sum().item<double>() / a_line.numel());
        v.push_back((s_par / (s_perp + 1e-12) < 2.0f).sum().item<double>() / s_par.numel());
        // --- accumulated xyz gradient norm (combined loss, from the last step) ---
        if (gaussians_->xyz_gradient_accum_.defined() &&
            gaussians_->xyz_gradient_accum_.size(0) == mask.size(0))
            quant5(gaussians_->xyz_gradient_accum_.index({mask}), v);
        else
            for (int i = 0; i < 7; ++i) v.push_back(std::nan(""));
    }

    // --- separated gradient magnitudes and their cosine angle (env-gated) ---
    std::vector<double> g;
    if (want_grad) {
        try {
            std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
            for (auto& kv : scene_->keyframes()) views.push_back(kv.second);
            const std::size_t idx0 = frozen_eval_kf_seq_[0];
            if (idx0 < views.size() && views[idx0]) {
                auto pkf = views[idx0];
                // photometric
                gaussians_->optimizer_->zero_grad(true);
                auto rp = GaussianRendererWithLine::renderWithLine(
                    pkf, pkf->image_height_, pkf->image_width_, gaussians_,
                    pipe_params_, background_, override_color_);
                auto mi = std::get<0>(rp) * undistort_mask_[pkf->camera_id_];
                loss_utils::l1_loss(mi, pkf->original_image_).backward();
                torch::Tensor gp;
                if (gaussians_->xyz_.grad().defined())
                    gp = gaussians_->xyz_.grad().index({mask}).detach().clone();
                gaussians_->optimizer_->zero_grad(true);
                // structural (reference weights)
                auto ls = gaussians_->computeVectorizedLineLoss(frozen_line_w_coherence_) +
                          gaussians_->computeLineShapeConstraint(frozen_line_w_ecc_,
                                                                 frozen_line_w_ori_);
                ls.backward();
                torch::Tensor gs;
                if (gaussians_->xyz_.grad().defined())
                    gs = gaussians_->xyz_.grad().index({mask}).detach().clone();
                torch::Tensor gr, gsc;
                if (gaussians_->rotation_.grad().defined())
                    gr = gaussians_->rotation_.grad().index({mask}).detach().clone();
                if (gaussians_->scaling_.grad().defined())
                    gsc = gaussians_->scaling_.grad().index({mask}).detach().clone();
                gaussians_->optimizer_->zero_grad(true);   // restore: grads were 0 here

                auto qn = [](const torch::Tensor& t, std::vector<double>& out) {
                    if (!t.defined() || t.numel() == 0) { out.push_back(std::nan("")); return; }
                    out.push_back(torch::norm(t, 2).item<double>());
                };
                qn(gp, g);
                qn(gs, g);
                qn(gr, g);
                qn(gsc, g);
                if (gp.defined() && gs.defined() && gp.numel() > 0) {
                    auto dot = (gp * gs).sum();
                    auto den = torch::norm(gp, 2) * torch::norm(gs, 2) + 1e-30;
                    g.push_back((dot / den).item<double>());   // cos angle
                    g.push_back((gs.norm() / (gp.norm() + 1e-30)).item<double>());
                } else {
                    g.push_back(std::nan(""));
                    g.push_back(std::nan(""));
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[FGM] grad diag failed: " << e.what() << std::endl;
            g.clear();
            for (int i = 0; i < 6; ++i) g.push_back(std::nan(""));
        }
    }

    {
        std::ofstream df(frozen_ablation_dir_ /
                         (tag + "_diag_step" + std::to_string(opt_steps) + ".csv"));
        df << "quantity,min,p10,p50,p90,max,mean,std\n";
        const char* nm[] = {"scale_parallel", "scale_perp_mean", "aspect_ratio",
                            "scale_axis1", "scale_axis2", "opacity_line"};
        for (int i = 0; i < 6; ++i) {
            df << nm[i];
            for (int j = 0; j < 7; ++j)
                df << ',' << std::setprecision(10) << v[i * 7 + j];
            df << '\n';
        }
        // Scalars, written into the p50 column so the header still parses.
        auto scalar = [&](const char* name, double val) {
            df << name << ",," << std::setprecision(10) << val << ",,,\n";
        };
        scalar("opacity_line_sum", v[42]);
        scalar("opacity_all_sum", v[43]);
        scalar("frac_line_opacity_lt_0.01", v[44]);
        scalar("frac_line_opacity_lt_0.05", v[45]);
        scalar("frac_line_aspect_lt_2", v[46]);
        if (v.size() >= 54) {
            df << "xyz_grad_accum_line";
            for (int j = 0; j < 7; ++j)
                df << ',' << std::setprecision(10) << v[47 + j];
            df << '\n';
        }
        if (!g.empty()) {
            const char* gn[] = {"grad_xyz_photometric", "grad_xyz_structural",
                                "grad_rotation_structural", "grad_scaling_structural",
                                "cos_angle_photo_struct", "struct_over_photo_ratio"};
            for (int i = 0; i < 6; ++i)
                scalar(gn[i], (i < static_cast<int>(g.size())) ? g[i] : std::nan(""));
        }
    }
    std::cerr << "[FGM][" << tag << "] diag step=" << opt_steps
              << " aspect p50=" << v[2 * 7 + 2] << " opacity_line mean=" << v[5 * 7 + 5]
              << " opacity_line_sum=" << v[42] << " frac_opacity_lt0.05=" << v[45]
              << " frac_aspect_lt2=" << v[46]
              << (g.size() > 4 ? (" cos=" + std::to_string(g[4])) : std::string())
              << std::endl;
}

// ---------------------------------------------------------------------------
// Controlled micro-benchmark (see header). One fixed parameter snapshot, one
// fixed view and resolution, no optimizer.step() for the loss variants.
//
// Each variant is measured in TWO loops that differ only by the backward call,
// after an identical warmup, and the derived backward time is the difference:
//     fwd_ms = loop(forward+loss) - loop(zero_grad)
//     bwd_ms = loop(forward+loss+backward) - loop(forward+loss)
// A sync brackets every loop so pending asynchronous work is never billed to
// the wrong section.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// perf-stage3 / 三: backward decomposition (PHOTO_SLAM_FGM_BWD_DECOMP=1).
//
// Splits one training iteration into stages measured INDEPENDENTLY on the same
// parameter snapshot, view, resolution, SH degree and mask:
//   A  image loss      forward (weighted L1+SSIM) evaluated on a diagnostic
//                      LEAF image, plus the gradient of that loss w.r.t. the
//                      image (the incoming gradient the renderer must consume)
//   B  renderer        backward, fed A's image gradient, WITHOUT recomputing
//                      the image loss (the render graph is rebuilt from the
//                      same state so the saved activations are identical)
//   C  public constraints (anisotropy + position/coherence + ecc + orientation)
//                      forward and backward, measured independently
//   D  full training step on the ORIGINAL full graph (reference) + optimizer
//
// Every stage reports host wall clock AND device (CUDA event) time; the gap is
// the CPU dispatch / synchronisation share. Stages are measured as separate,
// non-nested blocks, so nothing is double counted.
//
// The split is additionally VERIFIED: the parameter gradients of (A -> B) + C
// must match those of the original full path. Pre-stated tolerance:
//   global relative L2 <= 1e-3  AND  per-tensor max relative L2 <= 1e-3
// (float32 accumulation over ~2e5 parameters; the two paths reduce in a
// different order, so bitwise equality is NOT expected here - unlike the
// algebraic-identity checks in runEquivCheck, whose expectation is 0).
//
// detach() appears ONLY in this benchmark; it never runs during real training.
// ---------------------------------------------------------------------------
void GaussianMapperLine::runBackwardDecomposition(const std::string& tag, long long step)
{
    if (!gaussians_ || frozen_eval_kf_seq_.empty() || !gaussians_->xyz_.defined())
        return;
    std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
    for (auto& kv : scene_->keyframes())
        views.push_back(kv.second);
    const std::size_t idx0 = frozen_eval_kf_seq_[0];
    if (idx0 >= views.size() || !views[idx0])
        return;
    auto pkf = views[idx0];
    const int H = static_cast<int>(pkf->image_height_);
    const int Wd = static_cast<int>(pkf->image_width_);
    auto mask = undistort_mask_[pkf->camera_id_];
    auto gt = pkf->original_image_;

    const float lam = lambdaDssim();
    const float w_pos = frozen_line_w_coherence_;
    const float w_ecc = frozen_line_w_ecc_;
    const float w_ori = frozen_line_w_ori_;
    const float lam_aniso = opt_params_.lambda_anisotropy_;
    const float aniso_th = opt_params_.max_anisotropy_threshold_;

    const bool is_cuda = (device_type_ == torch::kCUDA);
    auto sync = [&]() { if (is_cuda) torch::cuda::synchronize(); };
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto msb = [](std::chrono::steady_clock::time_point a,
                  std::chrono::steady_clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
    };

    // Preserve the render-time buffers so the benchmark cannot perturb state.
    torch::Tensor saved_radii, saved_hits, saved_alpha;
    const bool had_radii = gaussians_->max_radii2D_.defined();
    if (had_radii) saved_radii = gaussians_->max_radii2D_.clone();
    const bool had_hits = gaussians_->debug_hit_count_.defined();
    if (had_hits) saved_hits = gaussians_->debug_hit_count_.clone();
    const bool had_alpha = gaussians_->debug_alpha_accum_.defined();
    if (had_alpha) saved_alpha = gaussians_->debug_alpha_accum_.clone();

    auto render_now = [&]() {
        auto rp = GaussianRendererWithLine::renderWithLine(
            pkf, H, Wd, gaussians_, pipe_params_, background_, override_color_);
        return std::get<0>(rp);
    };
    // Weighted L1 + SSIM on the masked image (identical formula to training).
    auto photo_on = [&](const torch::Tensor& img) {
        auto mi = img * mask;
        auto l1 = loss_utils::l1_loss(mi, gt);
        return (1.0f - lam) * l1 + lam * (1.0f - loss_utils::ssim(mi, gt, device_type_));
    };
    auto aniso_term = [&]() {
        auto sa = gaussians_->getScaling();
        auto mx = std::get<0>(torch::max(sa, 1));
        auto mn = std::get<0>(torch::min(sa, 1));
        auto ratio = mx / (mn + 1e-6);
        return lam_aniso * torch::pow(torch::clamp(ratio - aniso_th, 0), 2).mean();
    };
    auto struct_terms = [&]() {
        auto sa = gaussians_->getScaling();
        return gaussians_->computeVectorizedLineLoss(w_pos) +
               gaussians_->computeLineShapeConstraint(w_ecc, w_ori, sa);
    };
    auto zero_grads = [&]() {
        auto z = [](const torch::Tensor& t) {
            if (t.defined() && t.grad().defined()) t.grad().zero_(); };
        z(gaussians_->xyz_); z(gaussians_->opacity_); z(gaussians_->scaling_);
        z(gaussians_->rotation_); z(gaussians_->features_dc_);
        if (gaussians_->features_rest_.defined()) z(gaussians_->features_rest_);
    };

    const int WARM = 20, REPS = 50, NBLOCK = 3;
    auto time_it = [&](auto&& fn, double& host_ms, double& dev_ms) {
        for (int i = 0; i < WARM; ++i) fn();
        sync();
        double bh = 1e30, bd = 1e30;
        for (int b = 0; b < NBLOCK; ++b) {
            at::cuda::CUDAEvent e0(0u), e1(0u);
            if (is_cuda) e0.record();
            auto t0 = now();
            for (int i = 0; i < REPS; ++i) fn();
            sync();
            auto t1 = now();
            if (is_cuda) { e1.record(); e1.synchronize(); }
            const double h = msb(t0, t1) / REPS;
            const double d = is_cuda ? static_cast<double>(e0.elapsed_time(e1)) / REPS : h;
            if (h < bh) bh = h;
            if (d < bd) bd = d;
        }
        host_ms = bh;
        dev_ms = bd;
    };

    std::ofstream out(frozen_ablation_dir_ /
                      (tag + "_bwd_decomp_step" + std::to_string(step) + ".csv"));
    out << "tag,step,stage,host_ms_per_iter,dev_ms_per_iter,cpu_share_ms\n";
    auto emit = [&](const char* stage, double h, double d) {
        out << tag << ',' << step << ',' << stage << ',' << h << ',' << d << ','
            << (h - d) << '\n';
        std::cerr << "[BwdDecomp][" << tag << "] step=" << step << " " << stage
                  << "  host=" << h << " ms  dev=" << d << " ms  (cpu-dispatch "
                  << (h - d) << " ms)" << std::endl;
    };

    // ---- A: image loss forward, and forward+backward on a diagnostic leaf ----
    double a_f = 0, a_fb = 0, a_df = 0, a_dfb = 0;
    torch::Tensor last_leaf;
    {
        auto img0 = render_now().detach();   // benchmark-only detach
        auto fwd_fn = [&]() {
            auto leaf = img0.clone().requires_grad_(true);
            auto l = photo_on(leaf);
            last_leaf = leaf;
            (void)l;
        };
        time_it(fwd_fn, a_f, a_df);
        auto fb_fn = [&]() {
            auto leaf = img0.clone().requires_grad_(true);
            auto l = photo_on(leaf);
            l.backward();
            last_leaf = leaf;
        };
        time_it(fb_fn, a_fb, a_dfb);
    }
    // Reference incoming gradient for the renderer backward (from a clean run).
    torch::Tensor grad_image;
    {
        auto leaf = render_now().detach().clone().requires_grad_(true);
        auto l = photo_on(leaf);
        l.backward();
        grad_image = leaf.grad().detach().clone();
        auto gnorm = grad_image.norm().item().toFloat();
        std::cerr << "[BwdDecomp][" << tag << "] step=" << step
                  << " |dL/dimage|_2 = " << gnorm
                  << "  |image|_mean = " << leaf.detach().abs().mean().item().toFloat()
                  << std::endl;
        out << tag << ',' << step << ",image_grad_norm_L2,," << gnorm << ",\n";
    }
    emit("A_image_loss_fwd", a_f, a_df);
    emit("A_image_loss_fwd_bwd", a_fb, a_dfb);
    emit("A_image_loss_bwd", a_fb - a_f, a_dfb - a_df);

    // ---- B: renderer forward (reference) and renderer backward only ----
    double b_f = 0, b_df = 0, b_fb = 0, b_dfb = 0;
    {
        auto fwd_fn = [&]() { auto r = render_now(); (void)r; };
        time_it(fwd_fn, b_f, b_df);
        auto bwd_fn = [&]() {
            auto r = render_now();
            r.backward(grad_image);   // only the renderer backward
        };
        time_it(bwd_fn, b_fb, b_dfb);
    }
    emit("B_render_fwd", b_f, b_df);
    emit("B_render_fwd_bwd", b_fb, b_dfb);
    emit("B_render_backward_only", b_fb - b_f, b_dfb - b_df);

    // ---- C: public constraints, forward and forward+backward ----
    double c_f = 0, c_df = 0, c_fb = 0, c_dfb = 0;
    {
        auto fwd_fn = [&]() { auto l = aniso_term() + struct_terms(); (void)l; };
        time_it(fwd_fn, c_f, c_df);
        auto fb_fn = [&]() {
            auto l = aniso_term() + struct_terms();
            l.backward();
        };
        time_it(fb_fn, c_fb, c_dfb);
    }
    emit("C_constraints_fwd", c_f, c_df);
    emit("C_constraints_fwd_bwd", c_fb, c_dfb);
    emit("C_constraints_bwd", c_fb - c_f, c_dfb - c_df);

    // ---- D: original full training step (reference graph) ----
    double d_f = 0, d_df = 0, d_fb = 0, d_dfb = 0;
    {
        auto fwd_fn = [&]() {
            auto r = render_now();
            auto l = photo_on(r) + aniso_term() + struct_terms();
            (void)l;
        };
        time_it(fwd_fn, d_f, d_df);
        auto fb_fn = [&]() {
            auto r = render_now();
            auto l = photo_on(r) + aniso_term() + struct_terms();
            l.backward();
        };
        time_it(fb_fn, d_fb, d_dfb);
    }
    emit("D_full_fwd", d_f, d_df);
    emit("D_full_fwd_bwd", d_fb, d_dfb);
    emit("D_full_graph_backward", d_fb - d_f, d_dfb - d_df);
    // CPU share of one full forward+backward (host minus device). This is the
    // dispatch/synchronisation cost that a device-only profiler cannot see.
    emit("D_cpu_dispatch_share", d_fb - d_f - (d_dfb - d_df), 0.0);

    // optimizer.step on CLONED leaves (the training optimizer is never touched)
    {
        std::vector<torch::Tensor> v;
        v.push_back(gaussians_->xyz_.detach().clone().requires_grad_());
        v.push_back(gaussians_->features_dc_.detach().clone().requires_grad_());
        v.push_back(gaussians_->opacity_.detach().clone().requires_grad_());
        v.push_back(gaussians_->scaling_.detach().clone().requires_grad_());
        v.push_back(gaussians_->rotation_.detach().clone().requires_grad_());
        torch::optim::AdamOptions ao;
        ao.set_lr(1e-4);
        ao.eps() = 1e-15;
        auto opt = std::make_shared<torch::optim::Adam>(
            std::vector<torch::Tensor>{v[0]}, ao);
        for (std::size_t i = 1; i < v.size(); ++i)
            opt->add_param_group(std::vector<torch::Tensor>{v[i]});
        for (auto& t : v) t.mutable_grad() = torch::ones_like(t);
        for (int i = 0; i < WARM; ++i) opt->step();
        sync();
        auto t0 = now();
        for (int i = 0; i < REPS; ++i) opt->step();
        sync();
        emit("E_optimizer_step", msb(t0, now()) / REPS, msb(t0, now()) / REPS);
    }

    // ---- gradient-equivalence verification: split (A -> B) + C vs full ----
    auto snapshot = [&]() {
        std::vector<torch::Tensor> g;
        auto add = [&](const torch::Tensor& t) {
            torch::Tensor gg;
            if (t.defined() && t.grad().defined())
                gg = t.grad().detach().to(torch::kFloat32).flatten().clone();
            g.push_back(gg);
        };
        add(gaussians_->xyz_); add(gaussians_->opacity_); add(gaussians_->scaling_);
        add(gaussians_->rotation_); add(gaussians_->features_dc_);
        if (gaussians_->features_rest_.defined() && gaussians_->features_rest_.numel() > 0)
            add(gaussians_->features_rest_);
        return g;
    };
    auto rel_l2 = [](const torch::Tensor& a, const torch::Tensor& b) {
        const double num = (a - b).norm().item().toDouble();
        const double den = b.norm().item().toDouble() + 1e-12;
        return num / den;
    };

    std::vector<torch::Tensor> g_full, g_split;
    {   // reference: original full graph
        zero_grads();
        auto r = render_now();
        auto l = photo_on(r) + aniso_term() + struct_terms();
        l.backward();
        g_full = snapshot();
    }
    {   // split: image-loss grad -> renderer backward, then constraints
        zero_grads();
        auto r = render_now();
        auto leaf = r.detach().clone().requires_grad_(true);
        auto l_img = photo_on(leaf);
        l_img.backward();
        r.backward(leaf.grad());          // same value, detached path
        auto l_c = aniso_term() + struct_terms();
        l_c.backward();
        g_split = snapshot();
    }
    double worst = 0.0, num_sq = 0.0, den_sq = 0.0;
    for (std::size_t i = 0; i < g_full.size() && i < g_split.size(); ++i) {
        const bool ok_full = g_full[i].defined(), ok_split = g_split[i].defined();
        const double rl = (ok_full && ok_split) ? rel_l2(g_split[i], g_full[i])
                                                : (ok_full == ok_split ? 0.0 : 1e9);
        if (rl > worst) worst = rl;
        if (ok_full && ok_split) {
            num_sq += std::pow((g_split[i] - g_full[i]).norm().item().toDouble(), 2);
            den_sq += std::pow(g_full[i].norm().item().toDouble(), 2);
        }
        out << tag << ',' << step << ",grad_rel_l2_tensor" << i << ",," << rl << ",\n";
    }
    const double global_rel = std::sqrt(num_sq) / (std::sqrt(den_sq) + 1e-12);
    const int pass = (global_rel <= 1e-3 && worst <= 1e-3) ? 1 : 0;
    out << tag << ',' << step << ",GRAD_EQUIV_global_rel_l2,," << global_rel << ",\n";
    out << tag << ',' << step << ",GRAD_EQUIV_worst_tensor_rel_l2,," << worst << ",\n";
    out << tag << ',' << step << ",GRAD_EQUIV_tolerance_pass,," << pass << ",\n";
    std::cerr << "[BwdDecomp][" << tag << "] step=" << step
              << " GRAD EQUIV: global_rel_l2=" << global_rel << " worst_tensor="
              << worst << " (tolerance 1e-3) -> " << (pass ? "PASS" : "FAIL")
              << std::endl;
    out.close();

    // restore exactly the state the eval point was in
    if (had_radii && saved_radii.defined()) gaussians_->max_radii2D_.copy_(saved_radii);
    if (had_hits && saved_hits.defined()) gaussians_->debug_hit_count_.copy_(saved_hits);
    if (had_alpha && saved_alpha.defined()) gaussians_->debug_alpha_accum_.copy_(saved_alpha);
    gaussians_->optimizer_->zero_grad(true);
    sync();
}

// ---------------------------------------------------------------------------
// mapper-budget-v1: capture support.
//
// maybeSnapshotAndAudit() persists a FULL resumable model snapshot (all tensors,
// not a PLY) at the iterations listed in PHOTO_SLAM_SNAPSHOT_AT, plus the
// mapper-side schedule state needed to interpret it. The later rounds load these
// snapshots instead of restarting SLAM.
//
// lineSampleAuditLog() records, at the line-sampling entry, how many candidates
// existed and why they were rejected. Its purpose is to explain - by data, not
// by assumption - why the online run seeds almost no line Gaussians; nothing
// here relaxes a threshold.
// ---------------------------------------------------------------------------
void GaussianMapperLine::maybeSnapshotAndAudit()
{
    static const std::vector<long long> at = []() {
        std::vector<long long> v;
        const char* e = std::getenv("PHOTO_SLAM_SNAPSHOT_AT");
        if (e) {
            std::stringstream ss(e);
            std::string t;
            while (std::getline(ss, t, ','))
                if (!t.empty()) v.push_back(std::atoll(t.c_str()));
        }
        return v;
    }();
    static const std::string dir = []() {
        const char* e = std::getenv("PHOTO_SLAM_SNAPSHOT_DIR");
        return std::string(e ? e : "");
    }();
    if (dir.empty() || !gaussians_ || at.empty()) return;
    const long long it = getIteration();
    bool hit = false;
    for (long long a : at) if (a == it) hit = true;
    if (!hit) return;
    const std::string sp = dir + "/model_step" + std::to_string(it) + ".pt";
    gaussians_->saveFullSnapshot(sp);      // also records the line population
    std::ofstream m(dir + "/state_step" + std::to_string(it) + ".txt");
    m << "iteration " << it << '\n'
      << "default_sh " << default_sh_ << '\n'
      << "total_training_time_ms " << total_training_time_ms_ << '\n'
      << "n_gaussians " << gaussians_->auditTotal() << '\n'
      << "device " << (device_type_ == torch::kCUDA ? "cuda" : "cpu") << '\n';
    m.flush();
    std::cerr << "[Snapshot] step " << it << " -> " << sp << " (n="
              << gaussians_->auditTotal() << ", sh=" << default_sh_ << ")"
              << std::endl;
}

void GaussianMapperLine::lineSampleAuditLog(long long raw_points, long long candidates,
                                            long long rejected_voxel, long long accepted)
{
    static const std::string prefix = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_SAMPLE_AUDIT");
        return std::string(e ? e : "");
    }();
    if (prefix.empty()) return;
    if (!lsample_initialised_) {
        lsample_initialised_ = true;
        lsample_out_.open(prefix + "_linesample.csv", std::ios::out | std::ios::trunc);
        if (!lsample_out_.good()) return;
        lsample_out_ << "iteration,raw_points,candidates,"
                        "rejected_voxel_occupied,accepted\n";
    }
    ++lsample_calls_;
    lsample_out_ << getIteration() << ',' << raw_points << ',' << candidates << ','
                 << rejected_voxel << ',' << accepted << '\n';
    lsample_out_.flush();
}

// ---------------------------------------------------------------------------
// mapper-budget-v2-hash: the ONE algorithmic variable of this round.
//
// R100 accepts every valid inactive-geo candidate; R50 accepts a deterministic
// hash-selected half of the SAME candidate batch. The ranking key is
// splitmix64(batch_keyframe_id, candidate_index, seed) - stable, consumes no
// training random numbers, and keeps the original candidate order on insertion.
// It is explicitly NOT image-space stratification and makes no coverage claim.
// The selection (kept indices + their hashes) is written to disk for audit.
// ---------------------------------------------------------------------------
namespace {
inline std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
}  // namespace

void GaussianMapperLine::budgetInit()
{
    if (budget_initialised_) return;
    budget_initialised_ = true;
    if (const char* e = std::getenv("PHOTO_SLAM_INACTIVE_KEEP"))
        budget_keep_frac_ = static_cast<float>(std::atof(e));
    if (const char* e = std::getenv("PHOTO_SLAM_INACTIVE_SEED"))
        budget_seed_ = static_cast<std::uint64_t>(std::strtoull(e, nullptr, 10));
    if (const char* e = std::getenv("PHOTO_SLAM_BUDGET_DIR")) budget_dir_ = e;
    if (!budget_dir_.empty()) {
        std::filesystem::create_directories(budget_dir_);
        budget_kept_out_.open(budget_dir_ + "/inactive_kept.csv", std::ios::out | std::ios::trunc);
        budget_evt_out_.open(budget_dir_ + "/inactive_batches.csv", std::ios::out | std::ios::trunc);
        budget_step_out_.open(budget_dir_ + "/train_steps.csv", std::ios::out | std::ios::trunc);
        budget_kept_out_ << "batch,keyframe_fid,global_candidate_index,local_index,hash,kept\n";
        budget_evt_out_ << "batch,iteration,keyframe_fid,candidates,kept,keep_frac,seed\n";
        budget_step_out_ << "raw_iteration,route,keep_frac\n";
    }
    std::cerr << "[Budget] keep_frac=" << budget_keep_frac_ << " seed=" << budget_seed_
              << " dir=" << budget_dir_ << std::endl;
}

bool GaussianMapperLine::applyInactiveBudget(torch::Tensor& pts, torch::Tensor& cols,
                                           torch::Tensor* ref_kf_ids)
{
    budgetInit();
    const long long N = pts.defined() ? pts.size(0) : 0;
    if (N <= 0) return false;
    const long long base = budget_cand_;
    ++budget_batches_;

    if (budget_keep_frac_ >= 1.0f) {                 // R100: accept the whole batch
        budget_cand_ += N;
        budget_kept_ += N;
        if (budget_evt_out_.is_open())
            budget_evt_out_ << budget_batches_ << ',' << getIteration() << ','
                            << budget_cur_kf_fid_ << ',' << N << ',' << N << ','
                            << budget_keep_frac_ << ',' << budget_seed_ << '\n';
        return false;                                 // tensors unchanged
    }

    const long long n_keep =
        std::max(0LL, static_cast<long long>(std::llround(budget_keep_frac_ * static_cast<double>(N))));
    std::vector<std::uint64_t> h(static_cast<std::size_t>(N));
    for (long long i = 0; i < N; ++i) {
        const std::uint64_t key =
            splitmix64(static_cast<std::uint64_t>(budget_cur_kf_fid_ + 1) * 0x100000001B3ULL ^
                       static_cast<std::uint64_t>(base + i) * 0xC2B2AE3D27D4EB4FULL ^
                       budget_seed_);
        h[static_cast<std::size_t>(i)] = key;
    }
    torch::Tensor ht;
    {
        std::vector<std::int64_t> hi(static_cast<std::size_t>(N));
        for (long long i = 0; i < N; ++i) hi[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(h[static_cast<std::size_t>(i)]);
        ht = torch::from_blob(hi.data(), {N}, torch::TensorOptions().dtype(torch::kLong)).clone();
    }
    auto order = std::get<1>(torch::sort(ht, /*dim=*/0));          // ascending by hash
    auto keep_local = order.index({torch::indexing::Slice(0, n_keep)});
    auto mask = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool));
    mask.index_put_({keep_local}, true);

    if (budget_kept_out_.is_open()) {
        auto local = torch::nonzero(mask).flatten();
        auto lc = local.to(torch::kCPU);
        auto ac = lc.accessor<std::int64_t, 1>();
        for (long long i = 0; i < lc.size(0); ++i) {
            const long long li = ac[i];
            budget_kept_out_ << budget_batches_ << ',' << budget_cur_kf_fid_ << ','
                             << (base + li) << ',' << li << ',' << h[static_cast<std::size_t>(li)]
                             << ',' << 1 << '\n';
        }
        budget_kept_out_.flush();
    }
    budget_cand_ += N;
    budget_kept_ += n_keep;
    if (budget_evt_out_.is_open()) {
        budget_evt_out_ << budget_batches_ << ',' << getIteration() << ','
                        << budget_cur_kf_fid_ << ',' << N << ',' << n_keep << ','
                        << budget_keep_frac_ << ',' << budget_seed_ << '\n';
        budget_evt_out_.flush();
    }
    auto mdev = mask.to(pts.device());
    pts = pts.index({mdev}).contiguous();
    if (cols.defined()) cols = cols.index({mdev}).contiguous();
    if (ref_kf_ids && ref_kf_ids->defined() && ref_kf_ids->size(0) == N)
        *ref_kf_ids = ref_kf_ids->to(mdev.device()).index({mdev}).contiguous();
    return true;                                      // tensors filtered
}

void GaussianMapperLine::budgetSetCondition(float keep_frac, const std::string& dir)
{
    budget_keep_frac_ = keep_frac;
    budget_dir_ = dir;
    budget_initialised_ = true;          // overrides any env-based init
    budget_batches_ = budget_cand_ = budget_kept_ = 0;
    if (budget_kept_out_.is_open()) budget_kept_out_.close();
    if (budget_evt_out_.is_open()) budget_evt_out_.close();
    if (budget_step_out_.is_open()) budget_step_out_.close();
    if (!budget_dir_.empty()) {
        std::filesystem::create_directories(budget_dir_);
        budget_kept_out_.open(budget_dir_ + "/inactive_kept.csv", std::ios::out | std::ios::trunc);
        budget_evt_out_.open(budget_dir_ + "/inactive_batches.csv", std::ios::out | std::ios::trunc);
        budget_kept_out_ << "batch,keyframe_fid,global_candidate_index,local_index,hash,kept\n";
        budget_evt_out_ << "batch,iteration,keyframe_fid,candidates,kept,keep_frac,seed\n";
    }
    std::cerr << "[Budget][" << budget_dir_ << "] keep_frac=" << budget_keep_frac_
              << " seed=" << budget_seed_ << std::endl;
}

void GaussianMapperLine::budgetSummary(){
    if (budget_evt_out_.is_open()) {
        std::cerr << "[Budget] batches=" << budget_batches_ << " candidates=" << budget_cand_
                  << " kept=" << budget_kept_ << " keep_frac=" << budget_keep_frac_
                  << " effective="
                  << (budget_cand_ > 0 ? static_cast<double>(budget_kept_) / budget_cand_ : 0.0)
                  << std::endl;
    }
}

void GaussianMapperLine::recordTrainStep()
{
    // Per-step CSV only when explicitly asked for; the formal timing run leaves
    // it off so the hook costs one static-bool test plus the (cheap) snapshot
    // check.
    static const bool log_steps = (std::getenv("PHOTO_SLAM_BUDGET_STEPS") != nullptr);
    if (log_steps) {
        budgetInit();
        if (budget_step_out_.is_open())
            budget_step_out_ << getIteration() << ",train," << budget_keep_frac_ << '\n';
    }
    // ------------------------------------------------------------------
    // raster-perf-v6: per-optimizer-update rasteriser call trace.
    // Counters only (no CUDA events, no host sync), so enabling this does not
    // change the kernel timeline. OFF in formal performance runs.
    // ------------------------------------------------------------------
    static const bool trace_calls =
        (std::getenv("PHOTO_SLAM_RASTER_CALLTRACE") != nullptr);
    if (trace_calls)
    {
        long long c[6] = {0, 0, 0, 0, 0, 0};
        int d[4] = {0, 0, 0, 0};
        RasterCallTraceGet(c);
        RasterCallTraceDims(d);
        if (c[0] || c[1])
            printf("[CallTrace] iter=%lu fwd_calls=%lld bwd_calls=%lld "
                   "fwd_render=%lld bwd_render=%lld fwd_launch=%lld "
                   "bwd_launch=%lld P=%d sh=%d res=%dx%d backend=%s\n",
                   (unsigned long)getIteration(), c[0], c[1], c[2], c[3], c[4],
                   c[5], d[0], d[3], d[1], d[2],
                   RasterBackendGet() == 0 ? "A"
                                           : (RasterBackendGet() == 1 ? "B" : "C"));
        RasterCallTraceReset();
    }
    // Unified snapshot trigger: fires for step 0 / 100 / 9000 / 20000 whether the
    // step was executed by trainForOneIterationErrorGuided() or
    // trainForOneIteration().
    maybeSnapshotAndAudit();
}

void GaussianMapperLine::replayInitInactiveDriver()
{
    const char* e = std::getenv("PHOTO_SLAM_REPLAY_INACTIVE");
    replay_inactive_enabled_ = (e != nullptr && std::atoi(e) != 0);
    // The depth cache and its cursor are per-run accumulators. Without this the
    // second arm would inherit the FIRST arm's leftover candidates and the two
    // arms would no longer share the same external stream (measured: R50's first
    // batch contained keyframe ids left over from R100).
    depth_cached_ = 0;
    depth_cache_points_ = torch::Tensor();
    depth_cache_colors_ = torch::Tensor();
    replay_kf_cursor_ = 0;
    if (!replay_inactive_enabled_ || !scene_) return;
    replay_kf_order_.clear();
    for (auto& kv : scene_->keyframes()) replay_kf_order_.push_back(kv.second);
    if (const char* p = std::getenv("PHOTO_SLAM_REPLAY_INACTIVE_PERIOD"))
        replay_inactive_period_ = std::atoll(p);
    if (replay_inactive_period_ < 1) replay_inactive_period_ = 1;
    std::cerr << "[ReplayInactive] keyframes=" << replay_kf_order_.size()
              << " period=" << replay_inactive_period_ << std::endl;
}

void GaussianMapperLine::replayDriveInactiveGeo()
{
    if (!replay_inactive_enabled_ || frozen_fixed_count_) return;
    if (replay_kf_cursor_ >= replay_kf_order_.size()) return;
    if (getIteration() % replay_inactive_period_ != 0) return;
    auto pkf = replay_kf_order_[replay_kf_cursor_++];
    if (!pkf) return;
    // Deterministic: one frozen keyframe per period, ascending id order, the same
    // in every arm. This replaces the SLAM keyframe thread as the supplier of the
    // external geometry stream; the candidate GENERATION is therefore not
    // re-measured here (stated explicitly in the report).
    pkf->done_inactive_geo_densify_ = false;
    budget_cur_kf_fid_ = static_cast<long long>(pkf->fid_);
    increasePcdByKeyframeInactiveGeoDensify(pkf);
}

void GaussianMapperLine::runMicroBench(const std::string& tag, long long step)
{
    if (!gaussians_ || frozen_eval_kf_seq_.empty() || !gaussians_->xyz_.defined())
        return;
    std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
    for (auto& kv : scene_->keyframes())
        views.push_back(kv.second);
    const std::size_t idx0 = frozen_eval_kf_seq_[0];
    if (idx0 >= views.size() || !views[idx0])
        return;
    auto pkf = views[idx0];
    const int H = static_cast<int>(pkf->image_height_);
    const int Wd = static_cast<int>(pkf->image_width_);
    auto mask = undistort_mask_[pkf->camera_id_];
    auto gt = pkf->original_image_;

    const float w_pos = frozen_line_w_coherence_;
    const float w_ecc = frozen_line_w_ecc_;
    const float w_ori = frozen_line_w_ori_;
    const float lam = lambdaDssim();
    const float lam_aniso = opt_params_.lambda_anisotropy_;
    const float aniso_th = opt_params_.max_anisotropy_threshold_;

    auto sync = [&]() {
        if (device_type_ == torch::kCUDA)
            torch::cuda::synchronize();
    };

    // Preserve render-time buffers so the micro-benchmark cannot perturb state.
    torch::Tensor saved_radii, saved_hits, saved_alpha;
    const bool had_radii = gaussians_->max_radii2D_.defined();
    if (had_radii) saved_radii = gaussians_->max_radii2D_.clone();
    const bool had_hits = gaussians_->debug_hit_count_.defined();
    if (had_hits) saved_hits = gaussians_->debug_hit_count_.clone();
    const bool had_alpha = gaussians_->debug_alpha_accum_.defined();
    if (had_alpha) saved_alpha = gaussians_->debug_alpha_accum_.clone();

    auto buildLoss = [&](int variant) -> torch::Tensor {
        auto rp = GaussianRendererWithLine::renderWithLine(
            pkf, H, Wd, gaussians_, pipe_params_, background_, override_color_);
        if (variant == 0)
            return torch::Tensor();                       // render only
        auto mi = std::get<0>(rp) * mask;
        auto l1 = loss_utils::l1_loss(mi, gt);
        auto loss = (1.0 - lam) * l1 +
                    lam * (1.0 - loss_utils::ssim(mi, gt, device_type_));
        if (variant == 1)
            return loss;                                  // + photometric
        torch::Tensor scaling_act;
        {
            // anisotropy penalty: a public constraint, kept in every variant
            // that carries a loss at all
            auto sc = gaussians_->getScaling();
            scaling_act = sc;
            auto mx = std::get<0>(torch::max(sc, 1));
            auto mn = std::get<0>(torch::min(sc, 1));
            auto ratio = mx / (mn + 1e-6);
            loss = loss + lam_aniso *
                   torch::pow(torch::clamp(ratio - aniso_th, 0), 2).mean();
        }
        if (variant == 2)
            return loss;                                  // + aniso
        if (variant == 3)
            return loss + gaussians_->computeVectorizedLineLoss(w_pos);
        if (variant == 4)
            return loss + gaussians_->computeLineShapeConstraint(w_ecc, 0.0f);
        if (variant == 5)
            return loss + gaussians_->computeLineShapeConstraint(0.0f, w_ori);
        if (variant == 6)
            return loss + gaussians_->computeVectorizedLineLoss(w_pos) +
                   gaussians_->computeLineShapeConstraint(w_ecc, w_ori, scaling_act);
        if (variant == 7)
            return loss + gaussians_->computeVectorizedLineLossRef(w_pos) +
                   gaussians_->computeLineShapeConstraintRef(w_ecc, w_ori);
        // Variants 8/9 build a graph containing ONLY the structural loss, so the
        // photometric backward (which dominates and fluctuates) cannot mask the
        // difference being measured.
        if (variant == 8) {
            auto sa = gaussians_->getScaling();
            return gaussians_->computeVectorizedLineLoss(w_pos) +
                   gaussians_->computeLineShapeConstraint(w_ecc, w_ori, sa);
        }
        if (variant == 9)
            return gaussians_->computeVectorizedLineLossRef(w_pos) +
                   gaussians_->computeLineShapeConstraintRef(w_ecc, w_ori);
        return loss;
    };

    const char* nms[] = {"render_only", "photo", "photo_aniso", "photo_aniso_position",
                         "photo_aniso_ecc", "photo_aniso_ori",
                         "photo_aniso_all_struct_fast", "photo_aniso_all_struct_ref",
                         "struct_only_fast", "struct_only_ref"};
    const bool bwd[] = {false, true, true, true, true, true, true, true, true, true};
    const int NVARIANT = 10;
    const int WARM = 30, REPS = 200;

    // min-of-NBLOCK blocks: the GPU is shared, so the minimum block is the most
    // robust estimator of the launch-bound cost of this code.
    const int NBLOCK = 3;
    auto timeLoops = [&](int variant, bool with_bwd, double& out_ms) {
        double best = std::numeric_limits<double>::max();
        for (int b = 0; b < NBLOCK; ++b) {
            for (int i = 0; i < WARM; ++i) {
                gaussians_->optimizer_->zero_grad(true);
                auto l = buildLoss(variant);
                if (with_bwd && bwd[variant] && l.defined())
                    l.backward();
            }
            sync();
            gaussians_->optimizer_->zero_grad(true);
            sync();
            const auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < REPS; ++i) {
                gaussians_->optimizer_->zero_grad(true);
                auto l = buildLoss(variant);
                if (with_bwd && bwd[variant] && l.defined())
                    l.backward();
            }
            sync();
            best = std::min(best,
                std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                    std::chrono::steady_clock::now() - t0).count() / REPS);
        }
        out_ms = best;
    };

    double zero_ms = 0.0;
    {
        for (int i = 0; i < WARM; ++i) gaussians_->optimizer_->zero_grad(true);
        sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i) gaussians_->optimizer_->zero_grad(true);
        sync();
        zero_ms = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                      std::chrono::steady_clock::now() - t0).count() / REPS;
    }

    std::ofstream out(frozen_ablation_dir_ /
                      (tag + "_microbench_step" + std::to_string(step) + ".csv"));
    out << "variant,zero_grad_ms,fwd_only_ms,fwd_bwd_ms,derived_fwd_ms,derived_bwd_ms\n";
    std::cerr << "[MicroBench][" << tag << "] step=" << step << " zero_grad=" << zero_ms
              << " ms" << std::endl;
    for (int v = 0; v < NVARIANT; ++v) {
        double t_f, t_fb;
        timeLoops(v, false, t_f);
        if (!bwd[v]) {
            t_fb = t_f;
        } else {
            timeLoops(v, true, t_fb);
        }
        out << nms[v] << ',' << zero_ms << ',' << t_f << ',' << t_fb << ','
            << (t_f - zero_ms) << ',' << (t_fb - t_f) << '\n';
        std::cerr << "[MicroBench][" << tag << "] " << nms[v]
                  << "  fwd=" << (t_f - zero_ms) << " ms  bwd=" << (t_fb - t_f)
                  << " ms" << std::endl;
    }

    // optimizer.step cost, measured on CLONED leaves so the training optimizer's
    // state is never touched.
    {
        std::vector<torch::Tensor> v;
        v.push_back(gaussians_->xyz_.detach().clone().requires_grad_());
        v.push_back(gaussians_->features_dc_.detach().clone().requires_grad_());
        if (gaussians_->features_rest_.defined() && gaussians_->features_rest_.numel() > 0)
            v.push_back(gaussians_->features_rest_.detach().clone().requires_grad_());
        v.push_back(gaussians_->opacity_.detach().clone().requires_grad_());
        v.push_back(gaussians_->scaling_.detach().clone().requires_grad_());
        v.push_back(gaussians_->rotation_.detach().clone().requires_grad_());
        torch::optim::AdamOptions ao;
        ao.set_lr(1e-4);
        ao.eps() = 1e-15;
        auto opt = std::make_shared<torch::optim::Adam>(
            std::vector<torch::Tensor>{v[0]}, ao);
        for (std::size_t i = 1; i < v.size(); ++i)
            opt->add_param_group(std::vector<torch::Tensor>{v[i]});
        for (auto& t : v)
            t.mutable_grad() = torch::ones_like(t);
        for (int i = 0; i < WARM; ++i) opt->step();
        sync();
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < REPS; ++i) opt->step();
        sync();
        const double ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
                std::chrono::steady_clock::now() - t0).count() / REPS;
        out << "optimizer_step," << 0 << ',' << ms << ',' << ms << ',' << ms << ',' << 0
            << '\n';
        std::cerr << "[MicroBench][" << tag << "] optimizer_step=" << ms << " ms" << std::endl;
    }
    out.close();

    // restore exactly the state the eval point was in
    if (had_radii && saved_radii.defined()) gaussians_->max_radii2D_.copy_(saved_radii);
    if (had_hits && saved_hits.defined()) gaussians_->debug_hit_count_.copy_(saved_hits);
    if (had_alpha && saved_alpha.defined()) gaussians_->debug_alpha_accum_.copy_(saved_alpha);
    gaussians_->optimizer_->zero_grad(true);
    sync();
}

// ---------------------------------------------------------------------------
// Line-density experiment: deterministic, nested, direction-stratified subset.
// ---------------------------------------------------------------------------
torch::Tensor GaussianMapperLine::makeLineSubsetMask(float keep_frac) const
{
    torch::NoGradGuard ng;
    const auto is_line_cpu = gaussians_->is_line_.to(torch::kCPU).to(torch::kBool).flatten();
    auto keep = (~is_line_cpu).clone();                       // ordinary Gaussians: always kept
    auto idx = torch::nonzero(is_line_cpu).flatten();          // line rows, ascending order
    const long long nl = idx.size(0);
    if (nl == 0) return keep;
    if (keep_frac >= 1.0f) { keep.index_put_({idx}, true); return keep; }
    if (keep_frac <= 0.0f) return keep;                        // Q0: every line row dropped

    int k = static_cast<int>(std::lround(1.0f / keep_frac));
    if (k < 1) k = 1;

    const auto dir = gaussians_->line_dir_w_.to(torch::kCPU).to(torch::kFloat).contiguous();
    auto da = dir.accessor<float, 2>();
    auto ia = idx.accessor<int64_t, 1>();

    // Group by the quantised world direction so that a stride does not drop a
    // whole line. Canonicalise the sign first so opposite directions share a
    // group. (No explicit MapLine association exists at freeze time.)
    std::map<std::array<long long, 3>, std::vector<long long>> groups;
    for (long long p = 0; p < nl; ++p) {
        const long long row = ia[p];            // GLOBAL row index into line_dir_w_
        float x = da[row][0], y = da[row][1], z = da[row][2];
        if (x < 0.f || (x == 0.f && (y < 0.f || (y == 0.f && z < 0.f)))) { x = -x; y = -y; z = -z; }
        groups[{std::llround(x * 100.0), std::llround(y * 100.0),
                std::llround(z * 100.0)}].push_back(row);
    }
    std::vector<int64_t> kept;
    for (auto& kv : groups) {
        auto& v = kv.second;                       // ascending row index order
        for (std::size_t p = 0; p < v.size(); ++p)
            if (p % k == 0) kept.push_back(static_cast<int64_t>(v[p]));
    }
    if (!kept.empty()) {
        auto t = torch::from_blob(kept.data(), {static_cast<int64_t>(kept.size())},
                                  torch::TensorOptions().dtype(torch::kLong)).clone();
        keep.index_put_({t}, true);
    }
    std::cerr << "[Density] keep_line_frac=" << keep_frac << " stride=" << k
              << " direction_groups=" << groups.size() << " line_kept=" << kept.size()
              << " / " << nl << std::endl;
    return keep;
}

// ---------------------------------------------------------------------------
// Numerical equivalence check (PHOTO_SLAM_FGM_EQCHECK=1).
//
// Both implementations are evaluated on the SAME parameter snapshot, each from
// a fresh graph with zeroed gradients. Reported per expression:
//   max|d loss|               - absolute difference of the loss value
//   max|d grad|/<|grad|ref>   - relative difference of the target gradient
//
// Acceptance: the changes are algebraic identities (same rows gathered in the
// same order, the same exp() result reused, "term + 0" dropped, amax/amin
// instead of max/min values). IEEE-754 therefore yields bitwise-identical
// results, and max|d| == 0 is the expected outcome; 1e-6 relative is kept only
// as an allowance for a different ATen reduction schedule. A larger difference
// means the "equivalent" claim is wrong.
// ---------------------------------------------------------------------------
void GaussianMapperLine::runEquivCheck(const std::string& tag, long long step)
{
    if (!gaussians_ || !gaussians_->xyz_.defined() || !gaussians_->is_line_.defined())
        return;
    const float w_pos = frozen_line_w_coherence_;
    const float w_ecc = frozen_line_w_ecc_;
    const float w_ori = frozen_line_w_ori_;

    auto probe = [&](const std::function<torch::Tensor()>& fn,
                     const std::vector<torch::Tensor>& params,
                     double& loss_out, std::vector<torch::Tensor>& grads_out) {
        gaussians_->optimizer_->zero_grad(true);
        auto l = fn();
        loss_out = l.item().toDouble();
        l.backward();
        grads_out.clear();
        for (const auto& p : params)
            grads_out.push_back(p.grad().defined() ? p.grad().detach().clone()
                                                   : torch::Tensor());
        gaussians_->optimizer_->zero_grad(true);
    };
    auto relDiff = [](const torch::Tensor& a, const torch::Tensor& b) {
        if (!a.defined() || !b.defined()) return std::nan("");
        auto d = (a - b).abs().max().item<double>();
        auto r = b.abs().max().item<double>();
        return r > 0 ? d / r : d;
    };

    struct Res { std::string name; double dl; double dg; };
    std::vector<Res> rows;
    auto both = [&](const char* name,
                    const std::function<torch::Tensor()>& f_ref,
                    const std::function<torch::Tensor()>& f_opt,
                    const std::vector<torch::Tensor>& params) {
        double lr = 0, lo = 0;
        std::vector<torch::Tensor> gr, go;
        probe(f_ref, params, lr, gr);
        probe(f_opt, params, lo, go);
        double dg = 0;
        for (std::size_t i = 0; i < gr.size(); ++i)
            dg = std::max(dg, relDiff(gr[i], go[i]));
        rows.push_back({name, std::abs(lr - lo), dg});
    };

    // Each lambda must build its OWN tensors: the graph is freed by the first
    // backward, so sharing a tensor between the two probes would fail.
    both("position", [&]{ return gaussians_->computeVectorizedLineLossRef(w_pos); },
                     [&]{ return gaussians_->computeVectorizedLineLoss(w_pos); },
         {gaussians_->xyz_});
    both("ecc",
         [&]{ return gaussians_->computeLineShapeConstraintRef(w_ecc, 0.0f); },
         [&]{ auto sa = gaussians_->getScaling();
              return gaussians_->computeLineShapeConstraint(w_ecc, 0.0f, sa); },
         {gaussians_->scaling_});
    both("ori",
         [&]{ return gaussians_->computeLineShapeConstraintRef(0.0f, w_ori); },
         [&]{ auto sa = gaussians_->getScaling();
              return gaussians_->computeLineShapeConstraint(0.0f, w_ori, sa); },
         {gaussians_->rotation_});
    both("all_struct",
         [&]{ return gaussians_->computeVectorizedLineLossRef(w_pos) +
                      gaussians_->computeLineShapeConstraintRef(w_ecc, w_ori); },
         [&]{ auto sa = gaussians_->getScaling();
              return gaussians_->computeVectorizedLineLoss(w_pos) +
                     gaussians_->computeLineShapeConstraint(w_ecc, w_ori, sa); },
         {gaussians_->xyz_, gaussians_->scaling_, gaussians_->rotation_});
    // the exact substitution used by the eccentricity term:
    //   exp(full).index_select(idx)  ==  exp(scaling[idx])
    both("exp_subset_substitution",
         [&]{ return torch::exp(gaussians_->scaling_.index({gaussians_->is_line_})).sum(); },
         [&]{ return torch::exp(gaussians_->scaling_)
                      .index_select(0, gaussians_->line_idx_).sum(); },
         {gaussians_->scaling_});

    std::ofstream out(frozen_ablation_dir_ /
                      (tag + "_eqcheck_step" + std::to_string(step) + ".csv"));
    out << "expression,max_abs_delta_loss,max_rel_delta_grad,verdict\n";
    bool ok = true;
    for (const auto& r : rows) {
        const bool pass = (r.dl == 0.0) && (r.dg <= 1e-6);
        if (!pass) ok = false;
        out << r.name << ',' << std::setprecision(12) << r.dl << ',' << r.dg << ','
            << (pass ? "PASS" : "FAIL") << '\n';
        std::cerr << "[EqCheck][" << tag << "] " << r.name
                  << " max|dloss|=" << r.dl << " max_reld|grad|=" << r.dg
                  << (pass ? "  PASS" : "  FAIL") << std::endl;
    }
    std::cerr << "[EqCheck][" << tag << "] step=" << step
              << " overall=" << (ok ? "PASS" : "FAIL") << std::endl;
    gaussians_->optimizer_->zero_grad(true);
}

// Render a FIXED set of views and report mean PSNR/SSIM plus the eval wall time.
// Guarantees:
//   * NoGradGuard -> no gradient accumulation, no optimizer interaction.
//   * The raw iteration_ counter is asserted unchanged by the caller.
//   * Run-time render buffers are snapshotted and restored.
//   * Caller re-checks paramFingerprint() to prove the parameters are untouched.
void GaussianMapperLine::evalAtFrozenStep(long long opt_steps, const std::string& tag,
                                         double& psnr_out, double& ssim_out,
                                         double& l1_out, double& struct_ori_out,
                                         double& struct_shape_out,
                                         double& eval_seconds_out)
{
    psnr_out = -1.0;
    ssim_out = -1.0;
    l1_out = -1.0;
    struct_ori_out = -1.0;
    struct_shape_out = -1.0;
    eval_seconds_out = 0.0;
    if (frozen_eval_kf_seq_.empty() || !gaussians_)
        return;

    std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
    std::vector<std::size_t> ids;
    views.reserve(scene_->keyframes().size());
    for (auto& kv : scene_->keyframes()) {
        views.push_back(kv.second);
        ids.push_back(kv.first);
    }

    const auto pdir = result_dir_ / ("eval_step" + std::to_string(opt_steps));
    std::filesystem::create_directories(pdir);
    const auto img_dir = pdir / "image";
    const auto gt_dir = pdir / "image_gt";
    const auto resid_dir = pdir / "image_residual";
    std::filesystem::create_directories(img_dir);
    std::filesystem::create_directories(gt_dir);
    std::filesystem::create_directories(resid_dir);
    std::ofstream pf(pdir / "per_frame.csv");
    pf << "step,variant,kf_index,kf_id,fid,psnr,ssim,l1\n";

    torch::NoGradGuard ng;

    torch::Tensor saved_radii, saved_hits, saved_alpha;
    const bool had_radii = gaussians_->max_radii2D_.defined();
    if (had_radii) saved_radii = gaussians_->max_radii2D_.clone();
    const bool had_hits = gaussians_->debug_hit_count_.defined();
    if (had_hits) saved_hits = gaussians_->debug_hit_count_.clone();
    const bool had_alpha = gaussians_->debug_alpha_accum_.defined();
    if (had_alpha) saved_alpha = gaussians_->debug_alpha_accum_.clone();

    auto syncDev = [&]() {
        if (device_type_ == torch::kCUDA && gaussians_->xyz_.defined() &&
            gaussians_->xyz_.numel() > 0)
            (void)gaussians_->xyz_.select(0, 0).sum().item<double>();
    };

    double sum_psnr = 0.0, sum_ssim = 0.0, sum_l1 = 0.0;
    std::size_t n = 0;
    syncDev();
    const auto t0 = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < frozen_eval_kf_seq_.size(); ++i) {
        const std::size_t idx = frozen_eval_kf_seq_[i];
        if (idx >= views.size() || !views[idx])
            continue;
        auto pkf = views[idx];
        auto render_pkg = GaussianRendererWithLine::renderWithLine(
            pkf, pkf->image_height_, pkf->image_width_, gaussians_,
            pipe_params_, background_, override_color_);
        torch::Tensor masked_image =
            std::get<0>(render_pkg) * undistort_mask_[pkf->camera_id_];
        auto gt_image = pkf->original_image_;
        const float s = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
        const float p = loss_utils::psnr(masked_image, gt_image).item().toFloat();
        const float l1 = loss_utils::l1_loss(masked_image, gt_image).item().toFloat();
        sum_psnr += static_cast<double>(p);
        sum_ssim += static_cast<double>(s);
        sum_l1 += static_cast<double>(l1);
        ++n;
        // Persist render + GT for offline metrics (LPIPS).
        {
            char name[32];
            std::snprintf(name, sizeof(name), "%06zu.png", static_cast<std::size_t>(pkf->fid_));
            fgmSaveImage(masked_image, img_dir / name);
            fgmSaveImage(gt_image, gt_dir / name);
            // Signed residual centred at 0.5: red = render brighter than GT.
            torch::Tensor resid = (0.5f + 2.0f * (masked_image - gt_image)).clamp(0.f, 1.f);
            fgmSaveImage(resid, resid_dir / name);
        }
        pf << opt_steps << ',' << tag << ',' << idx << ',' << ids[idx] << ','
           << pkf->fid_ << ',' << std::fixed << std::setprecision(8) << p << ','
           << s << ',' << l1 << '\n';
    }
    syncDev();
    eval_seconds_out = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - t0).count() / 1e6;

    if (n > 0) {
        psnr_out = sum_psnr / static_cast<double>(n);
        ssim_out = sum_ssim / static_cast<double>(n);
        l1_out = sum_l1 / static_cast<double>(n);
    }

    // Structural diagnostics measured on the CURRENT state with FIXED reference
    // weights, so variant B (structural loss disabled) and C (enabled) report
    // exactly the same quantity. This is a state measurement, not the training
    // loss of the variant.
    try {
        if (gaussians_->is_line_.defined() && gaussians_->is_line_.numel() > 0) {
            struct_ori_out =
                gaussians_->computeVectorizedLineLoss(1.0f).item().toFloat();
            if (frozen_line_w_ecc_ > 0.0f || frozen_line_w_ori_ > 0.0f)
                struct_shape_out = gaussians_->computeLineShapeConstraint(
                                       frozen_line_w_ecc_, frozen_line_w_ori_)
                                       .item().toFloat();
        }
    } catch (const std::exception& e) {
        std::cerr << "[FGM] structural diagnostic failed: " << e.what() << std::endl;
    }

    if (had_radii && saved_radii.defined()) gaussians_->max_radii2D_.copy_(saved_radii);
    if (had_hits && saved_hits.defined()) gaussians_->debug_hit_count_.copy_(saved_hits);
    if (had_alpha && saved_alpha.defined()) gaussians_->debug_alpha_accum_.copy_(saved_alpha);
    pf.close();
}

// Replay several mapper configurations on one frozen SLAM geometry.
//
// The SLAM side (trajectory, sparse point cloud, line samples, keyframe images)
// is frozen once. Each condition then rebuilds the Gaussian model from that
// exact same input, resets the iteration counter and the keyframe sampling
// budget, and runs the same refinement entry point for the same number of
// steps. Only the mapper-side switches differ:
//   A  point baseline: no line-seeded Gaussians, no line loss, ordinary densify
//   B  line sampling, no structural regularisation, ordinary densify
//   C  B + structural (line) regularisation, ordinary densify
//   D  C + directional (line-aware) densification
// B, C and D therefore share an identical initialisation and pruning strategy.
void GaussianMapperLine::runFrozenAblation()
{
    const char* dir_env = std::getenv("PHOTO_SLAM_FROZEN_ABLATION");
    frozen_ablation_dir_ = std::filesystem::path(dir_env);
    std::filesystem::create_directories(frozen_ablation_dir_);

    // Fixed optimization budget (defaults to Optimization.refined_gaussian_max_iter_num).
    frozen_ablation_steps_ = opt_params_.refined_gaussian_max_iter_num_;
    if (const char* s = std::getenv("PHOTO_SLAM_FROZEN_STEPS")) {
        const int v = std::atoi(s);
        if (v > 0)
            frozen_ablation_steps_ = v;
    }

    // Condition list, e.g. PHOTO_SLAM_FROZEN_CONDS="A,B,C,D".
    frozen_conditions_.clear();
    std::string conds = "A,B,C,D";
    if (const char* c = std::getenv("PHOTO_SLAM_FROZEN_CONDS"))
        conds = c;
    {
        std::string item;
        for (std::size_t i = 0; i <= conds.size(); ++i) {
            if (i < conds.size() && conds[i] != ',') {
                item += conds[i];
                continue;
            }
            const auto b = item.find_first_not_of(" \t");
            const auto e = item.find_last_not_of(" \t");
            item = (b == std::string::npos) ? std::string() : item.substr(b, e - b + 1);
            if (!item.empty()) {
                FrozenAblationCondition c;
                c.tag = item;
                // The switch letter is the FIRST character, so the same
                // configuration can be repeated in one process with distinct
                // tags ("B0,B1") for a same-scene repeat control.
                const char base = item[0];
                if (base == 'A') {
                    c.use_line_gaussians = false; c.line_loss = false; c.directional_densify = false;
                } else if (base == 'B') {
                    c.use_line_gaussians = true;  c.line_loss = false; c.directional_densify = false;
                } else if (base == 'C') {
                    c.use_line_gaussians = true;  c.line_loss = true;  c.directional_densify = false;
                } else if (base == 'D') {
                    c.use_line_gaussians = true;  c.line_loss = true;  c.directional_densify = true;
                } else if (base == 'E') {
                    // orientation term only
                    c.use_line_gaussians = true;  c.line_loss = true;
                    c.directional_densify = false;
                    c.w_position = 0.0f; c.w_ecc = 0.0f; c.w_orient = -1.0f;
                } else if (base == 'F') {
                    // eccentricity (elongation) term only
                    c.use_line_gaussians = true;  c.line_loss = true;
                    c.directional_densify = false;
                    c.w_position = 0.0f; c.w_ecc = -1.0f; c.w_orient = 0.0f;
                } else if (base == 'R' && item.size() > 1 &&
                           (item.substr(1) == "100" || item.substr(1) == "50" ||
                            item.substr(1) == "A" || item.substr(1) == "C")) {
                    // mapper-budget-v2-hash (R100/R50) and raster-perf-v5 (RA/RC).
                    // All of these: line structural loss OFF, no directional
                    // densify, no line-Gaussian subsetting. RA and RC differ ONLY
                    // in the rasteriser backward backend.
                    c.use_line_gaussians = true;  c.line_loss = false;
                    c.directional_densify = false;
                    const std::string sfx = item.substr(1);
                    if (sfx == "A")      c.raster_backend = 0;
                    else if (sfx == "C") c.raster_backend = 2;
                    else                 c.inactive_keep = (sfx == "100") ? 1.0f : 0.5f;
                } else if (base == 'Q') {
                    // Line-density conditions: ordinary Gaussians untouched,
                    // structural loss OFF, only the number of line Gaussians
                    // changes. Q0 / Q25 / Q50 / Q100.
                    c.use_line_gaussians = true;  c.line_loss = false;
                    c.directional_densify = false;
                    const std::string num = item.substr(1);
                    const int pct = num.empty() ? 100 : std::atoi(num.c_str());
                    c.keep_line_frac = static_cast<float>(pct) / 100.0f;
                } else if (base == 'R' && item.size() > 1 &&
                           (item.substr(1) == "100" || item.substr(1) == "50")) {
                    // mapper-budget-v2-hash: the ONE algorithmic variable.
                    // Both arms are identical except for the inactive-geo
                    // acceptance fraction. Line structural loss OFF, no
                    // directional densify, no line-Gaussian subsetting.
                    c.use_line_gaussians = true;  c.line_loss = false;
                    c.directional_densify = false;
                    c.inactive_keep = (item.substr(1) == "100") ? 1.0f : 0.5f;
                } else if (base == 'J' || base == 'K') {
                    // all three structural terms, implementation forced:
                    //   J = pre-optimisation (S-original), K = optimised (S-fast)
                    c.use_line_gaussians = true;  c.line_loss = true;
                    c.directional_densify = false;
                    c.force_ref = (base == 'J') ? 1 : 0;
                } else if (base == 'G') {
                    // perpendicular offset (position) term only
                    c.use_line_gaussians = true;  c.line_loss = true;
                    c.directional_densify = false;
                    c.w_position = -1.0f; c.w_ecc = 0.0f; c.w_orient = 0.0f;
                } else {
                    std::cerr << "[FrozenABL] unknown condition '" << item << "' - skipped." << std::endl;
                    item.clear();
                    continue;
                }
                frozen_conditions_.push_back(c);
            }
            item.clear();
        }
    }
    if (frozen_conditions_.empty()) {
        std::cerr << "[FrozenABL] no valid conditions - aborting." << std::endl;
        return;
    }
    if (scene_->keyframes().empty()) {
        std::cerr << "[FrozenABL] empty scene - aborting." << std::endl;
        return;
    }

    // ---- Freeze the geometry ONCE ----
    const std::map<point3D_id_t, Point3D> frozen_pcd = scene_->cached_point_cloud_;
    const float frozen_extent = scene_->cameras_extent_;
    std::size_t n_map_point = 0, n_line_sample = 0;
    for (const auto& kv : frozen_pcd) {
        if (kv.second.source_ == PointSourceType::LINE_SAMPLED)
            ++n_line_sample;
        else
            ++n_map_point;
    }
    frozen_line_w_coherence_ = opt_params_.weight_line_coherence_;
    frozen_line_w_ecc_ = opt_params_.weight_line_shape_ecc_;
    frozen_line_w_ori_ = opt_params_.weight_line_shape_ori_;

    std::cerr << "[FrozenABL] frozen scene: kfs=" << scene_->keyframes().size()
              << " map_points=" << n_map_point
              << " line_samples=" << n_line_sample
              << " cameras_extent=" << frozen_extent
              << " steps=" << frozen_ablation_steps_
              << " conditions=" << frozen_conditions_.size() << std::endl;

    const std::filesystem::path base_result_dir = result_dir_;

    // ---- Save the frozen snapshot + hashes (input identity) ----
    const std::string snap_text = frozenAblSerializeScene(
        *scene_, frozen_pcd, frozen_extent, n_map_point, n_line_sample);
    {
        std::ofstream snap(frozen_ablation_dir_ / "snapshot.txt");
        snap << snap_text;
    }
    const std::string image_key = frozenAblImageKey(*scene_);
    const std::string input_hash = frozenAblHashHex(snap_text);
    const std::string image_hash = frozenAblHashHex(image_key);
    std::cerr << "[FrozenABL] input_hash=" << input_hash
              << " image_hash=" << image_hash << std::endl;

    // ---- Fixed-geometry convergence experiment configuration ----
    // Count freeze: densify/clone/split/prune, opacity reset, inactive-geo
    // densify and repeated line sampling are all disabled, so the number of
    // Gaussians (total and line) is constant for the whole run.
    frozen_fixed_count_ = true;
    if (const char* fc = std::getenv("PHOTO_SLAM_FROZEN_FIXED_COUNT"))
        frozen_fixed_count_ = (std::atoi(fc) != 0);

    // Evaluation points expressed in COMPLETED OPTIMIZER STEPS.
    frozen_eval_steps_.clear();
    {
        std::string spec = "0,100,250,500,1000,2000,3000,5000";
        if (const char* es = std::getenv("PHOTO_SLAM_FROZEN_EVAL_STEPS"))
            spec = es;
        std::string item;
        auto flush = [&]() {
            if (item.find_first_not_of(" \t") != std::string::npos) {
                const long long v = std::atoll(item.c_str());
                if (v >= 0 && v <= frozen_ablation_steps_)
                    frozen_eval_steps_.push_back(v);
            }
            item.clear();
        };
        for (std::size_t i = 0; i <= spec.size(); ++i) {
            if (i < spec.size() && spec[i] != ',') {
                item += spec[i];
                continue;
            }
            flush();
        }
        std::sort(frozen_eval_steps_.begin(), frozen_eval_steps_.end());
        frozen_eval_steps_.erase(
            std::unique(frozen_eval_steps_.begin(), frozen_eval_steps_.end()),
            frozen_eval_steps_.end());
        if (frozen_eval_steps_.empty() ||
            frozen_eval_steps_.back() != static_cast<long long>(frozen_ablation_steps_))
            frozen_eval_steps_.push_back(
                static_cast<long long>(frozen_ablation_steps_));
    }

    // Fixed, shared evaluation views: deterministic, evenly spaced keyframes,
    // identical for every variant and every repeat.
    frozen_eval_kf_seq_.clear();
    {
        const std::size_t nkf = scene_->keyframes().size();
        int want = 24;
        if (const char* ne = std::getenv("PHOTO_SLAM_FROZEN_EVAL_VIEWS"))
            want = std::atoi(ne);
        if (want > 0 && nkf > 0) {
            const std::size_t k =
                std::min<std::size_t>(static_cast<std::size_t>(want), nkf);
            for (std::size_t i = 0; i < k; ++i)
                frozen_eval_kf_seq_.push_back((i * nkf) / k);
        }
    }
    {
        std::ofstream ef(frozen_ablation_dir_ / "eval_views.txt");
        ef << "# fixed evaluation keyframes (index into keyframe map order)\n";
        for (auto v : frozen_eval_kf_seq_)
            ef << v << '\n';
        std::ofstream sf(frozen_ablation_dir_ / "eval_steps.txt");
        sf << "# completed_optimizer_steps\n";
        for (auto v : frozen_eval_steps_)
            sf << v << '\n';
    }
    std::cerr << "[FrozenABL] fixed_count=" << (frozen_fixed_count_ ? 1 : 0)
              << " eval_views=" << frozen_eval_kf_seq_.size() << " eval_points=";
    for (auto v : frozen_eval_steps_)
        std::cerr << v << ' ';
    std::cerr << std::endl;

    // ---- Pre-generate the shared keyframe sampling sequence ----
    preGenerateFrozenKfSequence(frozen_ablation_steps_);
    const std::string seq_hash = frozenAblHashHex(frozenAblJoinSeq(frozen_kf_seq_));
    {
        std::ofstream sf(frozen_ablation_dir_ / "kf_sequence.txt");
        sf << "# steps=" << frozen_kf_seq_.size() << " hash=" << seq_hash << "\n";
        for (std::size_t i = 0; i < frozen_kf_seq_.size(); ++i)
            sf << i << ' ' << frozen_kf_seq_[i] << '\n';
    }
    std::cerr << "[FrozenABL] kf_sequence steps=" << frozen_kf_seq_.size()
              << " hash=" << seq_hash << std::endl;

    // ---- Canonical initialisation checkpoint, shared by every variant ----
    // Taken AFTER createFromPcd + trainingSetup and BEFORE any photometric
    // optimizer step, so it is the untrained initialisation. A plain PLY is not
    // enough: it cannot carry is_line_, line_dir_w_ or xyz_init_.
    //
    // If a checkpoint for THIS input already exists it is reused instead of being
    // rebuilt, so the two conditions can be run in SEPARATE processes and still
    // share one byte-identical initialisation. Identity of the input is checked
    // via input_hash before reuse.
    frozen_checkpoint_path_ = frozen_ablation_dir_ / "init_checkpoint.pt";
    const std::filesystem::path fp_path = frozen_ablation_dir_ / "init_fingerprint.txt";
    bool reused_checkpoint = false;
    {
        std::string rec_input_hash, rec_fp;
        long long rec_all = -1, rec_line = -1;
        if (std::filesystem::exists(frozen_checkpoint_path_) && std::filesystem::exists(fp_path)) {
            std::ifstream cf(fp_path);
            std::string k, v;
            while (cf >> k >> v) {
                if (k == "fingerprint") rec_fp = v;
                else if (k == "gaussians") rec_all = std::atoll(v.c_str());
                else if (k == "line_gaussians") rec_line = std::atoll(v.c_str());
                else if (k == "input_hash") rec_input_hash = v;
            }
            if (!rec_fp.empty() && rec_input_hash == input_hash) {
                frozen_ref_fingerprint_ = rec_fp;
                frozen_count_ref_all_ = rec_all;
                frozen_count_ref_line_ = rec_line;
                reused_checkpoint = true;
            } else {
                std::cerr << "[FrozenABL] existing checkpoint rejected "
                          << "(input_hash " << rec_input_hash << " vs " << input_hash
                          << ") - rebuilding." << std::endl;
            }
        }
    }
    if (!reused_checkpoint) {
        auto ref = std::make_shared<GaussianModelLine>(model_params_);
        ref->createFromPcd(frozen_pcd, frozen_extent);
        ref->trainingSetup(opt_params_);
        frozen_ref_fingerprint_ = ref->paramFingerprint();
        frozen_count_ref_all_ = static_cast<long long>(ref->getXYZ().size(0));
        frozen_count_ref_line_ =
            (ref->is_line_.defined() && ref->is_line_.numel() > 0)
                ? ref->is_line_.to(torch::kCPU).sum().item().toLong()
                : 0LL;
        ref->saveInitCheckpoint(frozen_checkpoint_path_);
        std::ofstream cf(fp_path);
        cf << "fingerprint " << frozen_ref_fingerprint_ << "\n"
           << "gaussians " << frozen_count_ref_all_ << "\n"
           << "line_gaussians " << frozen_count_ref_line_ << "\n"
           << "checkpoint " << frozen_checkpoint_path_.string() << "\n"
           << "input_hash " << input_hash << "\n"
           << "image_hash " << image_hash << "\n"
           << "kf_seq_hash " << seq_hash << "\n";
    }
    std::cerr << "[FrozenABL] init checkpoint=" << frozen_checkpoint_path_.string()
              << (reused_checkpoint ? " (REUSED)" : " (freshly built)")
              << " fingerprint=" << frozen_ref_fingerprint_
              << " gaussians=" << frozen_count_ref_all_
              << " line_gaussians=" << frozen_count_ref_line_ << std::endl;

    // ---- Condition order: rotated per repeat so no condition always goes first ----
    std::vector<FrozenAblationCondition> order;
    const int n_cond = static_cast<int>(frozen_conditions_.size());
    int rot = 0;
    if (const char* r = std::getenv("PHOTO_SLAM_FROZEN_ROTATE"))
        rot = std::atoi(r);
    rot = n_cond > 0 ? ((rot % n_cond) + n_cond) % n_cond : 0;
    for (int i = 0; i < n_cond; ++i)
        order.push_back(frozen_conditions_[(i + rot) % n_cond]);
    {
        std::string os;
        for (const auto& c : order)
            os += c.tag;
        std::cerr << "[FrozenABL] condition order=" << os << std::endl;
    }

    std::string cond_tags;
    for (const auto& c : frozen_conditions_)
        cond_tags += c.tag;
    const std::filesystem::path summary_path =
        frozen_ablation_dir_ / ("summary_" + cond_tags + ".csv");
    std::ofstream csv(summary_path);
    csv << "condition,use_line_gaussians,line_loss,directional_densify,psnr,ssim,"
           "gaussians_init,gaussians_final,gaussians_point,gaussians_line,"
           "opt_steps,refine_seconds,nan,grad_finite,exit,"
           "fixed_count,init_fingerprint,init_restore,init_fp_match,"
           "count_violation,eval_count,eval_params_stable,mapping_seconds,"
           "w_position_eff,w_ecc_eff,w_orient_eff,use_ref_struct_loss,"
           "keep_line_frac,nonline_fp,"
           "input_hash,image_hash,kf_seq_hash\n";

    for (const auto& c : order) {
        std::cerr << "\n[FrozenABL] ===== condition " << c.tag << " =====" << std::endl;

        double seconds = -1.0;
        double psnr = -1.0, ssim = -1.0;
        long long n_gauss_init = 0, n_gauss_final = 0, n_pt_gauss = 0, n_line_gauss = 0;
        long long opt_steps = 0;
        int has_nan = -1;
        int grad_finite = -1;
        int exit_code = 0;
        // --- fixed-geometry convergence experiment, per-condition state ---
        double mapping_seconds = 0.0;
        int init_restore = 0;
        int init_fp_match = -1;
        std::string init_fingerprint;
        long long eval_count = 0;
        int eval_params_stable = 1;
        std::vector<std::vector<double>> eval_rows;
        float w_position_eff = -1.0f, w_ecc_eff = -1.0f, w_orient_eff = -1.0f;
        // Non-line fingerprint: proves the ordinary Gaussians are identical
        // across the line-density conditions.
        std::string nonline_fp;
        long long n_line_init = 0;

        try {
            // Identical input, minus the line-sampled seeds for the point baseline.
            std::map<point3D_id_t, Point3D> pcd;
            for (const auto& kv : frozen_pcd) {
                if (!c.use_line_gaussians && kv.second.source_ == PointSourceType::LINE_SAMPLED)
                    continue;
                pcd.emplace(kv.first, kv.second);
            }

            // Fresh model/optimiser, exactly as the normal first-time setup does.
            gaussians_.reset();
            gaussians_ = std::make_shared<GaussianModelLine>(model_params_);
            gaussians_->createFromPcd(pcd, frozen_extent);
            gaussians_->trainingSetup(opt_params_);
            n_gauss_init = gaussians_->getXYZ().size(0);

            // Restore the SAME initialisation checkpoint. The fingerprint covers
            // xyz/opacity/scaling/rotation/SH AND is_line_/line_dir_w_/xyz_init_
            // plus the Gaussian counts, which a plain PLY cannot express.
            if (n_gauss_init == frozen_count_ref_all_) {
                init_restore = static_cast<int>(
                    gaussians_->restoreInitCheckpoint(frozen_checkpoint_path_));
            } else {
                std::cerr << "[FrozenABL][" << c.tag
                          << "] cloud differs from the checkpoint reference ("
                          << n_gauss_init << " vs " << frozen_count_ref_all_
                          << ") - checkpoint NOT applied." << std::endl;
            }
            // Line-density condition: drop rows from the SAME restored
            // checkpoint. Ordinary Gaussians keep their exact initial
            // parameters; only line rows are removed. A fresh optimizer is then
            // built over the filtered tensors.
            if (c.keep_line_frac < 1.0f) {
                auto keepmask = makeLineSubsetMask(c.keep_line_frac);
                gaussians_->keepSubset(keepmask);
                gaussians_->trainingSetup(opt_params_);
                n_gauss_init = gaussians_->getXYZ().size(0);
            }
            nonline_fp = gaussians_->paramFingerprintNonLine();
            // The count reference is the CONDITION's own post-subset count, so
            // count_violation means "the count changed during this condition"
            // rather than "this condition has fewer line Gaussians than Q100".
            frozen_count_ref_all_ = static_cast<long long>(gaussians_->xyz_.size(0));
            frozen_count_ref_line_ =
                (gaussians_->is_line_.defined() && gaussians_->is_line_.numel() > 0)
                    ? gaussians_->is_line_.to(torch::kCPU).sum().item().toLong() : 0LL;
            frozen_count_violation_ = 0;
            frozen_eval_seconds_total_ = 0.0;
            frozen_use_ref_ = c.force_ref;
            init_fingerprint = gaussians_->paramFingerprint();
            init_fp_match = (init_fingerprint == frozen_ref_fingerprint_) ? 1 : 0;
            if (!init_fp_match)
                std::cerr << "[FrozenABL][" << c.tag
                          << "] INIT FINGERPRINT MISMATCH: " << init_fingerprint
                          << " != " << frozen_ref_fingerprint_ << std::endl;
            assertFrozenCounts("after-restore");

            // Mapper-side switches. `w_*` < 0 means "configuration value", which
            // keeps the reported weights byte-identical to the config.
            auto effW = [](float override_v, float cfg_v) {
                return override_v >= 0.0f ? override_v : cfg_v;
            };
            opt_params_.weight_line_coherence_ =
                c.line_loss ? effW(c.w_position, frozen_line_w_coherence_) : 0.0f;
            opt_params_.weight_line_shape_ecc_ =
                c.line_loss ? effW(c.w_ecc, frozen_line_w_ecc_) : 0.0f;
            opt_params_.weight_line_shape_ori_ =
                c.line_loss ? effW(c.w_orient, frozen_line_w_ori_) : 0.0f;
            w_position_eff = opt_params_.weight_line_coherence_;
            w_ecc_eff = opt_params_.weight_line_shape_ecc_;
            w_orient_eff = opt_params_.weight_line_shape_ori_;
            ::setenv("PHOTO_SLAM_ABL_DIRECTIONAL_DENSIFY", c.directional_densify ? "1" : "0", 1);

            // Identical iteration counter, keyframe sampling budget and
            // sampling sequence for every condition and every repeat.
            // Mapper-level per-run schedule state MUST be reset too, otherwise
            // the second condition in a process starts from the first one's
            // schedule: default_sh_ ramps 0->3 over 5000 steps and is not a
            // member of the model, so it survived gaussians_.reset().
            iteration_ = 0;
            frozen_opt_steps_ = 0;
            default_sh_ = 0;
            ema_loss_for_log_ = 0.0f;
            total_training_time_ms_ = 0.0;
            frozen_train_views_flat_.clear();
            frozen_kf_replay_ = true;
            frozen_kf_step_ = 0;
            replayInitInactiveDriver();   // mapper-budget-v2-hash: reset per condition
            // Per-keyframe training state also lives on the keyframes, which are
            // shared by all conditions. getCurrentGausPyramidLevel() CONSUMES
            // gaus_pyramid_times_of_use_ (15 uses per sub-level per keyframe), so
            // without this reset the second condition would already have every
            // pyramid budget exhausted and would train at full resolution while
            // the first one trained on downsampled images.
            for (auto& kfit : scene_->keyframes()) {
                kfit.second->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
                kfit.second->tracking_error_ = 1.0f;
            }
            for (auto& kfit : scene_->keyframes())
                kfit.second->remaining_times_of_use_ = 1000;
            kfs_used_times_.clear();
            {
                int seq_seed = 12345;
                if (const char* e = std::getenv("PHOTO_SLAM_FROZEN_SEQ_SEED"))
                    seq_seed = std::atoi(e);
                std::srand(static_cast<unsigned>(seq_seed));
            }

            result_dir_ = base_result_dir / c.tag;
            std::filesystem::create_directories(result_dir_);
            // mapper-budget-v2-hash: per-condition budget + its own audit files.
            budgetSetCondition(c.inactive_keep, result_dir_.string());
            // raster-perf-v5: pin the rasteriser backward backend for this arm.
            if (c.raster_backend >= 0) SetRasterBackend(c.raster_backend);

            // Train wall time covers exactly the optimizer steps (boundaries
            // synchronised so pending CUDA work is not billed to metrics).
            auto syncDevice = [&]() {
                if (device_type_ == torch::kCUDA && gaussians_ && gaussians_->xyz_.defined()
                    && gaussians_->xyz_.numel() > 0)
                    (void)gaussians_->xyz_.select(0, 0).sum().item<double>();
            };
            syncDevice();

            // Evaluations sit at fixed COMPLETED-OPTIMIZER-STEP boundaries. Eval
            // is a pure measurement: parameters and the raw iteration counter are
            // re-checked around every eval point.
            std::size_t next_eval = 0;
            auto runDueEvals = [&]() {
                while (next_eval < frozen_eval_steps_.size() &&
                       frozen_eval_steps_[next_eval] == frozen_opt_steps_) {
                    const long long it_before = getIteration();
                    const std::string fp_before = gaussians_->paramFingerprint();
                    double ep = -1.0, es = -1.0, el1 = -1.0, eori = -1.0, eshape = -1.0, et = 0.0;
                    evalAtFrozenStep(frozen_opt_steps_, c.tag, ep, es, el1, eori, eshape, et);
                    diagAtFrozenStep(frozen_opt_steps_, c.tag);
                    if (std::getenv("PHOTO_SLAM_FGM_MICROBENCH"))
                        runMicroBench(c.tag, frozen_opt_steps_);
                    if (std::getenv("PHOTO_SLAM_FGM_BWD_DECOMP"))
                        runBackwardDecomposition(c.tag, frozen_opt_steps_);
                    if (std::getenv("PHOTO_SLAM_FGM_EQCHECK"))
                        runEquivCheck(c.tag, frozen_opt_steps_);
                    const int fp_stable = (fp_before == gaussians_->paramFingerprint()) ? 1 : 0;
                    const int it_stable = (getIteration() == it_before) ? 1 : 0;
                    frozen_eval_seconds_total_ += et;
                    const long long n_all = static_cast<long long>(gaussians_->getXYZ().size(0));
                    const long long n_line = (gaussians_->is_line_.defined() &&
                                              gaussians_->is_line_.numel() > 0)
                                                 ? gaussians_->is_line_.to(torch::kCPU).sum().item().toLong()
                                                 : 0LL;
                    const int nan_free =
                        torch::isfinite(gaussians_->getXYZ()).all().item<bool>() ? 1 : 0;
                    // Schedule-identity witness: remaining pyramid budget over all
                    // keyframes must match between conditions at every eval point.
                    long long pyr_remaining = 0;
                    for (const auto& kfit : scene_->keyframes())
                        for (int v : kfit.second->gaus_pyramid_times_of_use_)
                            pyr_remaining += v;
                    eval_rows.push_back({static_cast<double>(frozen_opt_steps_),
                                         static_cast<double>(getIteration()),
                                         mapping_seconds, frozen_eval_seconds_total_, et,
                                         ep, es, el1, eori, eshape,
                                         static_cast<double>(fp_stable),
                                         static_cast<double>(it_stable),
                                         static_cast<double>(nan_free),
                                         static_cast<double>(n_all),
                                         static_cast<double>(n_line),
                                         static_cast<double>(default_sh_),
                                         static_cast<double>(pyr_remaining)});
                    if (!fp_stable || !it_stable)
                        eval_params_stable = 0;
                    ++eval_count;
                    std::cerr << "[FrozenABL][" << c.tag << "] eval step=" << frozen_opt_steps_
                              << " psnr=" << ep << " ssim=" << es << " l1=" << el1
                              << " struct_ori=" << eori << " struct_shape=" << eshape
                              << " map_s=" << mapping_seconds << " eval_s=" << et
                              << " params_unchanged=" << fp_stable
                              << " iteration_unchanged=" << it_stable << std::endl;
                    ++next_eval;
                }
            };
            // Step-0 eval BEFORE any optimizer update: identical numbers here are
            // the direct proof that both variants start from the same state.
            runDueEvals();

            const auto t0 = std::chrono::steady_clock::now();
            g_gpu_prof.resetAcc(prof_refine_phase_);
            while (frozen_opt_steps_ < frozen_ablation_steps_ && !isStopped()) {
                syncDevice();
                const auto t_step0 = std::chrono::steady_clock::now();
                // mapper-budget-v2-hash: supply the external geometry stream (the
                // inactive-geo补点 entry) deterministically BEFORE the step, so the
                // accepted points take part in that step's photometric update.
                // Identical order and payload in every arm; no SLAM KF thread.
                replayDriveInactiveGeo();
                trainForOneIterationErrorGuided();
                syncDevice();
                mapping_seconds +=
                    std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - t_step0).count() / 1e6;
                assertFrozenCounts("training-step");
                runDueEvals();
                if (frozen_opt_steps_ % 2000 == 0 && frozen_opt_steps_ > 0)
                    std::cerr << "[FrozenABL][" << c.tag << "] opt_steps "
                              << frozen_opt_steps_ << " / " << frozen_ablation_steps_ << std::endl;
            }
            syncDevice();
            seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count() / 1000.0;
            opt_steps = frozen_opt_steps_;

            // Per-condition GPU section breakdown (needs PHOTO_SLAM_GPU_PROF=1).
            if (std::getenv("PHOTO_SLAM_GPU_PROF")) {
                std::cerr << "[FGM][" << c.tag << "] per-section profile:" << std::endl;
                g_gpu_prof.print();
            }

            // Per-eval-point metrics: supports both the equal-step and the
            // equal-mapping-time comparison.
            {
                std::ofstream mf(frozen_ablation_dir_ / (c.tag + "_metrics.csv"));
                mf << "condition,completed_optimizer_steps,raw_iteration,"
                      "cum_mapping_seconds,cum_eval_seconds,eval_seconds,"
                      "eval_psnr,eval_ssim,eval_l1,struct_ori,struct_shape,"
                      "params_unchanged,iteration_unchanged,params_finite,"
                      "total_gaussians,line_gaussians,default_sh,pyr_remaining_uses\n";
                for (const auto& r : eval_rows) {
                    mf << c.tag;
                    for (double v : r)
                        mf << ',' << std::setprecision(10) << v;
                    mf << '\n';
                }
            }

            // The views ACTUALLY trained on, from the live selection path.
            {
                std::ofstream tv(frozen_ablation_dir_ / (c.tag + "_train_views.csv"));
                tv << "condition,completed_optimizer_steps,kf_index,kf_fid\n";
                for (std::size_t i = 0; i + 2 < frozen_train_views_flat_.size(); i += 3)
                    tv << c.tag << ',' << frozen_train_views_flat_[i] << ','
                       << frozen_train_views_flat_[i + 1] << ','
                       << frozen_train_views_flat_[i + 2] << '\n';
            }

            renderAndRecordAllKeyframes("_shutdown");
            savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");

            n_gauss_final = gaussians_->getXYZ().size(0);
            n_line_gauss = gaussians_->is_line_.defined()
                               ? gaussians_->is_line_.sum().item<int64_t>() : 0;
            n_pt_gauss = n_gauss_final - n_line_gauss;
            has_nan = torch::isnan(gaussians_->getXYZ()).any().item<bool>() ? 1 : 0;
            grad_finite = (gaussians_->xyz_gradient_accum_.defined()
                           && torch::isfinite(gaussians_->xyz_gradient_accum_).all().item<bool>()) ? 1 : 0;
            {
                const auto pdir = result_dir_ / (std::to_string(getIteration()) + "_shutdown");
                psnr = frozenAblMeanValueColumn(pdir / "psnr.txt");
                ssim = frozenAblMeanValueColumn(pdir / "dssim.txt");
            }
        }
        catch (const c10::Error& e) {
            std::cerr << "[FrozenABL][" << c.tag << "] TORCH ERROR: " << e.msg() << std::endl;
            exit_code = 3;
        }
        catch (const std::exception& e) {
            std::cerr << "[FrozenABL][" << c.tag << "] EXCEPTION: " << e.what() << std::endl;
            exit_code = 1;
        }
        catch (...) {
            std::cerr << "[FrozenABL][" << c.tag << "] UNKNOWN EXCEPTION" << std::endl;
            exit_code = 2;
        }

        std::cerr << "[FrozenABL][" << c.tag << "] psnr=" << psnr
                  << " ssim=" << ssim
                  << " gaussians=" << n_gauss_final
                  << " point=" << n_pt_gauss
                  << " line=" << n_line_gauss
                  << " opt_steps=" << opt_steps
                  << " seconds=" << seconds
                  << " nan=" << has_nan
                  << " grad_finite=" << grad_finite
                  << " exit=" << exit_code << std::endl;
        csv << c.tag << ',' << (c.use_line_gaussians ? 1 : 0) << ','
            << (c.line_loss ? 1 : 0) << ',' << (c.directional_densify ? 1 : 0) << ','
            << psnr << ',' << ssim << ','
            << n_gauss_init << ',' << n_gauss_final << ','
            << n_pt_gauss << ',' << n_line_gauss << ','
            << opt_steps << ',' << seconds << ','
            << has_nan << ',' << grad_finite << ',' << exit_code << ','
            << (frozen_fixed_count_ ? 1 : 0) << ',' << init_fingerprint << ','
            << init_restore << ',' << init_fp_match << ','
            << frozen_count_violation_ << ',' << eval_count << ','
            << eval_params_stable << ',' << mapping_seconds << ','
            << w_position_eff << ',' << w_ecc_eff << ',' << w_orient_eff << ','
            << (useRefStructLoss() ? 1 : 0) << ','
            << c.keep_line_frac << ',' << nonline_fp << ','
            << input_hash << ',' << image_hash << ',' << seq_hash << '\n';
        csv.flush();
    }

    ::unsetenv("PHOTO_SLAM_ABL_DIRECTIONAL_DENSIFY");
    frozen_kf_replay_ = false;
    result_dir_ = base_result_dir;
    std::cerr << "[FrozenABL] all conditions finished; summary: "
              << summary_path.string() << std::endl;
}

// gaussian_mapper_line.cpp
std::map<point3D_id_t, int> GaussianMapperLine::buildPntIdMap()
{
    std::map<point3D_id_t, int> id_map;
    
    // 将 GPU 上的 ID 下载到 CPU 进行遍历
    // 注意：如果是频繁调用，可以考虑在 GPU 上直接处理，但 std::map 只能在 CPU 维护
    torch::Tensor ids_cpu = gaussians_->point_ids_.to(torch::kCPU);
    auto accessor = ids_cpu.accessor<int64_t, 1>();

    for (int i = 0; i < ids_cpu.size(0); ++i) {
        // point3D_id_t 是 unsigned long
        id_map[static_cast<point3D_id_t>(accessor[i])] = i;
    }
    
    return id_map;
}

void GaussianMapperLine::trainColmap()
{
    // Prepare multi resolution images for training
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());
        if (device_type_ == torch::kCUDA) {
            cv::cuda::GpuMat img_gpu;
            img_gpu.upload(pkf->img_undist_);
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::cuda::GpuMat img_resized;
                cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
            }
        }
        else {
            pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
            for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
                cv::Mat img_resized;
                cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
                pkf->gaus_pyramid_original_image_[l] =
                    tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
            }
        }
    }

    // Prepare for training
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        scene_->cameras_extent_ = std::get<1>(scene_->getNerfppNorm());
        gaussians_->createFromPcd(scene_->cached_point_cloud_, scene_->cameras_extent_);
        std::unique_lock<std::mutex> lock(mutex_settings_);
        gaussians_->trainingSetup(opt_params_);
        this->initial_mapped_ = true;
    }

    // Main loop: gaussian splatting training
    while (!isStopped()) {
        // Invoke training once
        trainForOneIteration();

        if (getIteration() >= opt_params_.iterations_)
            break;
    }

    // Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Save and clear
    renderAndRecordAllKeyframes("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");

    signalStop();
}

// Error-guided training 加权随机采样（Roulette Wheel Selection / 轮盘赌算法），让 L1 Loss 更大（即 PSNR 更低）的图像拥有更大的概率被选中
void GaussianMapperLine::trainForOneIterationErrorGuided()
{
    const auto budget_iter_start = std::chrono::steady_clock::now();
    increaseIteration(1);
    maybeSnapshotAndAudit();   // mapper-budget-v1: full snapshots at fixed steps
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframeLine> viewpoint_cam = useOneTrainingErrorSlidingWindowKeyframe();

    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    // Per-iteration file I/O is expensive; log keyframe usage periodically
    // (final state is still written at shutdown).
    if (getIteration() % 200 == 0)
        writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    g_gpu_prof.markStart();
    auto render_pkg = GaussianRendererWithLine::renderWithLine(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);
    g_gpu_prof.markForwardEnd();

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
    
    ////added by zdg(同一个线段上的高斯中心（XYZ）应该落在该线段的轴线上)对于属于同一条线的点 Pi​，已知其线方向单位向量为 d，线段上的一点（例如质心）为 Pcenter​
    //torch::Tensor loss_line = gaussians_->computeLineCoherenceLoss(0.1f); // 权重 0.1
    //auto loss_line_shape = gaussians_->computeLineShapeConstraint(0.05f, 0.1f); // 形状约束
    //loss = loss + loss_line + loss_line_shape;  //modified by zdg

    // ==============================================================================
    // 🌟 [新增] 各向异性惩罚 (Anisotropy Penalty)
    // ==============================================================================
    // 限制高斯球的最大缩放比例。通常对于 Line Gaussians 设为 500-1000 较安全。
    // ==============================================================================
    // 使用从配置文件读取的参数
    float max_aniso_th = opt_params_.max_anisotropy_threshold_; 
    float lambda_aniso = opt_params_.lambda_anisotropy_; 

    // Structural weights are read here so the scale activation can be shared
    // between the anisotropy penalty and the eccentricity term below.
    float w_ori = opt_params_.weight_line_coherence_;
    float w_shape_ecc = opt_params_.weight_line_shape_ecc_;
    float w_shape_ori = opt_params_.weight_line_shape_ori_;

    // Stage-1: exp(scaling_) is needed by the anisotropy penalty AND by the
    // eccentricity term. Build it at most ONCE per iteration and pass it down,
    // instead of letting each term construct its own activation. When neither
    // term is active nothing is built at all.
    torch::Tensor scaling_act;
    const bool need_scale_act = (lambda_aniso > 0.0f) || (w_shape_ecc > 0.0f);
    if (need_scale_act)
        scaling_act = gaussians_->getScaling();

    if (lambda_aniso > 0.0f) {
        // NOTE: max(dim)/min(dim) VALUES are used, NOT amax/amin. The values are
        // identical but the GRADIENTS are not: max(dim) routes the gradient to
        // the single argmax, while amax spreads it over every tied extremum.
        // Measured in this LibTorch (2.12) with the real initialisation pattern
        // (scale_axis1 == scale_axis2 exactly): relative gradient difference
        // 0.5. Reverted deliberately - the substitution changed training
        // semantics. See reports/perf_stage2_report.md section 1.
        auto max_res = torch::max(scaling_act, /*dim=*/1);
        torch::Tensor max_s = std::get<0>(max_res);
        auto min_res = torch::min(scaling_act, /*dim=*/1);
        torch::Tensor min_s = std::get<0>(min_res);
        torch::Tensor ratio = max_s / (min_s + 1e-6);
        torch::Tensor loss_aniso =
            torch::pow(torch::clamp(ratio - max_aniso_th, /*min=*/0), 2).mean();
        loss = loss + (lambda_aniso * loss_aniso);
    }
    // ==============================================================================

    // ==============================================================================
    // 🌟 核心改进：引入结构正则化 L_str (Equation 9 in your paper) 及其消融开关
    // ==============================================================================
    // 从配置文件中读取权重，而不是硬编码 0.1f 或 0.05f
    // 如果权重为 0，则代表正在做 "关闭 L_str" 的消融实验 (Ablation Study)
    // w_ori / w_shape_ecc / w_shape_ori are read above, next to the anisotropy
    // penalty, so one scale activation can serve both.

    // Formula-preserving: all three terms are individually weighted inside the
    // callees, so the block must also be entered when ONLY the orientation weight
    // is non-zero (otherwise an orientation-only ablation would silently be a
    // no-op and would be misread as "the term has no effect").
    if (lineLossEvery() > 0 &&
        (w_ori > 0.0f || w_shape_ecc > 0.0f || w_shape_ori > 0.0f)
        && (getIteration() < 100 || getIteration() % lineLossEvery() == 0)) {
        torch::Tensor loss_str = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device_type_));
        
        // 方向一致性损失 L_ori
        if (w_ori > 0.0f) {
            // 建议：将权重乘法移到函数外部，让函数只返回纯粹的 Loss 标量，更符合 PyTorch 规范
            //loss_str = loss_str +  gaussians_->computeLineCoherenceLoss(w_ori);
            loss_str = loss_str + (useRefStructLoss()
                                       ? gaussians_->computeVectorizedLineLossRef(w_ori)
                                       : gaussians_->computeVectorizedLineLoss(w_ori));
        }
        // 形状各向异性约束 L_ecc
        // Formula-preserving decoupling: each term is already multiplied by its
        // own weight inside computeLineShapeConstraint(), so calling it with only
        // one weight non-zero changes no expression - it only allows the
        // eccentricity and orientation terms to be ablated independently. The
        // previous `&&` gate made them inseparable.
        if (w_shape_ecc > 0.0f || w_shape_ori > 0.0f) {
            loss_str = loss_str + (useRefStructLoss()
                                       ? gaussians_->computeLineShapeConstraintRef(w_shape_ecc, w_shape_ori)
                                       : gaussians_->computeLineShapeConstraint(
                                             w_shape_ecc, w_shape_ori, scaling_act));
        }
        
        loss = loss + loss_str;
    }

    g_gpu_prof.markLineLossEnd();

    loss.backward();
    if (recovery_diag_hook_) recovery_diag_hook_("after_backward");
    g_gpu_prof.markBackwardEnd();

    // ==============================================================================
    // 🌟 性能改进：DEBUG 代码必须用宏或 Flag 包裹，否则严重拖慢帧率
    // ==============================================================================
    
#ifdef DEBUG_LINE_GRADIENTS
    // 【诊断代码】
    if (getIteration() % 10 == 0) {
        // 1. 检查 xyz 是否开启了梯度需求
        bool req = gaussians_->getXYZ().requires_grad();
    
        // 2. 检查梯度是否计算出来了 (grad norm)
        float grad_norm = 0.0f;
        if (gaussians_->getXYZ().grad().defined()) {
            grad_norm = gaussians_->getXYZ().grad().norm().item<float>();
        }

        // 3. 检查参数指针是否一致
        // 获取优化器里存的参数指针
        auto opt_param = gaussians_->optimizer_->param_groups()[0].params()[0]; 
        // 获取模型里存的参数指针
        auto model_param = gaussians_->getXYZ();
        // 比较内存地址 (data_ptr)
        bool ptr_match = (opt_param.data_ptr() == model_param.data_ptr());

        std::cerr << "[DEBUG Grad] Iter:" << getIteration() 
              << " | ReqGrad: " << req 
              << " | GradNorm: " << grad_norm 
              << " | PtrMatch: " << (ptr_match ? "YES" : "NO!!!") << std::endl;
    }
#endif

    // 这一步同步是为了确保 backward 彻底完成，但 loss.item() 下方已经强制同步，故移除冗余的整卡同步。
    {
        torch::NoGradGuard no_grad;
        // Reduce per-iteration CPU-GPU syncs: EMA/tracking-error are only
        // needed for logging and error-guided keyframe selection, which do
        // not require per-iteration freshness (error changes slowly).
        if (getIteration() < 100 || getIteration() % 10 == 0) {
            ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;
            viewpoint_cam->tracking_error_ = Ll1.item().toFloat();
        }

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // Densification (frozen-count experiment: count must not change)
        // Append phase: this gate disables the ACTUAL densify/clone/split/
        // prune path, extra anisotropic/tiny prunes and opacity reset.
        if (!frozen_fixed_count_ && !append_phase_ &&
            getIteration() < opt_params_.densify_until_iter_) {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
                gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval()== 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
                //gaussians_->densifyAndPrune(
                //    densifyGradThreshold(),
                //    densify_min_opacity_,//0.005,//
                //    scene_->cameras_extent_,
                //    size_threshold
                //);
                gaussians_->densifyAndPruneWithLineAwareness(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold
                );

                // ==============================================================================
                // 🌟 [新增] 强制剔除极致拉伸点 (Post-Pruning)
                // ==============================================================================
                // 在 densify 结束后，立即扫描并干掉由于梯度更新意外拉伸过长的“针”
                if (getIteration() > opt_params_.densify_from_iter_) {
                    // 使用从配置文件读取的参数
                    float absolute_max_ratio = opt_params_.absolute_max_ratio_; // 物理极限，超过此比例必是伪影
                    gaussians_->pruneExceedinglyAnisotropic(absolute_max_ratio);
                }

                // 🌟 手段 C：强制剔除“极小点” (Min-Scaling Pruning)
                // 很多刺眼的点是因为 Scaling 极小但 Opacity 极高。
                // 在这里调用一个自定义函数来清理它们（见下文）
                gaussians_->pruneTinyGaussians(1e-6f);
            }

            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }
        g_gpu_prof.markDensifyEnd();

        // ==========================================================
        // 🌟 [新增代码]：补全时间结算与进度报告
        // ==========================================================
        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        total_training_time_ms_ += iter_time;

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
        {
            // GaussianTrainerLine::trainingReport(
            //     getIteration(),
            //     opt_params_.iterations_,
            //     Ll1,
            //     loss,
            //     ema_loss_for_log_,
            //     loss_utils::l1_loss,
            //     iter_time,
            //     *gaussians_,
            //     *scene_,
            //     pipe_params_,
            //     background_
            // );

            writeTrainingLog(iter_time, ema_loss_for_log_, gaussians_->xyz_.size(0));
        }
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            writeAdamAudit("step_entry");
            if (append_phase_ && append_diag_mode_ &&
                append_completed_steps_ == 0)
                appendRecordFirstStepAudit("first_step_before");
            gaussians_->optimizer_->step();
            ++frozen_opt_steps_;   // real optimizer-step counter (ablation)
            if (append_phase_) {
                ++append_completed_steps_;
                if (append_diag_mode_ && append_config_.steps > 0 &&
                    append_completed_steps_ == append_config_.steps)
                    appendRecordFirstStepAudit("last_step_after");
            }
            gaussians_->optimizer_->zero_grad(true);
            // Keep rotation quaternions unit-norm: the raw rotation_ norm is
            // unconstrained (losses only see the normalized quaternion), so it
            // can drift to zero and degenerate the normalize() backward.
            // Append phase normalizes in place so the optimizer parameter and
            // Adam moments are not replaced/reset during the measured window.
            if (getIteration() % 50 == 0)
                gaussians_->normalizeRotationQuaternions(append_phase_);
        }
        writeAdamAudit("after_update");
        if (recovery_diag_hook_) recovery_diag_hook_("after_update");
        g_gpu_prof.markOptimizerEnd();
    }
    g_gpu_prof.endIteration(prof_refine_phase_);

    // Unified cumulative-training-time boundary (see GaussianMapper).
    if (time_budget_mode_) {
        if (device_type_ == torch::kCUDA)
            torch::cuda::synchronize();
        cumulative_train_seconds_ +=
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - budget_iter_start).count();
    }
}

// ---------------------------------------------------------------------------
// Append refinement (line build)
// ---------------------------------------------------------------------------

bool GaussianMapperLine::appendRefinementEnabled()
{
    append_config_ = append_refinement::readConfig();
    time_budget_config_ = append_refinement::readTimeBudget();
    time_budget_mode_ = time_budget_config_.enabled();
    if (const char* e = std::getenv("PHOTO_SLAM_APPEND_DIAG"))
        append_diag_mode_ = (std::atoi(e) != 0);
    else
        append_diag_mode_ = false;
    // Budget mode is primary; PHOTO_SLAM_APPEND_STEPS becomes a safety cap.
    return append_config_.enabled() || time_budget_config_.enabled();
}

long long GaussianMapperLine::appendValidatedCount()
{
    if (!gaussians_ || !gaussians_->xyz_.defined())
        return -1;
    return static_cast<long long>(gaussians_->xyz_.size(0));
}

long long GaussianMapperLine::appendValidatedLineCount()
{
    if (!gaussians_ || !gaussians_->is_line_.defined() ||
        gaussians_->is_line_.numel() == 0)
        return 0;
    return static_cast<long long>(
        gaussians_->is_line_.to(torch::kCPU).sum().item<int64_t>());
}

bool GaussianMapperLine::appendPreflight(std::string& reason,
                                         int& drained_batches)
{
    if (!pSLAM_) {
        reason = "no_pSLAM";
        return false;
    }
    if (!pSLAM_->isShutDown()) {
        reason = "slam_not_shutdown";
        return false;
    }
    auto has_ops = [this]() {
        return pSLAM_->getAtlas()->hasMappingOperation();
    };
    auto drain_one = [this]() {
        combineMappingOperations_withLine();
    };
    const auto queue_start = std::chrono::steady_clock::now();
    const bool queue_ok = append_refinement::drainAndSettleQueue(
        has_ops, drain_one,
        append_config_.queue_timeout_ms,
        append_config_.queue_quiet_ms,
        drained_batches, reason);
    append_queue_seconds_ =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::steady_clock::now() - queue_start).count();
    if (!queue_ok)
        return false;
    if (!pSLAM_->isShutDown()) {
        reason = "slam_shutdown_revoked";
        return false;
    }
    if (scene_->keyframes().empty()) {
        reason = "empty_scene";
        return false;
    }
    return true;
}

void GaussianMapperLine::prepareAppendViewSequence()
{
    append_view_seq_.clear();
    append_actual_views_.clear();
    append_view_cursor_ = 0;
    if (!scene_) return;

    std::vector<unsigned long> fids;
    std::vector<int> remaining;
    fids.reserve(scene_->keyframes().size());
    remaining.reserve(scene_->keyframes().size());
    for (auto& kv : scene_->keyframes()) {
        fids.push_back(static_cast<unsigned long>(kv.second->fid_));
        remaining.push_back(kv.second->remaining_times_of_use_);
    }

    // Budget mode has no step count: pre-generate a generous deterministic
    // sequence (each cycle still uses every keyframe).
    const int planned_steps =
        (append_config_.steps > 0) ? append_config_.steps : 50000;
    append_view_seq_ = append_refinement::buildViewSequence(
        fids.size(), planned_steps, append_config_.seed);

    std::filesystem::create_directories(append_dir_);
    std::ofstream out(append_dir_ / "append_views_planned.csv");
    out << "step,map_index,fid,remaining_times_of_use_at_entry\n";
    for (std::size_t i = 0; i < append_view_seq_.size(); ++i) {
        const std::size_t idx = append_view_seq_[i];
        out << i << ',' << idx << ',' << fids[idx] << ','
            << remaining[idx] << '\n';
    }
    out.flush();

    long long sum_rem = 0, min_rem = 0, max_rem = 0;
    if (!remaining.empty()) {
        min_rem = remaining.front();
        max_rem = remaining.front();
        for (int v : remaining) {
            sum_rem += v;
            min_rem = std::min<long long>(min_rem, v);
            max_rem = std::max<long long>(max_rem, v);
        }
    }

    std::ostringstream os;
    os << "append_view_sequence=independent_precomputed_seed_" << append_config_.seed
       << " planned_steps=" << append_view_seq_.size()
       << " keyframes=" << fids.size()
       << " remaining_times_of_use_entry_sum=" << sum_rem
       << " min=" << min_rem << " max=" << max_rem
       << " budget_reset=none"
       << " note=append_scheduler_does_not_inherit_remaining_budget"
       << " lr_input=kfs_used_times_continues_from_natural_training";
    append_refinement::appendLine(append_dir_ / "append_summary.txt", os.str());
}

std::shared_ptr<GaussianKeyframeLine> GaussianMapperLine::useOneAppendKeyframe()
{
    if (append_view_cursor_ >= append_view_seq_.size())
        return nullptr;
    const std::size_t map_idx = append_view_seq_[append_view_cursor_];
    auto it = scene_->keyframes().begin();
    for (std::size_t i = 0; i < map_idx && it != scene_->keyframes().end(); ++i)
        ++it;
    if (it == scene_->keyframes().end())
        return nullptr;

    auto kf = it->second;
    const long long append_step = static_cast<long long>(append_view_cursor_);
    ++append_view_cursor_;

    const auto fid = kf->fid_;
    auto found = kfs_used_times_.find(fid);
    if (found == kfs_used_times_.end())
        kfs_used_times_[fid] = 1;
    else
        ++found->second;

    append_actual_views_.push_back(
        {append_step, static_cast<long long>(map_idx),
         static_cast<long long>(fid)});
    return kf;
}

void GaussianMapperLine::appendRecordFirstStepAudit(const char* stage)
{
    if (!append_adam_audit_ || !gaussians_ || !gaussians_->optimizer_)
        return;
    const auto info =
        append_refinement::captureGroupGradInfo(*gaussians_->optimizer_);
    if (std::string(stage) == "first_step_before")
        append_first_step_grads_ = info;
    for (const auto& g : info) {
        *append_adam_audit_ << stage << ',' << getIteration() << ','
                            << append_completed_steps_ << ','
                            << g.group_index << ','
                            << append_refinement::boolStr(g.has_param) << ','
                            << append_refinement::boolStr(g.grad_defined) << ','
                            << g.grad_l2 << ',' << g.adam_step << ','
                            << g.impl_ptr << ',' << g.data_ptr << '\n';
    }
    append_adam_audit_->flush();
}

void GaussianMapperLine::appendWriteBoundary(const std::filesystem::path& file,
                                             const char* stage)
{
    if (!gaussians_ || !gaussians_->optimizer_)
        return;
    const bool first = std::string(stage) == "before";
    std::ofstream out(file, std::ios::out | (first ? std::ios::trunc : std::ios::app));
    if (first)
        out << "stage,group,has_param,grad_defined,grad_l2,adam_step,impl_ptr,data_ptr\n";
    const auto info =
        append_refinement::captureGroupGradInfo(*gaussians_->optimizer_);
    for (const auto& g : info) {
        out << stage << ',' << g.group_index << ','
            << append_refinement::boolStr(g.has_param) << ','
            << append_refinement::boolStr(g.grad_defined) << ','
            << g.grad_l2 << ',' << g.adam_step << ','
            << g.impl_ptr << ',' << g.data_ptr << '\n';
    }
    out.flush();
}

void GaussianMapperLine::appendWriteStatus(bool ok, const std::string& reason)
{
    append_ok_ = ok;
    append_fail_reason_ = reason;
    std::filesystem::create_directories(append_dir_);
    std::ofstream out(append_dir_ / "append_status.txt",
                      std::ios::out | std::ios::trunc);
    out << std::fixed << std::setprecision(6);
    out << "ok=" << (ok ? 1 : 0) << '\n';
    out << "fail_reason=" << reason << '\n';
    out << "mode=" << append_refinement::appendModeName(time_budget_mode_,
                                                          append_config_.steps) << '\n';
    out << "budget_seconds=" << time_budget_config_.seconds << '\n';
    out << "budget_status=" << budget_status_ << '\n';
    out << "budget_unmatched=" << (budget_unmatched_ ? 1 : 0) << '\n';
    out << "budget_overshoot_seconds=" << budget_overshoot_seconds_ << '\n';
    out << "cumulative_train_seconds=" << cumulative_train_seconds_ << '\n';
    out << "natural_cumulative_train_seconds="
        << natural_cumulative_train_seconds_ << '\n';
    out << "append_steps_requested=" << append_config_.steps << '\n';
    out << "append_iterations=" << append_iterations_ << '\n';
    out << "append_completed_steps=" << append_completed_steps_ << '\n';
    out << "natural_completed_steps=" << natural_completed_steps_ << '\n';
    out << "raw_iteration=" << getIteration() << '\n';
    out << "append_start_iteration=" << append_start_iteration_ << '\n';
    out << "iteration_cap=" << opt_params_.iterations_ << '\n';
    out << "gaussians=" << appendValidatedCount() << '\n';
    out << "gaussians_at_entry=" << append_gaussian_count_before_ << '\n';
    out << "line_gaussians=" << appendValidatedLineCount() << '\n';
    out << "line_gaussians_at_entry=" << append_line_count_before_ << '\n';
    out << "natural_train_seconds=" << natural_train_seconds_ << '\n';
    out << "append_train_seconds=" << append_train_seconds_ << '\n';
    out << "append_preflight_seconds=" << append_preflight_seconds_ << '\n';
    out << "append_queue_seconds=" << append_queue_seconds_ << '\n';
    out << "natural_save_seconds=" << natural_save_seconds_ << '\n';
    out << "budget_save_seconds=" << budget_save_seconds_ << '\n';
    out << "append_validation_seconds=" << append_validation_seconds_ << '\n';
    out << "natural_end_ply_dir=" << natural_end_ply_dir_.string() << '\n';
    out << "budget_end_ply_dir=" << budget_end_ply_dir_.string() << '\n';
    out << "natural_end_iteration=" << natural_end_iteration_ << '\n';
    out << "budget_end_iteration=" << budget_end_iteration_ << '\n';
    out << "line_loss_every=" << lineLossEvery() << '\n';
    out << "line_loss_disabled_in_append=1\n";
    const char* backend = std::getenv("PHOTO_SLAM_RASTER_BACKEND");
    out << "raster_backend=" << (backend ? backend : "unset") << '\n';
    out << "timestamp=" << append_refinement::nowStamp() << '\n';
    out.flush();
}

void GaussianMapperLine::runAppendRefinement()
{
    append_dir_ = result_dir_ / "append";
    std::filesystem::create_directories(append_dir_);
    append_phase_ = false;
    append_ok_ = false;
    append_fail_reason_.clear();
    append_completed_steps_ = 0;
    append_iterations_ = 0;
    append_start_iteration_ = 0;
    append_gaussian_count_before_ = -1;
    append_line_count_before_ = -1;
    budget_status_ = time_budget_mode_ ? "pending" : "disabled";
    budget_unmatched_ = false;
    budget_overshoot_seconds_ = 0.0;
    append_queue_seconds_ = 0.0;
    natural_save_seconds_ = 0.0;
    budget_save_seconds_ = 0.0;
    natural_end_ply_dir_.clear();
    budget_end_ply_dir_.clear();
    natural_end_iteration_ = -1;
    budget_end_iteration_ = -1;

    {
        std::ofstream s(append_dir_ / "append_summary.txt",
                        std::ios::out | std::ios::trunc);
        s << "# append refinement (line build)\n";
        s << "enabled=1\n";
        s << "build=L_target\n";
        s << "mode=" << append_refinement::appendModeName(
                             time_budget_mode_, append_config_.steps) << '\n';
        s << "budget_seconds=" << time_budget_config_.seconds << '\n';
        s << "config_steps=" << append_config_.steps << '\n';
        s << "seed=" << append_config_.seed << '\n';
        s << "line_loss_disabled_in_append=1\n";
        s << "started=" << append_refinement::nowStamp() << '\n';
    }

    const auto preflight_start = std::chrono::steady_clock::now();
    std::string reason;
    int drained_batches = 0;
    if (!appendPreflight(reason, drained_batches)) {
        std::cerr << "[APPEND][FAIL] preflight: " << reason << std::endl;
        append_preflight_seconds_ =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - preflight_start).count() / 1000.0;
        appendWriteStatus(false, reason);
        return;
    }
    append_preflight_seconds_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - preflight_start).count() / 1000.0;
    std::ostringstream pre;
    pre << std::fixed << std::setprecision(6)
        << "slam_shutdown=1 queue_settled=1 drained_operation_batches="
        << drained_batches
        << " queue_drain_seconds=" << append_queue_seconds_
        << " preflight_seconds=" << append_preflight_seconds_;
    append_refinement::appendLine(append_dir_ / "append_summary.txt", pre.str());

    append_start_iteration_ = getIteration();
    natural_completed_steps_ = frozen_opt_steps_;
    natural_train_seconds_ = total_training_time_ms_ / 1000.0;
    natural_cumulative_train_seconds_ = cumulative_train_seconds_;

    // Natural-end model in the consistent boundary state (SLAM stopped and
    // queue empty). Evaluation is offline on this PLY.
    {
        const auto save_start = std::chrono::steady_clock::now();
        natural_end_iteration_ = getIteration();
        natural_end_ply_dir_ = append_dir_ / "natural_end";
        savePly(natural_end_ply_dir_);
        natural_save_seconds_ =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - save_start).count();
    }

    if (!time_budget_mode_ &&
        static_cast<long long>(append_start_iteration_) + append_config_.steps >
            static_cast<long long>(opt_params_.iterations_)) {
        reason = "iteration_cap_would_block_real_updates";
        std::cerr << "[APPEND][FAIL] " << reason
                  << " (start_iteration=" << append_start_iteration_
                  << " steps=" << append_config_.steps
                  << " cap=" << opt_params_.iterations_ << ")" << std::endl;
        appendWriteStatus(false, reason);
        return;
    }
    if (time_budget_mode_ &&
        static_cast<long long>(append_start_iteration_) >=
            static_cast<long long>(opt_params_.iterations_)) {
        budget_status_ = "iteration_cap";
        budget_unmatched_ = true;
        std::cerr << "[APPEND][BUDGET] unmatched: iteration cap already reached"
                  << " (start_iteration=" << append_start_iteration_
                  << " cap=" << opt_params_.iterations_ << ")" << std::endl;
    }

    append_param_snapshot_ =
        append_refinement::captureParamSnapshot(*gaussians_->optimizer_);
    append_adam_steps_before_ =
        append_refinement::captureAdamSteps(*gaussians_->optimizer_);
    append_kf_before_ = append_refinement::snapshotKeyframeStates(*scene_);
    append_camera_hash_before_ = append_refinement::cameraFingerprint(*scene_);
    append_gaussian_count_before_ = appendValidatedCount();
    append_line_count_before_ = appendValidatedLineCount();
    appendWriteBoundary(append_dir_ / "append_optimizer_boundary.csv", "before");

    prepareAppendViewSequence();

    if (append_diag_mode_) {
        append_adam_audit_ = std::make_unique<std::ofstream>(
            append_dir_ / "append_step_audit.csv",
            std::ios::out | std::ios::trunc);
        *append_adam_audit_
            << "stage,iteration,append_completed_steps,group,has_param,"
               "grad_defined,grad_l2,adam_step,impl_ptr,data_ptr\n";
    }

    // ---- real append training: SAME trainForOneIterationErrorGuided() -----
    append_phase_ = true;
    const auto train_start = std::chrono::steady_clock::now();
    int no_progress = 0;
    bool failed = false;
    const long long step_cap = (append_config_.steps > 0)
                                   ? static_cast<long long>(append_config_.steps)
                                   : std::numeric_limits<long long>::max();

    if (time_budget_mode_ &&
        natural_cumulative_train_seconds_ >= time_budget_config_.seconds) {
        budget_status_ = "natural_exceeds_budget";
        budget_unmatched_ = true;
        budget_overshoot_seconds_ =
            natural_cumulative_train_seconds_ - time_budget_config_.seconds;
        std::cerr << "[APPEND][BUDGET] unmatched: natural training already used "
                  << natural_cumulative_train_seconds_ << "s >= budget "
                  << time_budget_config_.seconds << "s; append skipped" << std::endl;
    }

    while (append_completed_steps_ < step_cap) {
        if (time_budget_mode_ &&
            cumulative_train_seconds_ >= time_budget_config_.seconds)
            break;

        if (static_cast<long long>(getIteration()) >=
            static_cast<long long>(opt_params_.iterations_)) {
            reason = "iteration_cap_reached_during_append";
            if (time_budget_mode_) {
                budget_status_ = "iteration_cap";
                budget_unmatched_ = true;
            }
            else {
                failed = true;
            }
            break;
        }
        if (append_view_cursor_ >= append_view_seq_.size()) {
            reason = "append_view_sequence_exhausted";
            failed = true;
            break;
        }

        const long long before = append_completed_steps_;
        try {
            trainForOneIterationErrorGuided();
        }
        catch (const std::exception& e) {
            reason = std::string("append_training_exception:") + e.what();
            failed = true;
            break;
        }
        catch (...) {
            reason = "append_training_exception:unknown";
            failed = true;
            break;
        }
        ++append_iterations_;

        if (device_type_ == torch::kCUDA)
            torch::cuda::synchronize();

        if (append_completed_steps_ == before) {
            ++no_progress;
            if (no_progress >= append_config_.max_no_progress) {
                reason = "no_progress_no_optimizer_step";
                failed = true;
                break;
            }
        }
        else {
            no_progress = 0;
        }
    }
    if (device_type_ == torch::kCUDA)
        torch::cuda::synchronize();
    append_phase_ = false;
    if (append_adam_audit_)
        append_adam_audit_->flush();

    if (time_budget_mode_) {
        append_train_seconds_ = std::max(
            0.0, cumulative_train_seconds_ - natural_cumulative_train_seconds_);
        if (cumulative_train_seconds_ >= time_budget_config_.seconds) {
            budget_status_ = "matched";
            budget_unmatched_ = false;
            budget_overshoot_seconds_ =
                cumulative_train_seconds_ - time_budget_config_.seconds;
        }
        else if (append_config_.steps > 0 &&
                 append_completed_steps_ >= append_config_.steps) {
            budget_status_ = "step_cap_before_budget";
            budget_unmatched_ = true;
        }
        else if (budget_status_ == "pending") {
            budget_status_ = "stopped_unknown";
            budget_unmatched_ = true;
        }
    }
    else {
        append_train_seconds_ =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - train_start).count();
    }

    {
        if (budget_status_ == "natural_exceeds_budget") {
            budget_end_ply_dir_ = natural_end_ply_dir_;
            budget_end_iteration_ = natural_end_iteration_;
            budget_save_seconds_ = 0.0;
        }
        else {
            const auto save_start = std::chrono::steady_clock::now();
            budget_end_iteration_ = getIteration();
            budget_end_ply_dir_ = append_dir_ / "budget_end";
            savePly(budget_end_ply_dir_);
            budget_save_seconds_ =
                std::chrono::duration_cast<std::chrono::duration<double>>(
                    std::chrono::steady_clock::now() - save_start).count();
        }
    }

    const auto validation_start = std::chrono::steady_clock::now();
    const auto param_diff = append_refinement::diffParamSnapshot(
        append_param_snapshot_, *gaussians_->optimizer_);
    const auto detailed = append_refinement::diffParamSnapshotDetailed(
        append_param_snapshot_, *gaussians_->optimizer_);
    const auto adam_steps_after =
        append_refinement::captureAdamSteps(*gaussians_->optimizer_);
    const auto kf_after = append_refinement::snapshotKeyframeStates(*scene_);
    const auto kf_cmp =
        append_refinement::compareKeyframeStates(append_kf_before_, kf_after);
    const auto camera_hash_after = append_refinement::cameraFingerprint(*scene_);
    const long long gaussian_count_after = appendValidatedCount();
    const long long line_count_after = appendValidatedLineCount();
    const bool slam_shutdown_after = pSLAM_ && pSLAM_->isShutDown();
    const bool queue_empty_after =
        pSLAM_ && !pSLAM_->getAtlas()->hasMappingOperation();

    // Adam-step verification: strict per-grad-group check only in explicit
    // diagnostic/acceptance mode. Normal/formal mode uses stage-boundary Adam
    // deltas and never reads per-step gradients.
    long long adam_expected_groups = -1;
    long long adam_checked_groups = 0;
    long long adam_advanced_groups = 0;
    bool adam_over_advanced = false;
    const bool strict_grad_audit =
        append_diag_mode_ && !append_first_step_grads_.empty();
    std::ostringstream adam_detail;
    if (strict_grad_audit) {
        adam_expected_groups = 0;
        for (std::size_t gi = 0; gi < append_first_step_grads_.size(); ++gi) {
            const auto& g = append_first_step_grads_[gi];
            if (!g.grad_defined)
                continue;
            ++adam_expected_groups;
            ++adam_checked_groups;
            long long before_step = (gi < append_adam_steps_before_.size())
                                        ? append_adam_steps_before_[gi] : -1;
            long long after_step = (gi < adam_steps_after.size())
                                       ? adam_steps_after[gi] : -1;
            const long long delta = after_step - before_step;
            if (append_completed_steps_ > 0 &&
                delta == append_completed_steps_)
                ++adam_advanced_groups;
            if (delta > append_completed_steps_)
                adam_over_advanced = true;
            adam_detail << " group" << gi << "_step_delta=" << delta;
        }
    }
    else {
        for (std::size_t gi = 0; gi < append_adam_steps_before_.size(); ++gi) {
            ++adam_checked_groups;
            long long before_step = append_adam_steps_before_[gi];
            long long after_step = (gi < adam_steps_after.size())
                                       ? adam_steps_after[gi] : -1;
            const long long delta = after_step - before_step;
            if (append_completed_steps_ > 0 &&
                delta == append_completed_steps_)
                ++adam_advanced_groups;
            if (delta > append_completed_steps_)
                adam_over_advanced = true;
            adam_detail << " group" << gi << "_step_delta=" << delta;
        }
    }

    bool adam_ok = (append_completed_steps_ == 0) ||
                   (strict_grad_audit
                        ? (adam_expected_groups > 0 &&
                           adam_advanced_groups == adam_expected_groups &&
                           !adam_over_advanced)
                        : (adam_advanced_groups > 0 && !adam_over_advanced));
    const bool steps_ok = time_budget_mode_
                              ? true
                              : (append_completed_steps_ == append_config_.steps);
    const bool params_updated = (append_completed_steps_ == 0) ||
                                (param_diff.changed_elements > 0);
    const bool params_finite =
        param_diff.nan_elements == 0 && param_diff.inf_elements == 0;
    const bool identity_ok =
        param_diff.identity_preserved && param_diff.shape_preserved;
    const bool count_ok =
        gaussian_count_after == append_gaussian_count_before_ &&
        line_count_after == append_line_count_before_;
    const bool geometry_ok =
        (kf_cmp[0] <= 1e-12) && (kf_cmp[1] <= 1e-12) &&
        (kf_cmp[2] == 0.0) &&
        (append_camera_hash_before_ == camera_hash_after) &&
        (append_kf_before_.size() == kf_after.size()) &&
        slam_shutdown_after && queue_empty_after &&
        (append_actual_views_.size() ==
         static_cast<std::size_t>(append_iterations_));
    const bool budget_ok =
        !time_budget_mode_ ||
        (budget_status_ == "matched" ||
         budget_status_ == "natural_exceeds_budget" ||
         budget_status_ == "iteration_cap" ||
         budget_status_ == "step_cap_before_budget");

    append_validation_seconds_ =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - validation_start).count() / 1000.0;

    {
        std::ofstream out(append_dir_ / "append_param_diff.csv",
                          std::ios::out | std::ios::trunc);
        out << "group,param_index,count,changed,max_abs_delta,l2_delta,"
               "nan,inf,identity_ok,shape_ok\n";
        for (const auto& d : detailed)
            out << d.group_index << ',' << d.param_index << ',' << d.count << ','
                << d.changed << ',' << d.max_abs_delta << ',' << d.l2_delta
                << ',' << d.nan_elements << ',' << d.inf_elements << ','
                << (d.identity_ok ? 1 : 0) << ',' << (d.shape_ok ? 1 : 0) << '\n';
    }
    {
        std::ofstream out(append_dir_ / "append_views_actual.csv",
                          std::ios::out | std::ios::trunc);
        out << "step,map_index,fid\n";
        for (const auto& v : append_actual_views_)
            out << v[0] << ',' << v[1] << ',' << v[2] << '\n';
    }
    appendWriteBoundary(append_dir_ / "append_optimizer_boundary.csv", "after");

    {
        std::ostringstream os;
        os << std::fixed << std::setprecision(6);
        os << "boundary_validation"
           << " mode=" << append_refinement::appendModeName(
                                time_budget_mode_, append_config_.steps)
           << " budget_seconds=" << time_budget_config_.seconds
           << " budget_status=" << budget_status_
           << " budget_unmatched=" << (budget_unmatched_ ? 1 : 0)
           << " budget_overshoot_seconds=" << budget_overshoot_seconds_
           << " cumulative_train_seconds=" << cumulative_train_seconds_
           << " natural_cumulative_train_seconds="
           << natural_cumulative_train_seconds_
           << " natural_completed_steps=" << natural_completed_steps_
           << " append_iterations=" << append_iterations_
           << " append_completed_steps=" << append_completed_steps_
           << " raw_iteration=" << getIteration()
           << " natural_train_seconds=" << natural_train_seconds_
           << " append_train_seconds=" << append_train_seconds_
           << " append_preflight_seconds=" << append_preflight_seconds_
           << " append_queue_seconds=" << append_queue_seconds_
           << " natural_save_seconds=" << natural_save_seconds_
           << " budget_save_seconds=" << budget_save_seconds_
           << " append_validation_seconds=" << append_validation_seconds_
           << " natural_end_iteration=" << natural_end_iteration_
           << " budget_end_iteration=" << budget_end_iteration_
           << " gaussians_entry=" << append_gaussian_count_before_
           << " gaussians_end=" << gaussian_count_after
           << " line_gaussians_entry=" << append_line_count_before_
           << " line_gaussians_end=" << line_count_after
           << " params_changed_elements=" << param_diff.changed_elements
           << " params_total_elements=" << param_diff.total_elements
           << " params_changed_tensors=" << param_diff.changed_param_tensors
           << " params_max_abs_delta=" << param_diff.max_abs_delta
           << " params_l2_delta=" << param_diff.l2_delta
           << " params_nan=" << param_diff.nan_elements
           << " params_inf=" << param_diff.inf_elements
           << " optimizer_identity_preserved="
           << append_refinement::boolStr(identity_ok)
           << " adam_groups_expected=" << adam_expected_groups
           << " adam_groups_checked=" << adam_checked_groups
           << " adam_groups_advanced_by_" << append_completed_steps_ << "="
           << adam_advanced_groups
           << " pose_max_delta=" << kf_cmp[0]
           << " intrinsics_max_delta=" << kf_cmp[1]
           << " changed_keyframes=" << kf_cmp[2]
           << " camera_fingerprint_entry=" << append_camera_hash_before_
           << " slam_shutdown_after="
           << append_refinement::boolStr(slam_shutdown_after)
           << " pending_operations_after="
           << append_refinement::boolStr(!queue_empty_after)
           << " camera_fingerprint_end=" << camera_hash_after
           << " append_views_used=" << append_actual_views_.size()
           << adam_detail.str();
        append_refinement::appendLine(append_dir_ / "append_summary.txt", os.str());
    }

    const bool ok = !failed && steps_ok && params_updated && params_finite &&
                    identity_ok && count_ok && geometry_ok && adam_ok &&
                    budget_ok;
    if (!ok && reason.empty()) {
        if (!steps_ok) reason = "append_step_count_mismatch";
        else if (!params_updated) reason = "no_parameter_update";
        else if (!params_finite) reason = "non_finite_parameter";
        else if (!identity_ok) reason = "optimizer_identity_changed";
        else if (!count_ok) reason = "gaussian_count_changed";
        else if (!geometry_ok) reason = "camera_or_keyframe_changed";
        else if (!adam_ok) reason = "adam_step_not_advanced";
        else if (!budget_ok) reason = "budget_status_invalid";
        else reason = "unknown";
    }
    appendWriteStatus(ok, ok ? "none" : reason);

    std::cerr << "[APPEND] " << (ok ? "OK" : "FAIL")
              << " build=L mode=" << append_refinement::appendModeName(
                     time_budget_mode_, append_config_.steps)
              << " steps=" << append_completed_steps_ << "/"
              << (append_config_.steps > 0
                      ? std::to_string(append_config_.steps)
                      : std::string("budget"))
              << " natural_steps=" << natural_completed_steps_
              << " raw_iteration=" << getIteration()
              << " gaussians=" << gaussian_count_after
              << " line_gaussians=" << line_count_after
              << " changed_elements=" << param_diff.changed_elements
              << " nan=" << param_diff.nan_elements
              << " pose_delta=" << kf_cmp[0]
              << " adam_groups=" << adam_advanced_groups << "/"
              << (strict_grad_audit ? adam_expected_groups : adam_checked_groups)
              << " cumulative_train_seconds=" << cumulative_train_seconds_
              << " budget_status=" << budget_status_
              << " budget_unmatched=" << (budget_unmatched_ ? 1 : 0)
              << " overshoot=" << budget_overshoot_seconds_
              << " train_seconds=" << append_train_seconds_
              << " reason=" << (ok ? "none" : reason) << std::endl;
    if (!ok)
        std::cerr << "[APPEND][FAIL] see " << (append_dir_ / "append_status.txt")
                  << std::endl;
}

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapperLine::trainForOneIteration()
{
    const auto budget_iter_start = std::chrono::steady_clock::now();
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframeLine> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

    // Per-iteration file I/O is expensive; log keyframe usage periodically.
    if (getIteration() % 200 == 0)
        writeKeyframeUsedTimes(result_dir_ / "used_times");

    // if (isdoingInactiveGeoDensify() && !viewpoint_cam->done_inactive_geo_densify_)
    //     increasePcdByKeyframeInactiveGeoDensify(viewpoint_cam);

    int training_level = num_gaus_pyramid_sub_levels_;
    int image_height, image_width;
    torch::Tensor gt_image, mask;
    if (isdoingGausPyramidTraining())
        training_level = viewpoint_cam->getCurrentGausPyramidLevel();
    if (training_level == num_gaus_pyramid_sub_levels_) {
        image_height = viewpoint_cam->image_height_;
        image_width = viewpoint_cam->image_width_;
        gt_image = viewpoint_cam->original_image_.cuda();
        mask = undistort_mask_[viewpoint_cam->camera_id_];
    }
    else {
        image_height = viewpoint_cam->gaus_pyramid_height_[training_level];
        image_width = viewpoint_cam->gaus_pyramid_width_[training_level];
        gt_image = viewpoint_cam->gaus_pyramid_original_image_[training_level].cuda();
        mask = scene_->cameras_.at(viewpoint_cam->camera_id_).gaus_pyramid_undistort_mask_[training_level];
    }

    // Mutex lock for usage of the gaussian model
    std::unique_lock<std::mutex> lock_render(mutex_render_);

    // Every 1000 its we increase the levels of SH up to a maximum degree
    if (getIteration() % 1000 == 0 && default_sh_ < model_params_.sh_degree_)
        default_sh_ += 1;
    // if (isdoingGausPyramidTraining())
    //     gaussians_->setShDegree(training_level);
    // else
        gaussians_->setShDegree(default_sh_);

    // Update learning rate
    if (pSLAM_) {
        int used_times = kfs_used_times_[viewpoint_cam->fid_];
        int step = (used_times <= opt_params_.position_lr_max_steps_ ? used_times : opt_params_.position_lr_max_steps_);
        float position_lr = gaussians_->updateLearningRate(step);
        setPositionLearningRateInit(position_lr);
    }
    else {
        gaussians_->updateLearningRate(getIteration());
    }

    gaussians_->setFeatureLearningRate(featureLearningRate());
    gaussians_->setOpacityLearningRate(opacityLearningRate());
    gaussians_->setScalingLearningRate(scalingLearningRate());
    gaussians_->setRotationLearningRate(rotationLearningRate());

    // Render
    g_gpu_prof.markStart();
    auto render_pkg = GaussianRendererWithLine::renderWithLine(
        viewpoint_cam,
        image_height,
        image_width,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    
    auto rendered_image = std::get<0>(render_pkg);
    auto viewspace_point_tensor = std::get<1>(render_pkg);
    auto visibility_filter = std::get<2>(render_pkg);
    auto radii = std::get<3>(render_pkg);
    g_gpu_prof.markForwardEnd();

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
    
    ////added by zdg(同一个线段上的高斯中心（XYZ）应该落在该线段的轴线上)对于属于同一条线的点 Pi​，已知其线方向单位向量为 d，线段上的一点（例如质心）为 Pcenter​
    //torch::Tensor loss_line = gaussians_->computeLineCoherenceLoss(0.1f); // 权重 0.1
    //auto loss_line_shape = gaussians_->computeLineShapeConstraint(0.05f, 0.1f); // 形状约束
    //loss = loss + loss_line + loss_line_shape;  //modified by zdg

    // ==============================================================================
    // 🌟 [新增] 各向异性惩罚 (Anisotropy Penalty)
    // ==============================================================================
    // 限制高斯球的最大缩放比例。通常对于 Line Gaussians 设为 500-1000 较安全。
    // ==============================================================================
    // 使用从配置文件读取的参数
    float max_aniso_th = opt_params_.max_anisotropy_threshold_; 
    float lambda_aniso = opt_params_.lambda_anisotropy_; 

    // Structural weights are read here so the scale activation can be shared
    // between the anisotropy penalty and the eccentricity term below.
    float w_ori = opt_params_.weight_line_coherence_;
    float w_shape_ecc = opt_params_.weight_line_shape_ecc_;
    float w_shape_ori = opt_params_.weight_line_shape_ori_;

    // Stage-1: exp(scaling_) is needed by the anisotropy penalty AND by the
    // eccentricity term. Build it at most ONCE per iteration and pass it down,
    // instead of letting each term construct its own activation. When neither
    // term is active nothing is built at all.
    torch::Tensor scaling_act;
    const bool need_scale_act = (lambda_aniso > 0.0f) || (w_shape_ecc > 0.0f);
    if (need_scale_act)
        scaling_act = gaussians_->getScaling();

    if (lambda_aniso > 0.0f) {
        // NOTE: max(dim)/min(dim) VALUES are used, NOT amax/amin. The values are
        // identical but the GRADIENTS are not: max(dim) routes the gradient to
        // the single argmax, while amax spreads it over every tied extremum.
        // Measured in this LibTorch (2.12) with the real initialisation pattern
        // (scale_axis1 == scale_axis2 exactly): relative gradient difference
        // 0.5. Reverted deliberately - the substitution changed training
        // semantics. See reports/perf_stage2_report.md section 1.
        auto max_res = torch::max(scaling_act, /*dim=*/1);
        torch::Tensor max_s = std::get<0>(max_res);
        auto min_res = torch::min(scaling_act, /*dim=*/1);
        torch::Tensor min_s = std::get<0>(min_res);
        torch::Tensor ratio = max_s / (min_s + 1e-6);
        torch::Tensor loss_aniso =
            torch::pow(torch::clamp(ratio - max_aniso_th, /*min=*/0), 2).mean();
        loss = loss + (lambda_aniso * loss_aniso);
    }
    // ==============================================================================

    // ==============================================================================
    // 🌟 核心改进：引入结构正则化 L_str (Equation 9 in your paper) 及其消融开关
    // ==============================================================================
    // 从配置文件中读取权重，而不是硬编码 0.1f 或 0.05f
    // 如果权重为 0，则代表正在做 "关闭 L_str" 的消融实验 (Ablation Study)
    // w_ori / w_shape_ecc / w_shape_ori are read above, next to the anisotropy
    // penalty, so one scale activation can serve both.

    // Formula-preserving: all three terms are individually weighted inside the
    // callees, so the block must also be entered when ONLY the orientation weight
    // is non-zero (otherwise an orientation-only ablation would silently be a
    // no-op and would be misread as "the term has no effect").
    if (lineLossEvery() > 0 &&
        (w_ori > 0.0f || w_shape_ecc > 0.0f || w_shape_ori > 0.0f)
        && (getIteration() < 100 || getIteration() % lineLossEvery() == 0)) {
        torch::Tensor loss_str = torch::zeros({1}, torch::dtype(torch::kFloat32).device(device_type_));
        
        // 方向一致性损失 L_ori
        if (w_ori > 0.0f) {
            // 建议：将权重乘法移到函数外部，让函数只返回纯粹的 Loss 标量，更符合 PyTorch 规范
            //loss_str = loss_str +  gaussians_->computeLineCoherenceLoss(w_ori);
            loss_str = loss_str + (useRefStructLoss()
                                       ? gaussians_->computeVectorizedLineLossRef(w_ori)
                                       : gaussians_->computeVectorizedLineLoss(w_ori));
        }
        // 形状各向异性约束 L_ecc
        // Formula-preserving decoupling: each term is already multiplied by its
        // own weight inside computeLineShapeConstraint(), so calling it with only
        // one weight non-zero changes no expression - it only allows the
        // eccentricity and orientation terms to be ablated independently. The
        // previous `&&` gate made them inseparable.
        if (w_shape_ecc > 0.0f || w_shape_ori > 0.0f) {
            loss_str = loss_str + (useRefStructLoss()
                                       ? gaussians_->computeLineShapeConstraintRef(w_shape_ecc, w_shape_ori)
                                       : gaussians_->computeLineShapeConstraint(
                                             w_shape_ecc, w_shape_ori, scaling_act));
        }
        
        loss = loss + loss_str;
    }

    g_gpu_prof.markLineLossEnd();

    loss.backward();
    g_gpu_prof.markBackwardEnd();

    // ==============================================================================
    // 🌟 性能改进：DEBUG 代码必须用宏或 Flag 包裹，否则严重拖慢帧率
    // ==============================================================================
    
#ifdef DEBUG_LINE_GRADIENTS
    // 【诊断代码】
    if (getIteration() % 10 == 0) {
        // 1. 检查 xyz 是否开启了梯度需求
        bool req = gaussians_->getXYZ().requires_grad();
    
        // 2. 检查梯度是否计算出来了 (grad norm)
        float grad_norm = 0.0f;
        if (gaussians_->getXYZ().grad().defined()) {
            grad_norm = gaussians_->getXYZ().grad().norm().item<float>();
        }

        // 3. 检查参数指针是否一致
        // 获取优化器里存的参数指针
        auto opt_param = gaussians_->optimizer_->param_groups()[0].params()[0]; 
        // 获取模型里存的参数指针
        auto model_param = gaussians_->getXYZ();
        // 比较内存地址 (data_ptr)
        bool ptr_match = (opt_param.data_ptr() == model_param.data_ptr());

        std::cerr << "[DEBUG Grad] Iter:" << getIteration() 
              << " | ReqGrad: " << req 
              << " | GradNorm: " << grad_norm 
              << " | PtrMatch: " << (ptr_match ? "YES" : "NO!!!") << std::endl;
    }
#endif

    // 这一步同步是为了确保 backward 彻底完成，但 loss.item() 下方已经强制同步，故移除冗余的整卡同步。
    {
        torch::NoGradGuard no_grad;
        if (getIteration() < 100 || getIteration() % 10 == 0) {
            ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;
        }

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // Densification (frozen-count experiment: count must not change)
        if (!frozen_fixed_count_ && getIteration() < opt_params_.densify_until_iter_) {
            // Keep track of max radii in image-space for pruning
            gaussians_->max_radii2D_.index_put_(
                {visibility_filter},
                torch::max(gaussians_->max_radii2D_.index({visibility_filter}),
                            radii.index({visibility_filter})));
            // if (!isdoingGausPyramidTraining() || training_level < num_gaus_pyramid_sub_levels_)
                gaussians_->addDensificationStats(viewspace_point_tensor, visibility_filter);

            if ((getIteration() > opt_params_.densify_from_iter_) &&
                (getIteration() % densifyInterval()== 0)) {
                int size_threshold = (getIteration() > prune_big_point_after_iter_) ? 20 : 0;
                //gaussians_->densifyAndPrune(
                //    densifyGradThreshold(),
                //    densify_min_opacity_,//0.005,//
                //    scene_->cameras_extent_,
                //    size_threshold
                //);
                gaussians_->densifyAndPruneWithLineAwareness(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold
                );

                // ==============================================================================
                // 🌟 [新增] 强制剔除极致拉伸点 (Post-Pruning)
                // ==============================================================================
                // 在 densify 结束后，立即扫描并干掉由于梯度更新意外拉伸过长的“针”
                if (getIteration() > opt_params_.densify_from_iter_) {
                    // 使用从配置文件读取的参数
                    float absolute_max_ratio = opt_params_.absolute_max_ratio_; // 物理极限，超过此比例必是伪影
                    gaussians_->pruneExceedinglyAnisotropic(absolute_max_ratio);
                }

                // 🌟 手段 C：强制剔除“极小点” (Min-Scaling Pruning)
                // 很多刺眼的点是因为 Scaling 极小但 Opacity 极高。
                // 在这里调用一个自定义函数来清理它们（见下文）
                gaussians_->pruneTinyGaussians(1e-6f);
            }

            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }
        g_gpu_prof.markDensifyEnd();

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // 统计总耗时
        total_training_time_ms_ += iter_time;

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0)) 
        {
            // GaussianTrainerLine::trainingReport(
            //     getIteration(),
            //     opt_params_.iterations_,
            //     Ll1,
            //     loss,
            //     ema_loss_for_log_,
            //     loss_utils::l1_loss,
            //     iter_time,
            //     *gaussians_,
            //     *scene_,
            //     pipe_params_,
            //     background_
            // );
            writeTrainingLog(iter_time, ema_loss_for_log_, gaussians_->xyz_.size(0));
        }
            
        if ((all_keyframes_record_interval_ && getIteration() % all_keyframes_record_interval_ == 0)
            // || loop_closure_iteration_
            )
        {
            renderAndRecordAllKeyframes();
            savePly(result_dir_ / std::to_string(getIteration()) / "ply");
        }

        if (loop_closure_iteration_)
            loop_closure_iteration_ = false;

        // Optimizer step
        if (getIteration() < opt_params_.iterations_) {
            gaussians_->optimizer_->step();
            ++frozen_opt_steps_;   // real optimizer-step counter (ablation)
            gaussians_->optimizer_->zero_grad(true);
            // Keep rotation quaternions unit-norm: the raw rotation_ norm is
            // unconstrained (losses only see the normalized quaternion), so it
            // can drift to zero and degenerate the normalize() backward.
            if (getIteration() % 50 == 0)
                gaussians_->normalizeRotationQuaternions();
        }
        g_gpu_prof.markOptimizerEnd();
    }
    g_gpu_prof.endIteration(prof_refine_phase_);

    // Unified cumulative-training-time boundary (see GaussianMapper).
    if (time_budget_mode_) {
        if (device_type_ == torch::kCUDA)
            torch::cuda::synchronize();
        cumulative_train_seconds_ +=
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - budget_iter_start).count();
    }
}

bool GaussianMapperLine::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapperLine::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

bool GaussianMapperLine::hasMetInitialMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

bool GaussianMapperLine::hasMetIncrementalMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

/*
pointsLocal 确实几乎起不到什么作用;但是，在 Photo-SLAM 这个特定系统中，它有一个特定的用途，主要与 “几何加密策略 (Geometry Densification)” 有关，特别是在单目（Monocular）模式下。
void GaussianMapperLine::increasePcdByKeyframeInactiveGeoDensify(std::shared_ptr<GaussianKeyframeLine> pkf)
{
    // Frozen-count experiment: adding Gaussians here would break the invariant.
    if (frozen_fixed_count_ && frozen_kf_replay_)
        return;
    // ...
    // 1. 将 pointsLocal 转为 Tensor
    torch::Tensor kps_point_local_tensor = torch::from_blob(pkf->kps_point_local_.data(), ...);
    
    // 2. 检查哪些特征点有有效的 3D 坐标 (Z > 0)
    // 也就是判断这个 2D 特征点是否已经被三角化了
    torch::Tensor kps_has3D_tensor = torch::where(
        kps_point_local_tensor.index({..., 2}) > 0.0f, true, false);

    // 3. 调用单目几何加密算法
    // 利用这些已知的“种子点” (pointsLocal)，去推测周围像素的深度，从而生成更多的高斯球
    auto result = monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(..., kps_point_local_tensor, ...);
    
    // ...
}

*/


// ===========================================================================
// gba-recovery-v1: controlled two-arm experiment.
//
// One immutable GBA event is captured. Both arms start from the SAME
// in-process snapshot (Gaussian parameters, Adam state, learning-rate/SH/
// pyramid schedule, keyframe usage and pose). Both receive the same corrected
// camera poses and replay the same pre-generated training-view sequence.
// PoseOnly keeps the Gaussian geometry; PoseAndMap also applies the current
// ref-keyframe correction. External geometry is frozen (no add/densify/clone/
// split/prune/opacity reset). Evaluation order is exchanged on a second pass.
// ===========================================================================
GaussianMapperLine::RecoverySnapshot GaussianMapperLine::captureRecoverySnapshot()
{
    RecoverySnapshot s;
    if (!gaussians_) return s;
    auto cp = [](const torch::Tensor& t) {
        return t.defined() ? t.detach().clone() : torch::Tensor();
    };
    s.xyz = cp(gaussians_->xyz_);
    s.xyz_init = cp(gaussians_->xyz_init_);
    s.is_line = cp(gaussians_->is_line_);
    s.line_dir_w = cp(gaussians_->line_dir_w_);
    s.ref_kf_id = cp(gaussians_->ref_kf_id_);
    s.coord_event_id = cp(gaussians_->coord_event_id_);
    s.features_dc = cp(gaussians_->features_dc_);
    s.features_rest = cp(gaussians_->features_rest_);
    s.scaling = cp(gaussians_->scaling_);
    s.rotation = cp(gaussians_->rotation_);
    s.opacity = cp(gaussians_->opacity_);
    s.point_ids = cp(gaussians_->point_ids_);
    s.exist_since_iter = cp(gaussians_->exist_since_iter_);
    s.max_radii2D = cp(gaussians_->max_radii2D_);
    s.xyz_grad_accum = cp(gaussians_->xyz_gradient_accum_);
    s.denom = cp(gaussians_->denom_);

    s.adam.resize(6);
    if (gaussians_->optimizer_)
    {
        auto& groups = gaussians_->optimizer_->param_groups();
        auto& state = gaussians_->optimizer_->state();
        for (std::size_t i = 0; i < groups.size() && i < 6; ++i)
        {
            if (groups[i].params().empty()) continue;
            auto& p = groups[i].params()[0];
            auto it = state.find(p.unsafeGetTensorImpl());
            if (it == state.end()) continue;
            auto& st = static_cast<torch::optim::AdamParamState&>(*it->second);
            s.adam[i].step = static_cast<long long>(st.step());
            s.adam[i].exp_avg = cp(st.exp_avg());
            s.adam[i].exp_avg_sq = cp(st.exp_avg_sq());
        }
        s.lrs.resize(groups.size(), 0.0f);
        for (std::size_t i = 0; i < groups.size(); ++i)
            s.lrs[i] = static_cast<torch::optim::AdamOptions&>(
                           groups[i].options()).lr();
    }

    s.iteration = iteration_;
    s.frozen_opt_steps = frozen_opt_steps_;
    s.active_sh = gaussians_->active_sh_degree_;
    s.default_sh = default_sh_;
    s.ema_loss = ema_loss_for_log_;
    s.kfs_used_times = kfs_used_times_;
    for (auto& kv : scene_->keyframes())
    {
        auto& pkf = kv.second;
        if (!pkf) continue;
        RecoveryKfState k;
        k.remaining_times_of_use = pkf->remaining_times_of_use_;
        k.pyramid_times_of_use = pkf->gaus_pyramid_times_of_use_;
        k.tracking_error = pkf->tracking_error_;
        k.done_inactive_densify = pkf->done_inactive_geo_densify_;
        Sophus::SE3d pose = pkf->getPose();
        k.q = pose.unit_quaternion();
        k.t = pose.translation();
        k.has_pose = true;
        s.kfs[kv.first] = k;
    }
    return s;
}

void GaussianMapperLine::restoreRecoverySnapshot(const RecoverySnapshot& s)
{
    if (!gaussians_) return;
    torch::NoGradGuard ng;
    auto cp = [](torch::Tensor& dst, const torch::Tensor& src, const char* name) {
        if (!src.defined()) return;
        TORCH_CHECK(dst.defined(), "[GbaRecovery] ", name, " is undefined");
        TORCH_CHECK(dst.sizes() == src.sizes(), "[GbaRecovery] ", name,
                    " shape mismatch: ", dst.sizes(), " vs ", src.sizes());
        dst.copy_(src);
    };
    cp(gaussians_->xyz_, s.xyz, "xyz");
    cp(gaussians_->xyz_init_, s.xyz_init, "xyz_init");
    cp(gaussians_->is_line_, s.is_line, "is_line");
    cp(gaussians_->line_dir_w_, s.line_dir_w, "line_dir_w");
    cp(gaussians_->ref_kf_id_, s.ref_kf_id, "ref_kf_id");
    cp(gaussians_->coord_event_id_, s.coord_event_id, "coord_event_id");
    cp(gaussians_->features_dc_, s.features_dc, "features_dc");
    cp(gaussians_->features_rest_, s.features_rest, "features_rest");
    cp(gaussians_->scaling_, s.scaling, "scaling");
    cp(gaussians_->rotation_, s.rotation, "rotation");
    cp(gaussians_->opacity_, s.opacity, "opacity");
    cp(gaussians_->point_ids_, s.point_ids, "point_ids");
    cp(gaussians_->exist_since_iter_, s.exist_since_iter, "exist_since_iter");
    cp(gaussians_->max_radii2D_, s.max_radii2D, "max_radii2D");
    cp(gaussians_->xyz_gradient_accum_, s.xyz_grad_accum, "xyz_gradient_accum");
    cp(gaussians_->denom_, s.denom, "denom");
    gaussians_->invalidateLineCache();

    if (gaussians_->optimizer_)
    {
        auto& groups = gaussians_->optimizer_->param_groups();
        auto& state = gaussians_->optimizer_->state();
        for (std::size_t i = 0; i < groups.size() && i < s.lrs.size(); ++i)
            static_cast<torch::optim::AdamOptions&>(groups[i].options())
                .lr(s.lrs[i]);
        for (std::size_t i = 0; i < groups.size() && i < 6; ++i)
        {
            if (groups[i].params().empty()) continue;
            auto& p = groups[i].params()[0];
            auto key = p.unsafeGetTensorImpl();
            auto it = state.find(key);
            if (it == state.end())
            {
                auto st = std::make_unique<torch::optim::AdamParamState>();
                if (i < s.adam.size() && s.adam[i].step >= 0)
                    st->step(s.adam[i].step);
                if (i < s.adam.size() && s.adam[i].exp_avg.defined())
                    st->exp_avg(s.adam[i].exp_avg.clone());
                if (i < s.adam.size() && s.adam[i].exp_avg_sq.defined())
                    st->exp_avg_sq(s.adam[i].exp_avg_sq.clone());
                state[key] = std::move(st);
            }
            else
            {
                auto& st = static_cast<torch::optim::AdamParamState&>(*it->second);
                if (i < s.adam.size() && s.adam[i].step >= 0)
                    st.step(s.adam[i].step);
                if (i < s.adam.size() && s.adam[i].exp_avg.defined())
                    st.exp_avg(s.adam[i].exp_avg.clone());
                if (i < s.adam.size() && s.adam[i].exp_avg_sq.defined())
                    st.exp_avg_sq(s.adam[i].exp_avg_sq.clone());
            }
        }
    }

    iteration_ = s.iteration;
    frozen_opt_steps_ = s.frozen_opt_steps;
    default_sh_ = s.default_sh;
    ema_loss_for_log_ = s.ema_loss;
    kfs_used_times_ = s.kfs_used_times;
    gaussians_->setShDegree(s.active_sh);

    for (auto& kv : s.kfs)
    {
        auto pkf = scene_->getKeyframe(kv.first);
        if (!pkf) continue;
        if (kv.second.has_pose)
        {
            pkf->setPose(kv.second.q, kv.second.t);
            pkf->computeTransformTensors();
        }
        pkf->remaining_times_of_use_ = kv.second.remaining_times_of_use;
        pkf->gaus_pyramid_times_of_use_ = kv.second.pyramid_times_of_use;
        pkf->tracking_error_ = kv.second.tracking_error;
        pkf->done_inactive_geo_densify_ = kv.second.done_inactive_densify;
    }
    frozen_kf_step_ = 0;
    frozen_train_views_flat_.clear();
}

GaussianMapperLine::RecoveryArmResult GaussianMapperLine::runRecoveryArm(
    const std::string& tag, const RecoverySnapshot& base,
    const RecoverySnapshot& kf_usage,
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
    unsigned long evt, bool apply_map, int steps,
    const std::vector<long long>& eval_steps,
    const std::filesystem::path& out_dir)
{
    RecoveryArmResult r;
    if (!gaussians_) return r;

    restoreRecoverySnapshot(base);
    diag_snapshot_rot_data_ = base.rotation.defined()
        ? reinterpret_cast<std::uintptr_t>(base.rotation.data_ptr()) : 0;
    // gba-state-audit: make the two arms' paths explicit in the log, because
    // the production default for the Gaussian deformation is now OFF.
    std::cerr << "[GbaRecovery] arm=" << tag << " deformation="
              << (apply_map ? "ON(applyGbaCorrections)" : "OFF(pose-only)")
              << std::endl;
    for (auto& kv : kf_usage.kfs)
    {
        auto pkf = scene_->getKeyframe(kv.first);
        if (!pkf) continue;
        pkf->remaining_times_of_use_ = kv.second.remaining_times_of_use;
        pkf->gaus_pyramid_times_of_use_ = kv.second.pyramid_times_of_use;
        pkf->tracking_error_ = kv.second.tracking_error;
        pkf->done_inactive_geo_densify_ = kv.second.done_inactive_densify;
    }
    frozen_fixed_count_ = true;
    frozen_kf_replay_ = true;
    frozen_kf_step_ = 0;
    frozen_train_views_flat_.clear();
    frozen_opt_steps_ = 0;
    result_dir_ = out_dir;
    std::filesystem::create_directories(out_dir);
    const long long base_count = gaussians_->xyz_.size(0);

    auto syncDev = [&]() {
        if (device_type_ == torch::kCUDA && gaussians_->xyz_.defined() &&
            gaussians_->xyz_.numel() > 0)
            (void)gaussians_->xyz_.select(0, 0).sum().item<double>();
    };

    syncDev();
    const auto t_sync0 = std::chrono::steady_clock::now();
    // Both arms receive exactly the same corrected camera poses.
    for (const auto& p : after_poses)
    {
        auto pkf = scene_->getKeyframe(p.first);
        if (!pkf) continue;
        pkf->setPose(p.second.unit_quaternion().cast<double>(),
                     p.second.translation().cast<double>());
        pkf->computeTransformTensors();
    }
    if (apply_map)
    {
        auto st = gaussians_->applyGbaCorrections(world_deltas, evt);
        r.updated_rows = st.nonzero_moved_rows;
        r.max_translation = st.max_translation;
        r.mean_translation = st.moved_rows > 0
            ? st.translation_sum / static_cast<double>(st.moved_rows) : 0.0;
        std::cerr << "[GbaRecovery] arm=" << tag << " moved=" << st.moved_rows
                  << " nonzero=" << st.nonzero_moved_rows
                  << " line=" << st.line_rows_moved << std::endl;
    }
    else
    {
        std::cerr << "[GbaRecovery] arm=" << tag
                  << " moved=0 (deformation intentionally not applied)"
                  << std::endl;
    }
    syncDev();
    r.sync_seconds = std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - t_sync0).count() / 1e6;

    r.param_fp0 = gaussians_->paramFingerprint();
    r.adam_fp0 = gaussians_->adamFingerprint();

    std::ofstream csv(out_dir / "metrics.csv");
    csv << "arm,step,psnr,ssim,l1,eval_seconds,cum_sync_seconds,"
           "cum_train_seconds,total_seconds,gaussians,line_gaussians,"
           "iteration,active_sh,default_sh\n";

    auto lineCount = [&]() -> long long {
        if (!gaussians_->is_line_.defined() || gaussians_->is_line_.numel() == 0)
            return 0;
        return gaussians_->is_line_.to(torch::kCPU).sum().item<int64_t>();
    };

    auto doEval = [&](long long step) {
        double p = -1.0, s = -1.0, l1 = -1.0, ori = -1.0, shape = -1.0, et = 0.0;
        evalAtFrozenStep(step, tag, p, s, l1, ori, shape, et);
        r.eval_quality[step] = {p, s};
        csv << tag << ',' << step << ',' << std::setprecision(10) << p << ','
            << s << ',' << l1 << ',' << et << ',' << r.sync_seconds << ','
            << r.train_seconds << ',' << (r.sync_seconds + r.train_seconds)
            << ',' << gaussians_->xyz_.size(0) << ',' << lineCount() << ','
            << iteration_ << ',' << gaussians_->active_sh_degree_ << ','
            << default_sh_ << '\n';
        csv.flush();
    };

    doEval(0);
    std::size_t next_eval = 1;
    int stall = 0;
    while (frozen_opt_steps_ < steps && !isStopped())
    {
        syncDev();
        const auto t_step0 = std::chrono::steady_clock::now();
        const long long before = frozen_opt_steps_;
        trainForOneIterationErrorGuided();
        syncDev();
        r.train_seconds += std::chrono::duration_cast<std::chrono::microseconds>(
                               std::chrono::steady_clock::now() - t_step0).count() / 1e6;
        TORCH_CHECK(gaussians_->xyz_.size(0) == base_count,
                    "[GbaRecovery] arm ", tag, " changed the Gaussian count: ",
                    base_count, " -> ", gaussians_->xyz_.size(0));
        if (frozen_opt_steps_ == before)
        {
            if (++stall > 5) break;
        }
        else
        {
            stall = 0;
            if (steps <= 10)
                r.step_param_fp[frozen_opt_steps_] =
                    gaussians_->paramFingerprint();
        }
        while (next_eval < eval_steps.size() &&
               eval_steps[next_eval] <= frozen_opt_steps_)
        {
            if (eval_steps[next_eval] > 0 &&
                eval_steps[next_eval] == frozen_opt_steps_)
                doEval(frozen_opt_steps_);
            ++next_eval;
        }
    }
    r.steps = frozen_opt_steps_;
    r.gaussians = gaussians_->xyz_.size(0);
    r.line_gaussians = lineCount();
    r.param_fp = gaussians_->paramFingerprint();
    r.adam_fp = gaussians_->adamFingerprint();
    {
        std::ostringstream ss;
        ss << "it=" << iteration_ << ";sh=" << gaussians_->active_sh_degree_
           << "/" << default_sh_ << ";";
        for (const auto& kv : kfs_used_times_)
            ss << kv.first << ":" << kv.second << ",";
        ss << ";";
        for (const auto& kv : scene_->keyframes())
        {
            auto& pkf = kv.second;
            if (!pkf) continue;
            ss << kv.first << ":" << pkf->remaining_times_of_use_ << ":";
            for (int v : pkf->gaus_pyramid_times_of_use_) ss << v << "/";
            ss << ";";
        }
        ss << ";views=";
        for (int v : frozen_train_views_flat_) ss << v << ",";
        r.schedule_fp = frozenAblHashHex(ss.str());
    }
    r.ok = (r.steps == steps);
    return r;
}

// ---------------------------------------------------------------------------
// gba-recovery mechanism diagnosis (short, no 500-step matrix).
// ---------------------------------------------------------------------------
namespace {
struct PixelDispStats
{
    long long n_visible = 0;
    double mean = 0.0, p95 = 0.0, max = 0.0;
};

PixelDispStats computePixelDisp(const std::vector<Eigen::Vector3f>& before,
                                const std::vector<Eigen::Vector3f>& after,
                                const std::vector<std::uint8_t>& affected,
                                GaussianKeyframeLine& pkf)
{
    PixelDispStats s;
    if (pkf.intr_.size() < 4 || before.size() != after.size()) return s;
    const float fx = pkf.intr_[0], fy = pkf.intr_[1];
    const float cx = pkf.intr_[2], cy = pkf.intr_[3];
    const Sophus::SE3f Tcw = pkf.getPosef();
    const Eigen::Matrix3f R = Tcw.rotationMatrix();
    const Eigen::Vector3f t = Tcw.translation();
    auto proj = [&](const Eigen::Vector3f& X, float& u, float& v) -> bool {
        const Eigen::Vector3f Xc = R * X + t;
        if (Xc.z() <= 0.01f) return false;
        u = fx * Xc.x() / Xc.z() + cx;
        v = fy * Xc.y() / Xc.z() + cy;
        return u >= 0.0f && u < static_cast<float>(pkf.image_width_) &&
               v >= 0.0f && v < static_cast<float>(pkf.image_height_);
    };
    std::vector<float> ds;
    ds.reserve(before.size() / 8);
    double sum = 0.0;
    for (std::size_t i = 0; i < before.size(); ++i)
    {
        if (!affected.empty() && !affected[i]) continue;
        float u0, v0, u1, v1;
        if (!proj(before[i], u0, v0) || !proj(after[i], u1, v1)) continue;
        const float du = u1 - u0, dv = v1 - v0;
        const float d = std::sqrt(du * du + dv * dv);
        ds.push_back(d);
        sum += d;
        ++s.n_visible;
    }
    if (!ds.empty())
    {
        std::sort(ds.begin(), ds.end());
        s.mean = sum / static_cast<double>(ds.size());
        s.p95 = ds[std::min(ds.size() - 1,
                            static_cast<std::size_t>(0.95 * ds.size()))];
        s.max = ds.back();
    }
    return s;
}

std::vector<Eigen::Vector3f> tensorToPoints(const torch::Tensor& t)
{
    std::vector<Eigen::Vector3f> out;
    if (!t.defined()) return out;
    torch::Tensor c = t.detach().to(torch::kCPU).to(torch::kFloat).contiguous();
    auto a = c.accessor<float, 2>();
    out.resize(static_cast<std::size_t>(c.size(0)));
    for (int64_t i = 0; i < c.size(0); ++i)
        out[static_cast<std::size_t>(i)] = Eigen::Vector3f(a[i][0], a[i][1], a[i][2]);
    return out;
}
} // namespace

GaussianMapperLine::EvalViewRenders GaussianMapperLine::renderEvalViews()
{
    EvalViewRenders r;
    if (frozen_eval_kf_seq_.empty() || !gaussians_) return r;
    std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
    views.reserve(scene_->keyframes().size());
    for (auto& kv : scene_->keyframes()) views.push_back(kv.second);

    torch::NoGradGuard ng;
    torch::Tensor saved_radii, saved_hits, saved_alpha;
    const bool had_radii = gaussians_->max_radii2D_.defined();
    if (had_radii) saved_radii = gaussians_->max_radii2D_.clone();
    const bool had_hits = gaussians_->debug_hit_count_.defined();
    if (had_hits) saved_hits = gaussians_->debug_hit_count_.clone();
    const bool had_alpha = gaussians_->debug_alpha_accum_.defined();
    if (had_alpha) saved_alpha = gaussians_->debug_alpha_accum_.clone();

    for (std::size_t i = 0; i < frozen_eval_kf_seq_.size(); ++i)
    {
        const std::size_t idx = frozen_eval_kf_seq_[i];
        if (idx >= views.size() || !views[idx]) continue;
        auto pkf = views[idx];
        auto render_pkg = GaussianRendererWithLine::renderWithLine(
            pkf, pkf->image_height_, pkf->image_width_, gaussians_,
            pipe_params_, background_, override_color_);
        torch::Tensor masked =
            std::get<0>(render_pkg) * undistort_mask_[pkf->camera_id_];
        r.images.push_back(masked.detach().to(torch::kCPU).contiguous());
        r.psnr.push_back(loss_utils::psnr(masked, pkf->original_image_)
                             .item().toFloat());
        r.ssim.push_back(loss_utils::ssim(
                             masked, pkf->original_image_, device_type_)
                             .item().toFloat());
        r.fids.push_back(pkf->fid_);
    }

    if (had_radii && saved_radii.defined()) gaussians_->max_radii2D_.copy_(saved_radii);
    if (had_hits && saved_hits.defined()) gaussians_->debug_hit_count_.copy_(saved_hits);
    if (had_alpha && saved_alpha.defined()) gaussians_->debug_alpha_accum_.copy_(saved_alpha);
    return r;
}

GaussianMapperLine::DiagPoint GaussianMapperLine::captureDiagPoint(
    const std::string& label, bool with_grads)
{
    DiagPoint p;
    p.label = label;
    p.step = frozen_opt_steps_;
    p.iteration = iteration_;
    p.active_sh = gaussians_->active_sh_degree_;
    p.default_sh = default_sh_;
    auto& groups = gaussians_->optimizer_->param_groups();
    auto& state = gaussians_->optimizer_->state();
    for (std::size_t i = 0; i < groups.size() && i < 6; ++i)
    {
        if (groups[i].params().empty()) continue;
        auto& param = groups[i].params()[0];
        p.params.push_back(param.detach().to(torch::kCPU).clone());
        p.lrs.push_back(static_cast<torch::optim::AdamOptions&>(
                            groups[i].options()).lr());
        auto it = state.find(param.unsafeGetTensorImpl());
        p.adam_steps.push_back(
            it == state.end()
                ? -1LL
                : static_cast<long long>(
                      static_cast<torch::optim::AdamParamState&>(*it->second)
                          .step()));
        if (with_grads)
        {
            auto g = param.grad();
            p.grads.push_back(g.defined()
                                  ? g.detach().to(torch::kCPU).clone()
                                  : torch::Tensor());
        }
    }
    return p;
}

namespace
{
// Rotation matrix from (possibly unnormalised) quaternions, [N,4] -> [N,3,3].
torch::Tensor gbaRotationMatrix(const torch::Tensor& q_in)
{
    torch::Tensor q = q_in / q_in.norm(2, 1, true).clamp_min(1e-12f);
    torch::Tensor r = q.select(1, 0), x = q.select(1, 1), y = q.select(1, 2),
                  z = q.select(1, 3);
    torch::Tensor R = torch::zeros({q.size(0), 3, 3}, q.options());
    auto set = [&](int i, int j, torch::Tensor v) { R.select(1, i).select(1, j).copy_(v); };
    set(0, 0, 1 - 2 * (y * y + z * z));
    set(0, 1, 2 * (x * y - r * z));
    set(0, 2, 2 * (x * z + r * y));
    set(1, 0, 2 * (x * y + r * z));
    set(1, 1, 1 - 2 * (x * x + z * z));
    set(1, 2, 2 * (y * z - r * x));
    set(2, 0, 2 * (x * z - r * y));
    set(2, 1, 2 * (y * z + r * x));
    set(2, 2, 1 - 2 * (x * x + y * y));
    return R;
}

torch::Tensor gbaCovariance(const torch::Tensor& q, const torch::Tensor& scaling)
{
    torch::Tensor R = gbaRotationMatrix(q);
    torch::Tensor e = torch::exp(scaling.to(q.device()).to(q.dtype()));
    torch::Tensor RS = R * e.unsqueeze(1);
    return torch::bmm(RS, RS.transpose(1, 2));
}

// q and -q are the same rotation. The angle is computed in double precision
// from the relative quaternion, so identical float32 inputs give exactly 0
// (an acos(|dot|) formula has a ~0.1 deg floor on 75k rows).
double gbaQuatAngleDegMax(const torch::Tensor& a_in, const torch::Tensor& b_in)
{
    if (!a_in.defined() || !b_in.defined() || a_in.size(0) != b_in.size(0))
        return -1.0;
    // The relative rotation angle does not need normalised inputs: the scale
    // of a quaternion cancels in 2*atan2(|v|, |w|) of q_a^-1 * q_b. Keeping the
    // raw values makes identical inputs give exactly 0 instead of the
    // normalisation round-off floor.
    torch::Tensor ac = a_in.to(torch::kCPU).to(torch::kFloat32).contiguous();
    torch::Tensor bc = b_in.to(torch::kCPU).to(torch::kFloat32).contiguous();
    const float* qa = ac.data_ptr<float>();
    const float* qb = bc.data_ptr<float>();
    const long long n = ac.size(0);
    double worst = 0.0;
    auto dot4 = [](const float* x, const float* y) {
        return (double)x[0] * y[0] + (double)x[1] * y[1] +
               (double)x[2] * y[2] + (double)x[3] * y[3];
    };
    for (long long i = 0; i < n; ++i)
    {
        const float* x = qa + 4 * i;
        const float* y = qb + 4 * i;
        // relative quaternion q_a^-1 * q_b = (w, v)
        double w = dot4(x, y);
        double vx = x[0] * y[1] - x[1] * y[0] - x[2] * y[3] + x[3] * y[2];
        double vy = x[0] * y[2] + x[1] * y[3] - x[2] * y[0] - x[3] * y[1];
        double vz = x[0] * y[3] + x[2] * y[1] - x[3] * y[0] - x[1] * y[2];
        double nv = std::sqrt(vx * vx + vy * vy + vz * vz);
        double ang = 2.0 * std::atan2(nv, std::fabs(w));
        if (ang > worst) worst = ang;
    }
    return worst * 180.0 / M_PI;
}
}  // namespace

void GaussianMapperLine::writeAdamAudit(const char* stage)
{
    static const bool audit_enabled = []() {
        const char* p = std::getenv("PHOTO_SLAM_GBA_UPDATE_AUDIT");
        return p && p[0] != '\0';
    }();
    if (!audit_enabled || !gaussians_ || !gaussians_->optimizer_) return;
    if (adam_audit_updates_ >= 12) return;
    if (std::string(stage) == "step_entry")
    {
        if (iteration_ != adam_audit_last_iteration_)
        {
            adam_audit_last_iteration_ = iteration_;
            adam_audit_step_calls_this_iter_ = 0;
        }
        ++adam_audit_step_calls_this_iter_;
        ++adam_audit_step_calls_total_;
    }
    else
    {
        ++adam_audit_updates_;
    }
    if (!adam_audit_out_)
    {
        const char* path = std::getenv("PHOTO_SLAM_GBA_UPDATE_AUDIT");
        adam_audit_out_ = std::make_unique<std::ofstream>(path, std::ios::out);
        *adam_audit_out_
            << "stage,iteration,frozen_step,step_calls_total,"
               "step_calls_this_iter,group,member_name,member_impl,param_impl,"
               "same_impl,member_data,param_data,same_data_ptr,"
               "snapshot_rot_data,snapshot_shares_storage,shape0,shape1,"
               "stride0,stride1,dtype,device,requires_grad,is_leaf,grad_defined,"
               "grad_data,grad_l2,state_found,state_step,exp_avg_defined,"
               "exp_avg_l2,exp_avg_sq_l2,max_exp_avg_sq_defined,lr,beta1,beta2,"
               "eps,weight_decay,amsgrad,dup_param_count,group_param_count,"
               "rotation_norm_min,rotation_norm_max\n";
    }
    auto& groups = gaussians_->optimizer_->param_groups();
    auto& state = gaussians_->optimizer_->state();
    const std::vector<torch::Tensor> members = {
        gaussians_->xyz_, gaussians_->features_dc_, gaussians_->features_rest_,
        gaussians_->opacity_, gaussians_->scaling_, gaussians_->rotation_};
    const char* names[6] = {"xyz", "features_dc", "features_rest", "opacity",
                            "scaling", "rotation"};
    for (std::size_t i = 0; i < groups.size(); ++i)
    {
        if (groups[i].params().empty()) continue;
        auto& p = groups[i].params()[0];
        const torch::Tensor& m = (i < members.size()) ? members[i] : p;
        auto* pimpl = p.unsafeGetTensorImpl();
        std::uintptr_t pdata = reinterpret_cast<std::uintptr_t>(p.data_ptr());
        std::uintptr_t mdata = m.defined()
            ? reinterpret_cast<std::uintptr_t>(m.data_ptr()) : 0;
        auto g = p.grad();
        auto it = state.find(pimpl);
        auto& opt = static_cast<torch::optim::AdamOptions&>(groups[i].options());
        auto betas = opt.betas();
        double rot_min = 0.0, rot_max = 0.0;
        if (i == 5 && p.numel() > 0)
        {
            torch::NoGradGuard ng;
            auto n = torch::sqrt(torch::sum(p * p, 1));
            rot_min = n.min().item<double>();
            rot_max = n.max().item<double>();
        }
        int dup = 0;
        for (std::size_t j = 0; j < groups.size(); ++j)
            for (auto& q : groups[j].params())
                if (q.unsafeGetTensorImpl() == pimpl) ++dup;
        bool snap_shared = diag_snapshot_rot_data_ != 0 &&
                           pdata == diag_snapshot_rot_data_;
        *adam_audit_out_
            << stage << ',' << iteration_ << ',' << frozen_opt_steps_ << ','
            << adam_audit_step_calls_total_ << ','
            << adam_audit_step_calls_this_iter_ << ',' << i << ','
            << ((i < 6) ? names[i] : "extra") << ','
            << reinterpret_cast<std::uintptr_t>(m.defined()
                   ? m.unsafeGetTensorImpl() : nullptr) << ',' <<
            reinterpret_cast<std::uintptr_t>(pimpl) << ','
            << ((m.defined() && m.unsafeGetTensorImpl() == pimpl) ? 1 : 0)
            << ',' << mdata << ',' << pdata << ',' << (mdata == pdata ? 1 : 0)
            << ',' << diag_snapshot_rot_data_ << ',' << (snap_shared ? 1 : 0)
            << ',' << (p.dim() > 0 ? p.size(0) : -1) << ','
            << (p.dim() > 1 ? p.size(1) : -1) << ','
            << (p.dim() > 0 ? p.stride(0) : -1) << ','
            << (p.dim() > 1 ? p.stride(1) : -1) << ',' << p.dtype() << ','
            << p.device() << ',' << (p.requires_grad() ? 1 : 0) << ','
            << (p.is_leaf() ? 1 : 0) << ',' << (g.defined() ? 1 : 0) << ','
            << reinterpret_cast<std::uintptr_t>(g.defined() ? g.data_ptr()
                                                           : nullptr)
            << ','
            << (g.defined() ? g.norm().item<double>() : -1.0) << ','
            << ((it != state.end()) ? 1 : 0) << ','
            << ((it != state.end())
                    ? static_cast<long long>(
                          static_cast<torch::optim::AdamParamState&>(*it->second)
                              .step())
                    : -1LL)
            << ',';
        if (it != state.end())
        {
            auto& st = static_cast<torch::optim::AdamParamState&>(*it->second);
            *adam_audit_out_ << (st.exp_avg().defined() ? 1 : 0) << ','
                             << (st.exp_avg().defined()
                                     ? st.exp_avg().norm().item<double>()
                                     : -1.0)
                             << ','
                             << (st.exp_avg_sq().defined()
                                     ? st.exp_avg_sq().norm().item<double>()
                                     : -1.0)
                             << ',' << (st.max_exp_avg_sq().defined() ? 1 : 0);
        }
        else
        {
            *adam_audit_out_ << "0,-1,-1,0";
        }
        *adam_audit_out_ << ',' << opt.lr() << ',' << std::get<0>(betas) << ','
                         << std::get<1>(betas) << ',' << opt.eps() << ','
                         << opt.weight_decay() << ','
                         << (opt.amsgrad() ? 1 : 0) << ',' << dup << ','
                         << groups[i].params().size() << ',' << rot_min << ','
                         << rot_max << '\n';
        adam_audit_out_->flush();
    }
}

void GaussianMapperLine::runFirstUpdateReplay(const std::string& tag,
                                              const RotationAdamSample& s,
                                              const DiagArm& arm)
{
    if (!s.valid || !s.param.defined()) return;
    const torch::Tensor empty;
    torch::Tensor actual;
    for (const auto& pt : arm.points)
        if (pt.label == "first_update" && pt.params.size() >= 6)
            actual = pt.params[5];

    auto toDev = [&](const torch::Tensor& t) {
        return t.defined() ? t.to(device_type_).clone() : torch::Tensor();
    };

    std::vector<torch::Tensor> after_step, after_norm;
    for (int rep = 0; rep < 5; ++rep)
    {
        torch::Tensor p = s.param.to(device_type_).clone().detach();
        p.set_requires_grad(true);
        if (s.grad_defined && s.grad.defined())
            p.mutable_grad() = toDev(s.grad);

        torch::optim::AdamOptions opts(s.lr);
        opts.betas(std::make_tuple(s.beta1, s.beta2));
        opts.eps(s.eps);
        opts.weight_decay(s.weight_decay);
        opts.amsgrad(s.amsgrad);
        torch::optim::Adam adam({p}, opts);
        auto key = p.unsafeGetTensorImpl();
        auto st = std::make_unique<torch::optim::AdamParamState>();
        if (s.state_found)
        {
            st->step(s.step);
            if (s.exp_avg.defined()) st->exp_avg(toDev(s.exp_avg));
            if (s.exp_avg_sq.defined()) st->exp_avg_sq(toDev(s.exp_avg_sq));
            if (s.max_exp_avg_sq.defined())
                st->max_exp_avg_sq(toDev(s.max_exp_avg_sq));
        }
        adam.state()[key] = std::move(st);
        adam.step();
        if (device_type_ == torch::kCUDA) torch::cuda::synchronize();
        after_step.push_back(p.detach().to(torch::kCPU).clone());
        torch::Tensor q = after_step.back();
        torch::Tensor n =
            torch::sqrt(torch::sum(q * q, 1, true).clamp_min(1e-16f));
        after_norm.push_back((q / n).clone());
    }

    std::ofstream csv((diag_root_.empty() ? result_dir_.parent_path() : diag_root_) /
                      "first_update_replay.csv");
    csv << "tag,stage,replay,state_found,state_step,lr,beta1,beta2,eps,"
           "weight_decay,amsgrad,param_impl,member_impl,same_impl,"
           "grad_defined,grad_l2,raw_max_abs_vs_rep0,quat_angle_deg_vs_rep0,"
           "cov_max_abs_vs_rep0\n";
    auto emit = [&](const char* stage, int rep, const torch::Tensor& cur,
                    const torch::Tensor& base) {
        double raw = -1.0, ang = -1.0, cov = -1.0;
        if (rep > 0)
        {
            raw = (cur - base).abs().max().item<double>();
            ang = gbaQuatAngleDegMax(cur, base);
            cov = (gbaCovariance(cur, s.scaling) -
                   gbaCovariance(base, s.scaling))
                      .abs()
                      .max()
                      .item<double>();
        }
        csv << tag << ',' << stage << ',' << rep << ','
            << (s.state_found ? 1 : 0) << ',' << s.step << ',' << s.lr << ','
            << s.beta1 << ',' << s.beta2 << ',' << s.eps << ','
            << s.weight_decay << ',' << (s.amsgrad ? 1 : 0) << ','
            << s.param_impl << ',' << s.member_impl << ','
            << (s.param_impl == s.member_impl ? 1 : 0) << ','
            << (s.grad_defined ? 1 : 0) << ','
            << (s.grad.defined() ? s.grad.norm().item<double>() : -1.0) << ','
            << raw << ',' << ang << ',' << cov << '\n';
    };
    for (int rep = 0; rep < 5; ++rep) emit("entry", rep, s.param, s.param);
    for (int rep = 0; rep < 5; ++rep)
        emit("after_step", rep, after_step[rep], after_step[0]);
    for (int rep = 0; rep < 5; ++rep)
        emit("after_normalize", rep, after_norm[rep], after_norm[0]);
    csv.flush();

    std::ofstream act((diag_root_.empty() ? result_dir_.parent_path() : diag_root_) /
                      "first_update_actual.csv", std::ios::app);
    if (act.tellp() == 0)
        act << "tag,param_impl,member_impl,same_impl,grad_defined,state_found,"
           "state_step,replay_repeat_raw_max_abs,replay_repeat_quat_deg,"
           "replay_repeat_cov_max_abs,actual_vs_replay_raw_max_abs,"
           "actual_vs_replay_quat_deg,actual_vs_replay_cov_max_abs,"
           "actual_normalized_vs_replay_quat_deg\n";
    double rep_raw = 0.0, rep_ang = 0.0, rep_cov = 0.0;
    for (int rep = 1; rep < 5; ++rep)
    {
        rep_raw = std::max(rep_raw,
                           (after_step[rep] - after_step[0]).abs().max()
                               .item<double>());
        rep_ang = std::max(rep_ang,
                           gbaQuatAngleDegMax(after_step[rep], after_step[0]));
        rep_cov = std::max(
            rep_cov, (gbaCovariance(after_step[rep], s.scaling) -
                      gbaCovariance(after_step[0], s.scaling))
                         .abs().max().item<double>());
    }
    double a_raw = -1.0, a_ang = -1.0, a_cov = -1.0, a_norm_ang = -1.0;
    if (actual.defined() && actual.size(0) == after_step[0].size(0))
    {
        a_raw = (actual - after_step[0]).abs().max().item<double>();
        a_ang = gbaQuatAngleDegMax(actual, after_step[0]);
        a_cov = (gbaCovariance(actual, s.scaling) -
                 gbaCovariance(after_step[0], s.scaling))
                    .abs().max().item<double>();
        torch::Tensor n = torch::sqrt(
            torch::sum(actual * actual, 1, true).clamp_min(1e-16f));
        a_norm_ang =
            gbaQuatAngleDegMax(actual / n, after_norm[0]);
    }
    act << tag << ',' << s.param_impl << ',' << s.member_impl << ','
        << (s.param_impl == s.member_impl ? 1 : 0) << ','
        << (s.grad_defined ? 1 : 0) << ',' << (s.state_found ? 1 : 0) << ','
        << s.step << ',' << rep_raw << ',' << rep_ang << ',' << rep_cov << ','
        << a_raw << ',' << a_ang << ',' << a_cov << ',' << a_norm_ang << '\n';
    act.flush();
    std::cerr << "[GbaDiag] replay " << tag << " repeat_raw=" << rep_raw
              << " repeat_quat_deg=" << rep_ang << " repeat_cov=" << rep_cov
              << " actual_raw=" << a_raw << " actual_quat_deg=" << a_ang
              << " actual_cov=" << a_cov
              << " same_impl=" << (s.param_impl == s.member_impl) << std::endl;
}

void GaussianMapperLine::writeCrossArmInputDiff(const std::string& tag,
                                                const RotationAdamSample& s)
{
    if (!first_bw_rot_ref_.valid)
    {
        first_bw_rot_ref_ = s;
        return;
    }
    const RotationAdamSample& a = first_bw_rot_ref_;
    if (!a.param.defined() || !s.param.defined() ||
        a.param.sizes() != s.param.sizes())
        return;
    auto maxabs = [](const torch::Tensor& x, const torch::Tensor& y) {
        if (!x.defined() || !y.defined() || x.sizes() != y.sizes()) return -1.0;
        return (x - y).abs().max().item<double>();
    };
    torch::Tensor base = a.scaling.defined() ? a.scaling : s.scaling;
    std::ofstream csv((diag_root_.empty() ? result_dir_.parent_path() : diag_root_) /
                      "first_update_cross.csv");
    csv << "tag,arm_ref,param_max_abs,grad_max_abs,exp_avg_max_abs,"
           "exp_avg_sq_max_abs,state_step_ref,state_step_tag,"
           "quat_angle_deg,cov_max_abs,lr_ref,lr_tag,eps_ref,eps_tag,"
           "wd_ref,wd_tag,beta1_ref,beta1_tag,same_impl_ref,same_impl_tag,"
           "grad_defined_ref,grad_defined_tag,state_found_ref,state_found_tag\n";
    double cov = -1.0;
    if (base.defined() && base.size(0) == s.param.size(0))
        cov = (gbaCovariance(s.param, base) - gbaCovariance(a.param, base))
                  .abs().max().item<double>();
    csv << tag << ",ref," << maxabs(a.param, s.param) << ','
        << maxabs(a.grad, s.grad) << ',' << maxabs(a.exp_avg, s.exp_avg) << ','
        << maxabs(a.exp_avg_sq, s.exp_avg_sq) << ',' << a.step << ',' << s.step
        << ',' << gbaQuatAngleDegMax(a.param, s.param) << ',' << cov << ','
        << a.lr << ',' << s.lr << ',' << a.eps << ',' << s.eps << ','
        << a.weight_decay << ',' << s.weight_decay << ',' << a.beta1 << ','
        << s.beta1 << ',' << (a.param_impl == a.member_impl ? 1 : 0) << ','
        << (s.param_impl == s.member_impl ? 1 : 0) << ','
        << (a.grad_defined ? 1 : 0) << ',' << (s.grad_defined ? 1 : 0) << ','
        << (a.state_found ? 1 : 0) << ',' << (s.state_found ? 1 : 0) << '\n';
    csv.flush();
    std::cerr << "[GbaDiag] cross-arm input: param=" << maxabs(a.param, s.param)
              << " grad=" << maxabs(a.grad, s.grad)
              << " exp_avg=" << maxabs(a.exp_avg, s.exp_avg)
              << " exp_avg_sq=" << maxabs(a.exp_avg_sq, s.exp_avg_sq)
              << " quat_deg=" << gbaQuatAngleDegMax(a.param, s.param)
              << " cov=" << cov << std::endl;
}

std::string GaussianMapperLine::scheduleFingerprint() const
{    std::ostringstream ss;
    ss << iteration_ << ';' << gaussians_->active_sh_degree_ << '/'
       << default_sh_ << ';';
    for (const auto& kv : kfs_used_times_) ss << kv.first << ':' << kv.second << ',';
    ss << ';';
    for (const auto& kv : scene_->keyframes())
    {
        auto& pkf = kv.second;
        if (!pkf) continue;
        ss << kv.first << ':' << pkf->remaining_times_of_use_ << ':'
           << pkf->tracking_error_ << ':';
        for (int v : pkf->gaus_pyramid_times_of_use_) ss << v << '/';
        Eigen::Matrix4f M = pkf->getPosef().matrix();
        const std::uint32_t* w = reinterpret_cast<const std::uint32_t*>(M.data());
        for (int k = 0; k < 16; ++k) ss << std::hex << w[k];
        ss << std::dec << ';';
    }
    for (int v : frozen_train_views_flat_) ss << v << ',';
    return frozenAblHashHex(ss.str());
}

GaussianMapperLine::EvalViewRenders GaussianMapperLine::runStep0DiagArm(
    const std::string& tag, const RecoverySnapshot& base,
    const RecoverySnapshot& kf_usage,
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
    unsigned long evt, int mode)
{
    restoreRecoverySnapshot(base);
    for (auto& kv : kf_usage.kfs)
    {
        auto pkf = scene_->getKeyframe(kv.first);
        if (!pkf) continue;
        pkf->remaining_times_of_use_ = kv.second.remaining_times_of_use;
        pkf->gaus_pyramid_times_of_use_ = kv.second.pyramid_times_of_use;
        pkf->tracking_error_ = kv.second.tracking_error;
    }
    frozen_fixed_count_ = true;
    frozen_kf_replay_ = true;
    frozen_opt_steps_ = 0;
    result_dir_ = std::filesystem::path(result_dir_) / tag;
    std::filesystem::create_directories(result_dir_);

    for (const auto& p : after_poses)
    {
        auto pkf = scene_->getKeyframe(p.first);
        if (!pkf) continue;
        pkf->setPose(p.second.unit_quaternion().cast<double>(),
                     p.second.translation().cast<double>());
        pkf->computeTransformTensors();
    }
    if (mode == 1)
        gaussians_->applyGbaCorrections(world_deltas, evt,
                                        GaussianModelLine::GbaCorrectionMode::MeansOnly);
    else if (mode == 2)
        gaussians_->applyGbaCorrections(world_deltas, evt);

    EvalViewRenders r = renderEvalViews();
    double p, s, l1, ori, shape, et;
    evalAtFrozenStep(0, tag, p, s, l1, ori, shape, et);
    return r;
}

GaussianMapperLine::DiagArm GaussianMapperLine::runRepeatDiagArm(
    const std::string& tag, const RecoverySnapshot& base,
    const RecoverySnapshot& kf_usage,
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses)
{
    DiagArm arm;
    restoreRecoverySnapshot(base);
    diag_snapshot_rot_data_ = base.rotation.defined()
        ? reinterpret_cast<std::uintptr_t>(base.rotation.data_ptr()) : 0;
    first_bw_rot_ = RotationAdamSample();
    for (auto& kv : kf_usage.kfs)
    {
        auto pkf = scene_->getKeyframe(kv.first);
        if (!pkf) continue;
        pkf->remaining_times_of_use_ = kv.second.remaining_times_of_use;
        pkf->gaus_pyramid_times_of_use_ = kv.second.pyramid_times_of_use;
        pkf->tracking_error_ = kv.second.tracking_error;
    }
    frozen_fixed_count_ = true;
    frozen_kf_replay_ = true;
    frozen_kf_step_ = 0;
    frozen_train_views_flat_.clear();
    frozen_opt_steps_ = 0;
    result_dir_ = std::filesystem::path(result_dir_) / tag;
    std::filesystem::create_directories(result_dir_);
    for (const auto& p : after_poses)
    {
        auto pkf = scene_->getKeyframe(p.first);
        if (!pkf) continue;
        pkf->setPose(p.second.unit_quaternion().cast<double>(),
                     p.second.translation().cast<double>());
        pkf->computeTransformTensors();
    }
    arm.param_fp0 = gaussians_->paramFingerprint();
    arm.adam_fp0 = gaussians_->adamFingerprint();
    arm.schedule_fp0 = scheduleFingerprint();
    arm.points.push_back(captureDiagPoint("step0", false));
    arm.render0 = renderEvalViews();

    recovery_diag_hook_ = [&](const char* ev) {
        if (std::string(ev) == "after_backward" && frozen_opt_steps_ == 0)
        {
            arm.points.push_back(captureDiagPoint("first_backward", true));
            if (!first_bw_rot_.valid)
            {
                // gba-state-audit: take the state from the ACTUAL parameter
                // object (param group), not from the model member.
                auto& groups = gaussians_->optimizer_->param_groups();
                auto& state = gaussians_->optimizer_->state();
                RotationAdamSample s;
                if (groups.size() > 5 && !groups[5].params().empty())
                {
                    auto& p = groups[5].params()[0];
                    auto& opt =
                        static_cast<torch::optim::AdamOptions&>(groups[5].options());
                    auto betas = opt.betas();
                    s.valid = true;
                    s.param = p.detach().to(torch::kCPU).clone();
                    auto g = p.grad();
                    s.grad_defined = g.defined();
                    if (g.defined()) s.grad = g.detach().to(torch::kCPU).clone();
                    auto it = state.find(p.unsafeGetTensorImpl());
                    s.state_found = (it != state.end());
                    if (s.state_found)
                    {
                        auto& st = static_cast<torch::optim::AdamParamState&>(
                            *it->second);
                        s.step = static_cast<long long>(st.step());
                        if (st.exp_avg().defined())
                            s.exp_avg = st.exp_avg().detach().to(torch::kCPU).clone();
                        if (st.exp_avg_sq().defined())
                            s.exp_avg_sq =
                                st.exp_avg_sq().detach().to(torch::kCPU).clone();
                        if (st.max_exp_avg_sq().defined())
                            s.max_exp_avg_sq = st.max_exp_avg_sq()
                                                   .detach().to(torch::kCPU).clone();
                    }
                    if (gaussians_->scaling_.defined())
                        s.scaling =
                            gaussians_->scaling_.detach().to(torch::kCPU).clone();
                    s.param_impl = reinterpret_cast<std::uintptr_t>(
                        p.unsafeGetTensorImpl());
                    s.param_data =
                        reinterpret_cast<std::uintptr_t>(p.data_ptr());
                    s.member_impl = reinterpret_cast<std::uintptr_t>(
                        gaussians_->rotation_.unsafeGetTensorImpl());
                    s.member_data = reinterpret_cast<std::uintptr_t>(
                        gaussians_->rotation_.data_ptr());
                    s.lr = opt.lr();
                    s.beta1 = std::get<0>(betas);
                    s.beta2 = std::get<1>(betas);
                    s.eps = opt.eps();
                    s.weight_decay = opt.weight_decay();
                    s.amsgrad = opt.amsgrad();
                    first_bw_rot_ = s;
                    std::cerr << "[GbaDiag] rotation sample tag=" << tag
                              << " same_impl="
                              << (s.param_impl == s.member_impl)
                              << " grad_defined=" << s.grad_defined
                              << " state_found=" << s.state_found
                              << " adam_step=" << s.step << std::endl;
                    writeCrossArmInputDiff(tag, s);
                }
            }
        }
        if (std::string(ev) == "after_update")
        {
            if (frozen_opt_steps_ == 1)
                arm.points.push_back(captureDiagPoint("first_update", false));
            if (frozen_opt_steps_ == 10)
                arm.points.push_back(captureDiagPoint("step10", false));
        }
    };
    auto syncDev = [&]() {
        if (device_type_ == torch::kCUDA && gaussians_->xyz_.defined() &&
            gaussians_->xyz_.numel() > 0)
            (void)gaussians_->xyz_.select(0, 0).sum().item<double>();
    };
    while (frozen_opt_steps_ < 1 && !isStopped())
    {
        syncDev();
        trainForOneIterationErrorGuided();
        syncDev();
    }
    arm.render1 = renderEvalViews();
    while (frozen_opt_steps_ < 10 && !isStopped())
    {
        syncDev();
        trainForOneIterationErrorGuided();
        syncDev();
    }
    arm.render10 = renderEvalViews();
    recovery_diag_hook_ = nullptr;
    runFirstUpdateReplay(tag, first_bw_rot_, arm);
    first_bw_rot_ = RotationAdamSample();
    arm.view_seq = frozen_train_views_flat_;
    arm.ok = (frozen_opt_steps_ == 10);
    return arm;
}

void GaussianMapperLine::runGbaRecoveryDiagnosis(
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
    unsigned long evt)
{
    const char* dir_env = std::getenv("PHOTO_SLAM_GBA_RECOVERY_DIAG");
    if (!dir_env || dir_env[0] == '\0' || gba_recovery_done_ || !gaussians_)
        return;
    first_bw_rot_ref_ = RotationAdamSample();
    diag_root_ = std::filesystem::path(dir_env);
    gba_recovery_done_ = true;
    const std::filesystem::path root(dir_env);
    std::filesystem::create_directories(root);

    const auto saved_result_dir = result_dir_;
    const bool saved_fixed = frozen_fixed_count_;
    const bool saved_replay = frozen_kf_replay_;
    const std::size_t saved_kf_step = frozen_kf_step_;
    const auto saved_eval_seq = frozen_eval_kf_seq_;
    const auto saved_kf_seq = frozen_kf_seq_;
    const auto saved_train_views = frozen_train_views_flat_;
    const float saved_w0 = opt_params_.weight_line_coherence_;
    const float saved_w1 = opt_params_.weight_line_shape_ecc_;
    const float saved_w2 = opt_params_.weight_line_shape_ori_;
    const int saved_opacity = opt_params_.opacity_reset_interval_;

    frozen_eval_kf_seq_.clear();
    {
        const std::size_t nkf = scene_->keyframes().size();
        int want = 24;
        if (const char* e = std::getenv("PHOTO_SLAM_GBA_RECOVERY_VIEWS"))
            want = std::atoi(e);
        if (want > 0 && nkf > 0)
        {
            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(want), nkf);
            for (std::size_t i = 0; i < k; ++i)
                frozen_eval_kf_seq_.push_back((i * nkf) / k);
        }
    }

    RecoverySnapshot common = captureRecoverySnapshot();
    preGenerateFrozenKfSequence(10);
    RecoverySnapshot usage;
    for (auto& kv : scene_->keyframes())
    {
        auto& pkf = kv.second;
        if (!pkf) continue;
        RecoveryKfState k;
        k.remaining_times_of_use = pkf->remaining_times_of_use_;
        k.pyramid_times_of_use = pkf->gaus_pyramid_times_of_use_;
        k.tracking_error = pkf->tracking_error_;
        usage.kfs[kv.first] = k;
    }
    {
        std::ofstream st(root / "common_state.txt");
        st << "event_evt=" << evt << "\n"
           << "gaussians=" << gaussians_->xyz_.size(0) << "\n"
           << "line_gaussians="
           << (gaussians_->is_line_.defined()
                   ? gaussians_->is_line_.to(torch::kCPU).sum().item<int64_t>()
                   : 0)
           << "\nkfs=" << scene_->keyframes().size() << "\n"
           << "iteration=" << iteration_ << "\n"
           << "active_sh=" << gaussians_->active_sh_degree_ << "\n"
           << "default_sh=" << default_sh_ << "\n"
           << "param_fp=" << gaussians_->paramFingerprint() << "\n"
           << "adam_fp=" << gaussians_->adamFingerprint() << "\n";
    }
    {
        std::ofstream ev(root / "event_evt.txt");
        ev << "# kf_id, delta rotation/translation\n";
        for (const auto& d : world_deltas)
            ev << d.first << ',' << d.second.so3().log().norm() * 180.0 / M_PI
               << ',' << d.second.translation().norm() << '\n';
    }

    frozen_fixed_count_ = true;
    frozen_kf_replay_ = true;
    opt_params_.weight_line_coherence_ = 0.0f;
    opt_params_.weight_line_shape_ecc_ = 0.0f;
    opt_params_.weight_line_shape_ori_ = 0.0f;
    opt_params_.opacity_reset_interval_ = 0;
    SetRasterBackend(2);

    // --- step-0 deformation decomposition (no optimizer.step) ---
    auto po0 = runStep0DiagArm("PoseOnly", common, usage, after_poses,
                               world_deltas, evt, 0);
    const std::vector<Eigen::Vector3f> xyz_pre = tensorToPoints(common.xyz);
    {
        // Non-event rows: visibility under the corrected camera poses.
        std::unordered_set<unsigned long> event_ids;
        for (const auto& d : world_deltas) event_ids.insert(d.first);
        const std::size_t nrows = common.ref_kf_id.defined()
                                      ? static_cast<std::size_t>(common.ref_kf_id.numel())
                                      : 0;
        std::vector<std::uint8_t> unassigned(nrows, 0), scene_not_event(nrows, 0);
        if (common.ref_kf_id.defined())
        {
            auto ids = common.ref_kf_id.to(torch::kCPU).contiguous();
            auto a = ids.accessor<int64_t, 1>();
            for (int64_t i = 0; i < ids.size(0); ++i)
            {
                const std::size_t k = static_cast<std::size_t>(i);
                if (a[i] < 0) unassigned[k] = 1;
                else if (!event_ids.count(static_cast<unsigned long>(a[i])))
                    scene_not_event[k] = 1;
            }
        }
        std::vector<std::uint8_t> uncorrected(nrows, 0);
        long long n_unassigned = 0, n_scene_not_event = 0;
        for (std::size_t i = 0; i < nrows; ++i)
        {
            uncorrected[i] = (unassigned[i] || scene_not_event[i]) ? 1 : 0;
            n_unassigned += unassigned[i];
            n_scene_not_event += scene_not_event[i];
        }
        std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
        for (auto& kv : scene_->keyframes()) views.push_back(kv.second);
        std::ofstream vf(root / "non_event_visibility.csv");
        vf << "# rows: unassigned=" << n_unassigned
           << " scene_not_event=" << n_scene_not_event << "\n";
        vf << "view,fid,n_unassigned_visible,n_scene_not_event_visible,"
              "n_total_uncorrected_visible\n";
        for (std::size_t i = 0; i < frozen_eval_kf_seq_.size(); ++i)
        {
            const std::size_t idx = frozen_eval_kf_seq_[i];
            if (idx >= views.size() || !views[idx]) continue;
            auto su = computePixelDisp(xyz_pre, xyz_pre, unassigned, *views[idx]);
            auto ss = computePixelDisp(xyz_pre, xyz_pre, scene_not_event, *views[idx]);
            auto st = computePixelDisp(xyz_pre, xyz_pre, uncorrected, *views[idx]);
            vf << i << ',' << views[idx]->fid_ << ',' << su.n_visible << ','
               << ss.n_visible << ',' << st.n_visible << '\n';
        }
    }

    auto mo0 = runStep0DiagArm("MeansOnly", common, usage, after_poses,
                               world_deltas, evt, 1);
    auto fu0 = runStep0DiagArm("Full", common, usage, after_poses,
                               world_deltas, evt, 2);
    const std::vector<Eigen::Vector3f> xyz_post = tensorToPoints(gaussians_->xyz_);
    {
        std::ofstream cf(root / "step0_compare.csv");
        cf << "view,fid,psnr_poseonly,psnr_means,psnr_full,dpsnr_means,dpsnr_full,"
              "ssim_poseonly,ssim_means,ssim_full,dssim_means,dssim_full,"
              "render_maxdiff_means,render_maxdiff_full,"
              "render_meandiff_means,render_meandiff_full\n";
        for (std::size_t i = 0; i < po0.psnr.size() && i < mo0.psnr.size() &&
                                i < fu0.psnr.size(); ++i)
        {
            auto diff = [&](const EvalViewRenders& x) {
                torch::Tensor d = (x.images[i] - po0.images[i]).abs();
                return std::make_pair(d.max().item<float>(),
                                      d.mean().item<float>());
            };
            auto dm = diff(mo0), df = diff(fu0);
            cf << i << ',' << po0.fids[i] << ',' << po0.psnr[i] << ','
               << mo0.psnr[i] << ',' << fu0.psnr[i] << ','
               << (mo0.psnr[i] - po0.psnr[i]) << ','
               << (fu0.psnr[i] - po0.psnr[i]) << ',' << po0.ssim[i] << ','
               << mo0.ssim[i] << ',' << fu0.ssim[i] << ','
               << (mo0.ssim[i] - po0.ssim[i]) << ','
               << (fu0.ssim[i] - po0.ssim[i]) << ',' << dm.first << ','
               << df.first << ',' << dm.second << ',' << df.second << '\n';
        }
    }

    // --- repeat-run diagnosis: two PoseOnly arms, 10 steps ---
    auto repA = runRepeatDiagArm("RepeatA", common, usage, after_poses);
    auto repB = runRepeatDiagArm("RepeatB", common, usage, after_poses);

    auto groupName = [](int i) {
        static const char* n[6] = {"xyz", "features_dc", "features_rest",
                                   "opacity", "scaling", "rotation"};
        return i >= 0 && i < 6 ? n[i] : "?";
    };
    auto relL2 = [](const torch::Tensor& a, const torch::Tensor& b) {
        double nb = b.norm().item<double>();
        return (a - b).norm().item<double>() / std::max(nb, 1e-12);
    };
    auto quatAngleDeg = [](const torch::Tensor& a, const torch::Tensor& b) {
        torch::Tensor A = a.to(torch::kFloat) /
                          a.norm(2, 1, true).clamp_min(1e-12f);
        torch::Tensor B = b.to(torch::kFloat) /
                          b.norm(2, 1, true).clamp_min(1e-12f);
        torch::Tensor dot = (A * B).sum(1).abs().clamp_max(1.0f);
        return (2.0 * torch::acos(dot)).max().item<double>() * 180.0 / M_PI;
    };
    {
        std::ofstream df(root / "repeat_diag.csv");
        df << "stage,group,max_abs,rel_l2,value_min,value_max\n";
        for (const auto& pa : repA.points)
        {
            for (const auto& pb : repB.points)
            {
                if (pa.label != pb.label) continue;
                for (std::size_t i = 0; i < pa.params.size() &&
                                        i < pb.params.size(); ++i)
                {
                    auto& a = pa.params[i];
                    auto& b = pb.params[i];
                    df << pa.label << ',' << groupName(static_cast<int>(i)) << ','
                       << (a - b).abs().max().item<double>() << ','
                       << relL2(a, b) << ',' << b.min().item<double>() << ','
                       << b.max().item<double>() << '\n';
                }
                if (pa.grads.size() == pb.grads.size())
                {
                    for (std::size_t i = 0; i < pa.grads.size(); ++i)
                    {
                        if (!pa.grads[i].defined() || !pb.grads[i].defined())
                            continue;
                        df << pa.label << ",grad_" << groupName(static_cast<int>(i))
                           << ',' << (pa.grads[i] - pb.grads[i]).abs().max().item<double>()
                           << ',' << relL2(pa.grads[i], pb.grads[i]) << ','
                           << pb.grads[i].min().item<double>() << ','
                           << pb.grads[i].max().item<double>() << '\n';
                    }
                }
                break;
            }
        }
        // Displacement / scale / opacity / quaternion (sign-invariant).
        for (const auto& pa : repA.points)
        {
            for (const auto& pb : repB.points)
            {
                if (pa.label != pb.label) continue;
                if (pa.params.size() > 0 && pb.params.size() > 0)
                {
                    auto d = (pa.params[0] - pb.params[0]).norm(2, 1);
                    df << pa.label << ",xyz_disp_mean,"
                       << d.mean().item<double>() << ",xyz_disp_max,"
                       << d.max().item<double>() << ",0\n";
                }
                if (pa.params.size() > 3 && pb.params.size() > 3)
                {
                    auto oa = torch::sigmoid(pa.params[3]);
                    auto ob = torch::sigmoid(pb.params[3]);
                    df << pa.label << ",opacity_abs_change,"
                       << (oa - ob).abs().max().item<double>() << ","
                       << (oa - ob).abs().mean().item<double>() << ",0,0\n";
                }
                if (pa.params.size() > 4 && pb.params.size() > 4)
                {
                    auto sa = torch::exp(pa.params[4]);
                    auto sb = torch::exp(pb.params[4]);
                    df << pa.label << ",scale_abs_change,"
                       << (sa - sb).abs().max().item<double>() << ","
                       << (sa - sb).abs().mean().item<double>() << ",0,0\n";
                }
                if (pa.params.size() > 5 && pb.params.size() > 5)
                    df << pa.label << ",quat_angle_deg,"
                       << quatAngleDeg(pa.params[5], pb.params[5]) << ",0,0,0\n";
                break;
            }
        }
    }
    {
        std::ofstream rf(root / "repeat_renders.csv");
        rf << "stage,view,fid,psnr_A,psnr_B,dpsnr,ssim_A,ssim_B,dssim,"
              "render_maxdiff,render_meandiff\n";
        auto dump = [&](const char* stage, const EvalViewRenders& a,
                        const EvalViewRenders& b) {
            for (std::size_t i = 0; i < a.psnr.size() && i < b.psnr.size(); ++i)
            {
                torch::Tensor d = (a.images[i] - b.images[i]).abs();
                rf << stage << ',' << i << ',' << a.fids[i] << ',' << a.psnr[i]
                   << ',' << b.psnr[i] << ',' << (b.psnr[i] - a.psnr[i]) << ','
                   << a.ssim[i] << ',' << b.ssim[i] << ','
                   << (b.ssim[i] - a.ssim[i]) << ','
                   << d.max().item<float>() << ',' << d.mean().item<float>()
                   << '\n';
            }
        };
        dump("step0", repA.render0, repB.render0);
        dump("first_update", repA.render1, repB.render1);
        dump("step10", repA.render10, repB.render10);
    }
    {
        std::ofstream cf(root / "repeat_consistency.csv");
        cf << "check,value\n";
        cf << "step0_param_fp_equal,"
           << (repA.param_fp0 == repB.param_fp0 ? 1 : 0) << '\n';
        cf << "step0_adam_fp_equal,"
           << (repA.adam_fp0 == repB.adam_fp0 ? 1 : 0) << '\n';
        cf << "schedule_fp_equal,"
           << (repA.schedule_fp0 == repB.schedule_fp0 ? 1 : 0) << '\n';
        cf << "view_sequence_equal,"
           << (repA.view_seq == repB.view_seq ? 1 : 0) << '\n';
        int lr_equal = 1, adam_step_equal = 1;
        for (const auto& pa : repA.points)
            for (const auto& pb : repB.points)
                if (pa.label == "step0" && pb.label == "step0")
                {
                    lr_equal = (pa.lrs == pb.lrs) ? 1 : 0;
                    adam_step_equal = (pa.adam_steps == pb.adam_steps) ? 1 : 0;
                }
        cf << "lr_equal," << lr_equal << '\n';
        cf << "adam_step_equal," << adam_step_equal << '\n';
    }

    // --- correction scope: GBA and LOOP ---
    {
        std::ofstream sf(root / "gba_scope_kf.csv");
        sf << "kf_id,rot_deg,trans\n";
        for (const auto& d : world_deltas)
            sf << d.first << ','
               << d.second.so3().log().norm() * 180.0 / M_PI << ','
               << d.second.translation().norm() << '\n';
    }
    {
        std::vector<std::uint8_t> moved(xyz_pre.size(), 0);
        for (std::size_t i = 0; i < xyz_pre.size() && i < xyz_post.size(); ++i)
            moved[i] = (xyz_post[i] - xyz_pre[i]).norm() > 1e-6f ? 1 : 0;
        std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
        for (auto& kv : scene_->keyframes()) views.push_back(kv.second);
        std::ofstream vf(root / "gba_scope_views.csv");
        vf << "view,fid,n_moved_visible,pix_mean,pix_p95,pix_max\n";
        for (std::size_t i = 0; i < frozen_eval_kf_seq_.size(); ++i)
        {
            const std::size_t idx = frozen_eval_kf_seq_[i];
            if (idx >= views.size() || !views[idx]) continue;
            auto s = computePixelDisp(xyz_pre, xyz_post, moved, *views[idx]);
            vf << i << ',' << views[idx]->fid_ << ',' << s.n_visible << ','
               << s.mean << ',' << s.p95 << ',' << s.max << '\n';
        }
    }
    {
        std::ofstream lf(root / "loop_scope.csv");
        if (!loop_scope_available_)
        {
            lf << "available,0\n";
        }
        else
        {
            lf << "available,1\n";
            lf << "kf_id,rot_deg,trans,scene_present\n";
            for (const auto& e : loop_scope_kfs_)
                lf << e.kf_id << ',' << e.rot_deg << ',' << e.trans << ','
                   << (e.scene_present ? 1 : 0) << '\n';
            std::vector<std::shared_ptr<GaussianKeyframeLine>> views;
            for (auto& kv : scene_->keyframes()) views.push_back(kv.second);
            std::ofstream vf(root / "loop_scope_views.csv");
            vf << "view,fid,n_visible,pix_mean,pix_p95,pix_max\n";
            for (std::size_t i = 0; i < frozen_eval_kf_seq_.size(); ++i)
            {
                const std::size_t idx = frozen_eval_kf_seq_[i];
                if (idx >= views.size() || !views[idx]) continue;
                auto pkf = views[idx];
                const LoopScopeKf* hit = nullptr;
                for (const auto& e : loop_scope_kfs_)
                    if (e.kf_id == pkf->fid_ && e.scene_present)
                    { hit = &e; break; }
                if (!hit) continue;
                std::vector<Eigen::Vector3f> after(xyz_pre.size());
                for (std::size_t j = 0; j < xyz_pre.size(); ++j)
                    after[j] = hit->delta * xyz_pre[j];
                auto s = computePixelDisp(xyz_pre, after, {}, *pkf);
                vf << i << ',' << pkf->fid_ << ',' << s.n_visible << ','
                   << s.mean << ',' << s.p95 << ',' << s.max << '\n';
            }
        }
    }

    restoreRecoverySnapshot(common);
    frozen_fixed_count_ = saved_fixed;
    frozen_kf_replay_ = saved_replay;
    frozen_kf_step_ = saved_kf_step;
    frozen_eval_kf_seq_ = saved_eval_seq;
    frozen_kf_seq_ = saved_kf_seq;
    frozen_train_views_flat_ = saved_train_views;
    opt_params_.weight_line_coherence_ = saved_w0;
    opt_params_.weight_line_shape_ecc_ = saved_w1;
    opt_params_.weight_line_shape_ori_ = saved_w2;
    opt_params_.opacity_reset_interval_ = saved_opacity;
    result_dir_ = saved_result_dir;
    std::cerr << "[GbaDiag] done; root=" << root.string() << std::endl;
}

void GaussianMapperLine::runGbaRecoveryExperiment(
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
    unsigned long evt)
{
    const char* dir_env = std::getenv("PHOTO_SLAM_GBA_RECOVERY");
    if (!dir_env || dir_env[0] == '\0' || gba_recovery_done_ || !gaussians_)
        return;
    gba_recovery_done_ = true;
    const std::filesystem::path root(dir_env);
    std::filesystem::create_directories(root);

    int steps = 500;
    if (const char* e = std::getenv("PHOTO_SLAM_GBA_RECOVERY_STEPS"))
    {
        const int v = std::atoi(e);
        if (v > 0) steps = v;
    }
    std::vector<long long> eval_steps;
    {
        std::string spec = "0,10,50,100,250,500";
        if (const char* e = std::getenv("PHOTO_SLAM_GBA_RECOVERY_EVAL_STEPS"))
            spec = e;
        std::string item;
        for (std::size_t i = 0; i <= spec.size(); ++i)
        {
            if (i < spec.size() && spec[i] != ',') { item += spec[i]; continue; }
            if (item.find_first_not_of(" \t") != std::string::npos)
            {
                const long long v = std::atoll(item.c_str());
                if (v >= 0 && v <= steps) eval_steps.push_back(v);
            }
            item.clear();
        }
        std::sort(eval_steps.begin(), eval_steps.end());
        eval_steps.erase(std::unique(eval_steps.begin(), eval_steps.end()),
                         eval_steps.end());
        if (eval_steps.empty() || eval_steps.front() != 0)
            eval_steps.insert(eval_steps.begin(), 0);
        if (eval_steps.back() != steps) eval_steps.push_back(steps);
    }

    int want_views = 24;
    if (const char* e = std::getenv("PHOTO_SLAM_GBA_RECOVERY_VIEWS"))
        want_views = std::atoi(e);

    // Save production switch state.
    const auto saved_result_dir = result_dir_;
    const bool saved_fixed = frozen_fixed_count_;
    const bool saved_replay = frozen_kf_replay_;
    const std::size_t saved_kf_step = frozen_kf_step_;
    const auto saved_eval_seq = frozen_eval_kf_seq_;
    const auto saved_kf_seq = frozen_kf_seq_;
    const auto saved_train_views = frozen_train_views_flat_;
    const float saved_w0 = opt_params_.weight_line_coherence_;
    const float saved_w1 = opt_params_.weight_line_shape_ecc_;
    const float saved_w2 = opt_params_.weight_line_shape_ori_;
    const int saved_opacity = opt_params_.opacity_reset_interval_;

    // Fixed, shared evaluation views (never trained on).
    frozen_eval_kf_seq_.clear();
    {
        const std::size_t nkf = scene_->keyframes().size();
        if (want_views > 0 && nkf > 0)
        {
            const std::size_t k = std::min<std::size_t>(
                static_cast<std::size_t>(want_views), nkf);
            for (std::size_t i = 0; i < k; ++i)
                frozen_eval_kf_seq_.push_back((i * nkf) / k);
        }
    }

    // Common snapshot BEFORE the shared sampling sequence is generated, so the
    // production scheduler state can be restored exactly afterwards.
    RecoverySnapshot common = captureRecoverySnapshot();
    preGenerateFrozenKfSequence(steps);
    RecoverySnapshot usage;
    for (auto& kv : scene_->keyframes())
    {
        auto& pkf = kv.second;
        if (!pkf) continue;
        RecoveryKfState k;
        k.remaining_times_of_use = pkf->remaining_times_of_use_;
        k.pyramid_times_of_use = pkf->gaus_pyramid_times_of_use_;
        k.tracking_error = pkf->tracking_error_;
        k.done_inactive_densify = pkf->done_inactive_geo_densify_;
        usage.kfs[kv.first] = k;
    }

    const long long n_all = gaussians_->xyz_.size(0);
    const long long n_line = (gaussians_->is_line_.defined() &&
                              gaussians_->is_line_.numel() > 0)
        ? gaussians_->is_line_.to(torch::kCPU).sum().item<int64_t>() : 0;
    const std::string param_fp0 = gaussians_->paramFingerprint();
    const std::string adam_fp0 = gaussians_->adamFingerprint();
    {
        std::ofstream st(root / "common_state.txt");
        st << "head=" << "gba-recovery-v1\n"
           << "event_evt=" << evt << "\n"
           << "gaussians=" << n_all << "\n"
           << "line_gaussians=" << n_line << "\n"
           << "kfs=" << scene_->keyframes().size() << "\n"
           << "iteration=" << iteration_ << "\n"
           << "frozen_opt_steps=" << frozen_opt_steps_ << "\n"
           << "active_sh=" << gaussians_->active_sh_degree_ << "\n"
           << "default_sh=" << default_sh_ << "\n"
           << "param_fp=" << param_fp0 << "\n"
           << "adam_fp=" << adam_fp0 << "\n"
           << "steps=" << steps << "\n"
           << "eval_views=" << frozen_eval_kf_seq_.size() << "\n"
           << "raster_backend_env="
           << (std::getenv("PHOTO_SLAM_RASTER_BACKEND")
                   ? std::getenv("PHOTO_SLAM_RASTER_BACKEND") : "unset")
           << "\n";
    }
    {
        std::ofstream ev(root / ("event_evt" + std::to_string(evt) + ".txt"));
        ev << "# immutable GBA event: kf_id, before/after/delta 4x4 row-major\n";
        for (std::size_t i = 0; i < after_poses.size(); ++i)
        {
            const unsigned long kfid = after_poses[i].first;
            const Sophus::SE3f& after = after_poses[i].second;
            const Sophus::SE3f before = after * world_deltas[i].second;
            ev << kfid << '\n';
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    ev << before.matrix()(r, c) << (c == 3 ? '\n' : ' ');
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    ev << after.matrix()(r, c) << (c == 3 ? '\n' : ' ');
            for (int r = 0; r < 4; ++r)
                for (int c = 0; c < 4; ++c)
                    ev << world_deltas[i].second.matrix()(r, c)
                       << (c == 3 ? '\n' : ' ');
        }
    }
    {
        std::ofstream ef(root / "eval_views.txt");
        ef << "# fixed evaluation keyframes (map order index)\n";
        for (auto v : frozen_eval_kf_seq_) ef << v << '\n';
        std::ofstream sf(root / "kf_sequence.txt");
        sf << "# shared training sequence\n";
        for (auto v : frozen_kf_seq_) sf << v << '\n';
    }

    // Experiment switches: fixed count, no structural loss, no opacity reset.
    frozen_fixed_count_ = true;
    frozen_kf_replay_ = true;
    opt_params_.weight_line_coherence_ = 0.0f;
    opt_params_.weight_line_shape_ecc_ = 0.0f;
    opt_params_.weight_line_shape_ori_ = 0.0f;
    opt_params_.opacity_reset_interval_ = 0;
    SetRasterBackend(2);   // C

    const int check_steps = std::min(10, steps);
    std::vector<long long> check_eval = {0, check_steps};
    std::cerr << "[GbaRecovery] repeat check: " << check_steps << " steps x2"
              << std::endl;
    auto repA = runRecoveryArm("RepeatA", common, usage, after_poses,
                               world_deltas, evt, false, check_steps, check_eval,
                               root / "RepeatA");
    auto endA = captureRecoverySnapshot();
    auto repB = runRecoveryArm("RepeatB", common, usage, after_poses,
                               world_deltas, evt, false, check_steps, check_eval,
                               root / "RepeatB");
    auto endB = captureRecoverySnapshot();

    auto maxDiff = [](const torch::Tensor& a, const torch::Tensor& b) -> double {
        if (!a.defined() || !b.defined() || a.sizes() != b.sizes()) return -1.0;
        return (a.to(torch::kCPU).to(torch::kFloat) -
                b.to(torch::kCPU).to(torch::kFloat)).abs().max().item<double>();
    };
    const double d_xyz = maxDiff(endA.xyz, endB.xyz);
    const double d_rot = maxDiff(endA.rotation, endB.rotation);
    const double d_scale = maxDiff(endA.scaling, endB.scaling);
    const double d_op = maxDiff(endA.opacity, endB.opacity);
    const double d_dc = maxDiff(endA.features_dc, endB.features_dc);
    const double d_rest = maxDiff(endA.features_rest, endB.features_rest);
    double max_param_diff = std::max({d_xyz, d_rot, d_scale, d_op, d_dc, d_rest});
    std::cerr << "[GbaRecovery] repeat tensor diffs: xyz=" << d_xyz
              << " rot=" << d_rot << " scale=" << d_scale
              << " opacity=" << d_op << " dc=" << d_dc
              << " rest=" << d_rest << std::endl;
    double max_adam_diff = 0.0;
    for (std::size_t i = 0; i < endA.adam.size() && i < endB.adam.size(); ++i)
    {
        max_adam_diff = std::max(max_adam_diff,
                                 maxDiff(endA.adam[i].exp_avg, endB.adam[i].exp_avg));
        max_adam_diff = std::max(max_adam_diff,
                                 maxDiff(endA.adam[i].exp_avg_sq, endB.adam[i].exp_avg_sq));
    }
    auto qualityClose = [](const std::map<long long, std::pair<double, double>>& a,
                           const std::map<long long, std::pair<double, double>>& b) {
        if (a.size() != b.size()) return false;
        for (const auto& kv : a) {
            auto it = b.find(kv.first);
            if (it == b.end()) return false;
            if (std::abs(kv.second.first - it->second.first) > 1e-4) return false;
            if (std::abs(kv.second.second - it->second.second) > 1e-5) return false;
        }
        return true;
    };
    // Restore fidelity is proven at the START of the arm: identical parameter
    // and Adam fingerprints, identical step-0 render, identical schedule and
    // training-view sequence. Post-step tensor differences are reported but are
    // not a restore failure: CUDA rasteriser atomics do not guarantee a
    // bit-identical backward reduction order.
    const bool step0_ok =
        repA.param_fp0 == repB.param_fp0 &&
        repA.adam_fp0 == repB.adam_fp0 &&
        repA.schedule_fp == repB.schedule_fp &&
        qualityClose(repA.eval_quality, repB.eval_quality);
    const bool repeat_match = repA.ok && repB.ok && step0_ok;
    {
        std::ofstream rf(root / "repeat_check.csv");
        rf << "check,param_fp0,adam_fp0,schedule_fp,param_fp_end,adam_fp_end,"
              "psnr0,ssim0,psnrN,ssimN,steps,ok\n";
        auto row = [&](const char* n, const RecoveryArmResult& x) {
            const auto q0 = x.eval_quality.count(0) ? x.eval_quality.at(0)
                                                    : std::make_pair(-1.0, -1.0);
            const auto qn = x.eval_quality.count(check_steps)
                ? x.eval_quality.at(check_steps) : std::make_pair(-1.0, -1.0);
            rf << n << ',' << x.param_fp0 << ',' << x.adam_fp0 << ','
               << x.schedule_fp << ',' << x.param_fp << ',' << x.adam_fp
               << ',' << q0.first << ',' << q0.second << ',' << qn.first
               << ',' << qn.second << ',' << x.steps << ',' << (x.ok ? 1 : 0)
               << '\n';
        };
        row("A", repA);
        row("B", repB);
        rf << "step0_param_exact," << (repA.param_fp0 == repB.param_fp0 ? 1 : 0)
           << '\n';
        rf << "step0_adam_exact," << (repA.adam_fp0 == repB.adam_fp0 ? 1 : 0)
           << '\n';
        rf << "schedule_exact," << (repA.schedule_fp == repB.schedule_fp ? 1 : 0)
           << '\n';
        for (const auto& kv : repA.step_param_fp)
        {
            auto it = repB.step_param_fp.find(kv.first);
            rf << "step" << kv.first << "_param_exact,"
               << (it != repB.step_param_fp.end() && it->second == kv.second
                       ? 1 : 0)
               << '\n';
        }
        rf << "quality_tol," << (qualityClose(repA.eval_quality, repB.eval_quality) ? 1 : 0)
           << '\n';
        rf << "post_step_param_max_abs_diff," << max_param_diff << '\n';
        rf << "post_step_adam_max_abs_diff," << max_adam_diff << '\n';
        rf << "match," << (repeat_match ? 1 : 0) << '\n';
    }
    std::cerr << "[GbaRecovery] repeat check "
              << (repeat_match ? "PASS" : "FAIL")
              << " step0_param=" << (repA.param_fp0 == repB.param_fp0 ? 1 : 0)
              << " step0_adam=" << (repA.adam_fp0 == repB.adam_fp0 ? 1 : 0)
              << " schedule=" << (repA.schedule_fp == repB.schedule_fp ? 1 : 0)
              << " quality=" << (qualityClose(repA.eval_quality, repB.eval_quality) ? 1 : 0)
              << " post_step_param_diff=" << max_param_diff
              << " post_step_adam_diff=" << max_adam_diff << std::endl;

    if (!repeat_match)
    {
        std::cerr << "[GbaRecovery] repeat mismatch -> full arms skipped"
                  << std::endl;
    }
    else
    {
        auto po1 = runRecoveryArm("PoseOnly", common, usage, after_poses,
                                  world_deltas, evt, false, steps, eval_steps,
                                  root / "PoseOnly");
        auto pm1 = runRecoveryArm("PoseAndMap", common, usage, after_poses,
                                  world_deltas, evt, true, steps, eval_steps,
                                  root / "PoseAndMap");
        auto pm2 = runRecoveryArm("PoseAndMap_r2", common, usage, after_poses,
                                  world_deltas, evt, true, steps, eval_steps,
                                  root / "PoseAndMap_r2");
        auto po2 = runRecoveryArm("PoseOnly_r2", common, usage, after_poses,
                                  world_deltas, evt, false, steps, eval_steps,
                                  root / "PoseOnly_r2");

        std::ofstream sum(root / "summary.csv");
        sum << "arm,order,evt,steps,updated_rows,max_translation,"
               "mean_translation,sync_seconds,train_seconds,total_seconds,"
               "gaussians,line_gaussians,param_fp,adam_fp,"
               "step,psnr,ssim\n";
        auto write_arm = [&](const char* arm, int order,
                             const RecoveryArmResult& x) {
            for (const auto& q : x.eval_quality)
                sum << arm << ',' << order << ',' << evt << ',' << x.steps << ','
                    << x.updated_rows << ',' << x.max_translation << ','
                    << x.mean_translation << ',' << x.sync_seconds << ','
                    << x.train_seconds << ','
                    << (x.sync_seconds + x.train_seconds) << ',' << x.gaussians
                    << ',' << x.line_gaussians << ',' << x.param_fp << ','
                    << x.adam_fp << ',' << q.first << ',' << q.second.first
                    << ',' << q.second.second << '\n';
        };
        write_arm("PoseOnly", 1, po1);
        write_arm("PoseAndMap", 1, pm1);
        write_arm("PoseAndMap_r2", 2, pm2);
        write_arm("PoseOnly_r2", 2, po2);
        sum.flush();
        std::cerr << "[GbaRecovery] arms finished; summary="
                  << (root / "summary.csv").string() << std::endl;
    }

    // Restore the production state (pre-event, pre-sampling change).
    restoreRecoverySnapshot(common);
    frozen_fixed_count_ = saved_fixed;
    frozen_kf_replay_ = saved_replay;
    frozen_kf_step_ = saved_kf_step;
    frozen_eval_kf_seq_ = saved_eval_seq;
    frozen_kf_seq_ = saved_kf_seq;
    frozen_train_views_flat_ = saved_train_views;
    opt_params_.weight_line_coherence_ = saved_w0;
    opt_params_.weight_line_shape_ecc_ = saved_w1;
    opt_params_.weight_line_shape_ori_ = saved_w2;
    opt_params_.opacity_reset_interval_ = saved_opacity;
    result_dir_ = saved_result_dir;
}

void GaussianMapperLine::combineMappingOperations()
{
    // Get Mapping Operations
    // geom-sync-v1: one-shot CPU contract self-test (diagnostic only).
    {
        static const bool contract_test =
            (std::getenv("PHOTO_SLAM_SYNC_CONTRACT_TEST") != nullptr);
        static bool contract_done = false;
        if (contract_test && !contract_done) {
            contract_done = true;
            runRigidCorrectionContractTest();
        }
    }
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // std::cerr << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;

            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();

            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframeLine> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.   //not update the (world points) -> new camera pose not aligning the world points(????)
                // Otherwise create a new one
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();

                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                    // TODO: Update the world Line associated with this keyframe

                    //TODO: Update the world points associated with this keyframe
                }
                else {
                    handleNewKeyframe(kf);
                }
            }

            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cerr << "[Gaussian Mapper]Loop Closure Detected."
                      << std::endl;

            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
            // gba-recovery diagnosis: capture this LOOP event's per-keyframe
            // world delta once, before the legacy visible-point path mutates
            // the scene poses.
            if (std::getenv("PHOTO_SLAM_GBA_RECOVERY_DIAG") && !loop_scope_available_) {
                loop_scope_kfs_.clear();
                for (auto& kf : associated_kfs) {
                    LoopScopeKf e;
                    e.kf_id = std::get<0>(kf);
                    auto pkf = scene_->getKeyframe(e.kf_id);
                    e.scene_present = static_cast<bool>(pkf);
                    if (pkf) {
                        Sophus::SE3f oldp = pkf->getPosef();
                        Sophus::SE3f newp = std::get<2>(kf);
                        e.delta = newp.inverse() * oldp;
                        e.rot_deg = e.delta.so3().log().norm() * 180.0 / M_PI;
                        e.trans = e.delta.translation().norm();
                    }
                    loop_scope_kfs_.push_back(e);
                }
                loop_scope_available_ = !loop_scope_kfs_.empty();
                std::cerr << "[GbaDiag] LOOP scope captured kfs="
                          << loop_scope_kfs_.size() << std::endl;
            }
            // Mark the transformed points to avoid transforming more than once
            torch::Tensor point_not_transformed_flags =
                torch::full(
                    {gaussians_->xyz_.size(0)},
                    true,
                    torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
            int num_transformed = 0;
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframeLine> pkf = scene_->getKeyframe(kfid);
                // In case new points are added in handleNewKeyframe()
                int64_t num_new_points = gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
                if (num_new_points > 0)
                    point_not_transformed_flags = torch::cat({
                        point_not_transformed_flags,
                        torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                        /*dim=*/0);
                // If kf is already in the scene, evaluate the change in pose,
                // if too large we perform loop correction on its visible model points.
                // If not in the scene, create a new one.
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    // If is loop closure kf
// if (std::get<4>(kf)) {
// renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                        Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                        Sophus::SE3f inv_pose = pose.inverse();
                        Sophus::SE3f diff_pose = inv_pose * original_pose;
                        bool large_rot = !diff_pose.rotationMatrix().isApprox(
                            Eigen::Matrix3f::Identity(), large_rot_th_);
                        bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                            1.0, large_trans_th_);
                        if (large_rot || large_trans) {
                            std::cerr << "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
                                    << kfid << std::endl;
                            diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                            diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                            diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                            torch::Tensor diff_pose_tensor =
                                tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(), device_type_).transpose(0, 1);
                            {
                                std::unique_lock<std::mutex> lock_render(mutex_render_);
                                gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                    point_not_transformed_flags,
                                    diff_pose_tensor,
                                    pkf->world_view_transform_,
                                    pkf->full_proj_transform_,
                                    pkf->creation_iter_,
                                    stableNumIterExistence(),
                                    num_transformed,
                                    loop_kf_scale); // selected xyz *= s
                            }
                            // Give loop keyframes times of use
                            increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
// renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
// std::cerr<<num_transformed<<std::endl;
                        }
// }
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();
// if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");
                }
                else {
                    handleNewKeyframe(kf);
                }
            }
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
// keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));

            // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }

            // Mark this iteration
            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::GlobalBA:
        {
            // The point-only mapper never owns line/ref-keyframe metadata, so
            // it must not attempt a deformation. Log and reject.
            printf("[GBA-Apply] evt=%lu map=%lu REJECTED "
                   "reason=point_only_mapper\n", opr.eventId(), opr.mapGen());
            fflush(stdout);
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cerr << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
                      << std::endl;

            float s = opr.mfScale;
            Sophus::SE3f& T = opr.mT;
            if (initial_mapped_) {
                // Apply the scaled transformation on gaussian model points
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    gaussians_->applyScaledTransformation(s, T);
                }
                // Apply the scaled transformation to the scene
                scene_->applyScaledTransformation(s, T);
            }
            else { // TODO: the workflow should not come here, delete this branch
                // Apply the scaled transformation to the cached points
                for (auto& pt : scene_->cached_point_cloud_) {
                    // pt <- (s * Ryw * pt + tyw)
                    auto& pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast<double>() * pt_xyz;
                }

                // Apply the scaled transformation on gaussian keyframes
                for (auto& kfit : scene_->keyframes()) {
                    std::shared_ptr<GaussianKeyframeLine> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error("MappingOperation type not supported!");
        }
        break;
        }
    }
}

void GaussianMapperLine::combineMappingOperations_withLine()
{
    // Get Mapping Operations
    // geom-sync-v1: one-shot CPU contract self-test (diagnostic only).
    {
        static const bool contract_test =
            (std::getenv("PHOTO_SLAM_SYNC_CONTRACT_TEST") != nullptr);
        static bool contract_done = false;
        if (contract_test && !contract_done) {
            contract_done = true;
            runRigidCorrectionContractTest();
        }
    }
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // =========================================================
            // 1. Handle Keyframes (Update Pose or Add New)
            // =========================================================
            auto& associated_kfs = opr.associatedKeyFrames();

            for (auto& kf : associated_kfs) {
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframeLine> pkf = scene_->getKeyframe(kfid);
                
                if (pkf) {
                    // Update existing keyframe pose
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();

                    // Increase usage count for optimization window
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                }
                else {
                    // Create new keyframe
                    handleNewKeyframe_WithLine(kf);
                }
            }

            // =========================================================
            // 2. Handle New Map Points (Standard Point Cloud)
            // =========================================================
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                // Deformation anchors, parallel to the position triples. A size
                // mismatch is reported as unknown (-1) rather than guessed.
                const int64_t n_pts = static_cast<int64_t>(points.size() / 3);
                const auto& ref_ids = opr.associatedMapPointRefKfIds();
                torch::Tensor ref_tensor;
                if (static_cast<int64_t>(ref_ids.size()) == n_pts) {
                    ref_tensor = torch::from_blob(
                        reinterpret_cast<std::int64_t*>(
                            const_cast<long long*>(ref_ids.data())), {n_pts},
                        torch::TensorOptions().dtype(torch::kLong))
                        .clone().to(device_type_);
                }
                const int64_t n_before = gaussians_->xyz_.size(0);
                // 这里的 increasePcd 会把 is_line 设为 false
                gaussians_->increasePcd(points, colors, getIteration(), ref_tensor);
                lm_created_rows_total_ += gaussians_->xyz_.size(0) - n_before;
            }

            // =========================================================
            // 3. Handle New Map Lines (Line-Aware Gaussian Seeding)
            // =========================================================
            // 从 Operation 中提取纯数据 (Pos, Color, Direction)
            // 这些数据是在 LocalMapping 线程中通过 pML->SamplePoints... 提前计算好的
            auto& line_sampled_data = opr.associatedLineSampledPoints();
            const std::vector<float>& l_positions = std::get<0>(line_sampled_data);
            const std::vector<float>& l_colors    = std::get<1>(line_sampled_data);
            const std::vector<float>& l_dirs      = std::get<2>(line_sampled_data);

            size_t num_l_points = l_positions.size() / 3;

            // [DEBUG PROBE: Check Source Data]
            if (!l_positions.empty()) {
                if (ORB_SLAM3::IsLineLossDiag())
                    std::cerr << "[LINE-SAMPLE] op_line_points=" << num_l_points
                              << " pos_floats=" << l_positions.size() << std::endl;
                std::cerr << "[DEBUG Mapper] Received Line Sampled Points from SLAM:" << std::endl;
                std::cerr << "  > Pos Size: " << l_positions.size() << " (N=" << num_l_points << ")" << std::endl;
                std::cerr << "  > Dir Size: " << l_dirs.size() << std::endl;
                
                if (l_positions.size() % 3 != 0) {
                     std::cerr << "\033[1;31m[CRITICAL] Position vector size is NOT multiple of 3!\033[0m" << std::endl;
                }
                if (l_dirs.size() != l_positions.size()) {
                     std::cerr << "\033[1;31m[CRITICAL] Direction vector size mismatch!\033[0m" << std::endl;
                }
            }

            // 只有当有有效数据时才处理
            if (initial_mapped_ && num_l_points > 0) {
                // 安全检查
                if (l_positions.size() != l_colors.size() || l_positions.size() != l_dirs.size()) {
                    std::cerr << "[GaussianMapper] Error: Line sampled data size mismatch! " 
                              << "Pos: " << l_positions.size() 
                              << ", Col: " << l_colors.size() 
                              << ", Dir: " << l_dirs.size() << std::endl;
                } 
                else {
                    std::vector<Point3D> new_line_sample_points;
                    new_line_sample_points.reserve(num_l_points);

                    // 设定网格大小，建议与采样步长一致或略大
                    //const float v_size = 0.25f; //added by zdg
                    int raw_points = l_positions.size() / 3;
                    int filtered_points = 0;

                    for (size_t i = 0; i < num_l_points; ++i) {

                        Eigen::Vector3f p_pos(l_positions[i*3], l_positions[i*3+1], l_positions[i*3+2]);
                
                        // 🌟 核心去重判断：O(1) 复杂度
                        if (scene_->isVoxelOccupied(p_pos, opt_params_.voxel_size_)) {
                            filtered_points++;
                            continue; // 空间上已经存在线段高斯，跳过
                        }

                        Point3D p;
                        // Position
                        p.xyz_(0) = l_positions[i * 3 + 0];
                        p.xyz_(1) = l_positions[i * 3 + 1];
                        p.xyz_(2) = l_positions[i * 3 + 2];
                        
                        // Color (Assume 0-1 normalized in Atlas, or re-normalize if needed)
                        // 注意：如果 Atlas 里存的是 0-1 float，直接赋值；如果是 0-255，需除以 255.0
                        // 根据 Atlas.h 代码：l_colors 已经是 /255.0f 后的 float，所以直接赋值
                        p.color_(0) = l_colors[i * 3 + 0];
                        p.color_(1) = l_colors[i * 3 + 1];
                        p.color_(2) = l_colors[i * 3 + 2];

                        // Line Attributes
                        p.source_ = PointSourceType::LINE_SAMPLED;
                        p.line_dir_(0) = l_dirs[i * 3 + 0];
                        p.line_dir_(1) = l_dirs[i * 3 + 1];
                        p.line_dir_(2) = l_dirs[i * 3 + 2];
                        
                        // Sample Step (可以使用默认值，或者如果在 Atlas 中存了也可以读取)
                        //p.sample_step_ = 0.1f; // 需与 LocalMapping 中的采样步长一致
                        p.sample_step_ = opt_params_.line_sample_step_;
                        // Deformation anchor of the source MapLine's reference KF.
                        const auto& line_ref_ids =
                            opr.associatedLineSampledPointRefKfIds();
                        p.ref_kf_id_ = (i < line_ref_ids.size())
                            ? line_ref_ids[i] : -1LL;

                        new_line_sample_points.push_back(p);

                        // 记录到场景缓存和 VoxelGrid 中
                        scene_->addPointToVoxel(p_pos, opt_params_.voxel_size_);
                        
                        // (可选) 如果你想在 Scene 中也缓存这些点用于 Debug/SavePly
                        // scene_->cacheLineSampledPnts3D(0 /*dummy_line_id*/, p);
                    }

                    // 打印日志
                    std::cerr << "[Voxel Grid Report]: ID"
                            << " | Incoming: " << raw_points
                            << " | Accepted: " << filtered_points
                            << " | Rejected (Redundant): " << (raw_points - filtered_points)
                            << " | Usage Rate: " << (float)filtered_points/raw_points * 100 << "%" << std::endl;

                    // Push to GPU
                    if (!new_line_sample_points.empty()) {
                        lineSampleAuditLog(raw_points, num_l_points,
                                           filtered_points,
                                           static_cast<long long>(new_line_sample_points.size()));
                        torch::NoGradGuard no_grad;
                        std::unique_lock<std::mutex> lock_render(mutex_render_);
                        
                        // 调用重载版本的 increasePcd，它会处理 line_dir 和 anisotropic scale
                        gaussians_->auditSetReason("line_seed");
                        gaussians_->increasePcd(new_line_sample_points, getIteration());
                        gaussians_->auditSetReason("map_point_seed");
                        
                        std::cerr << "[GaussianMapper] Added " << new_line_sample_points.size() 
                                   << " line-sampled gaussians." << std::endl;
                    }
                }
            }
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cerr << "[Gaussian Mapper]Loop Closure Detected." << std::endl;

            float loop_kf_scale = opr.mfScale;
            auto& associated_kfs = opr.associatedKeyFrames();

            // Mark points to avoid double transform
            torch::Tensor point_not_transformed_flags = torch::full(
                    {gaussians_->xyz_.size(0)}, true,
                    torch::TensorOptions().device(device_type_).dtype(torch::kBool));

            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
            
            int num_transformed = 0;
            float sumTrans = 0.0f, sumRot = 0.0f, sumScale = 0.0f;
            int nLargeCorr = 0;
            float maxTrans = 0.0f, maxRotDeg = 0.0f;

            // Explicit loop-correction magnitude gate (cost-only, does NOT
            // affect loop validity): correct Gaussians of a KF only when
            //   transNorm > TRANS_THRESHOLD_METERS ||
            //   rotationAngleRad > ROT_THRESHOLD_RAD
            // Defaults 0.10 m and 2 deg; env-configurable.
            const char* et = std::getenv("PHOTO_SLAM_GAUSS_LOOP_TRANS_M");
            const float gaussLoopTransThrM = (et ? std::atof(et) : 0.10f);
            const char* er = std::getenv("PHOTO_SLAM_GAUSS_LOOP_ROT_DEG");
            const float gaussLoopRotThrRad = (er ? std::atof(er) : 2.0f) * (float)M_PI / 180.0f;
            int nKFsTotal = 0, nKFsWithPkf = 0, nKFsCorrected = 0, nKFsSkippedByGate = 0;

            for (auto& kf : associated_kfs) {
                nKFsTotal++;
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframeLine> pkf = scene_->getKeyframe(kfid);
                
                // Resize mask if new points added
                int64_t num_new_points = gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
                if (num_new_points > 0)
                    point_not_transformed_flags = torch::cat({
                        point_not_transformed_flags,
                        torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                        0);

                if (pkf) {
                    nKFsWithPkf++;
                    auto& pose = std::get<2>(kf);
                    
                    // Logic to detect large loop drift and correct geometry
                    Sophus::SE3f original_pose = pkf->getPosef();
                    Sophus::SE3f inv_pose = pose.inverse();
                    Sophus::SE3f diff_pose = inv_pose * original_pose;

                    maxTrans = std::max(maxTrans, diff_pose.translation().norm());
                    maxRotDeg = std::max(maxRotDeg,
                        (float)(Eigen::AngleAxisd(diff_pose.rotationMatrix().cast<double>()).angle() * 180.0 / M_PI));
                    
                    const float transNorm = diff_pose.translation().norm();
                    const float rotRad = (float)Eigen::AngleAxisd(diff_pose.rotationMatrix().cast<double>()).angle();
                    if (std::getenv("PHOTO_SLAM_GBA_RECOVERY_DIAG") && !loop_scope_available_) {
                        LoopScopeKf e;
                        e.kf_id = kfid;
                        e.rot_deg = rotRad * 180.0 / M_PI;
                        e.trans = transNorm;
                        e.scene_present = true;
                        e.delta = diff_pose;
                        loop_scope_kfs_.push_back(e);
                    }
                    const bool do_correct = (transNorm > gaussLoopTransThrM) || (rotRad > gaussLoopRotThrRad);
                    
                    if (do_correct) {
                        nKFsCorrected++;
                        std::cerr << "[Gaussian Mapper] Large loop correction for KF " << kfid << std::endl;
                        
                        // Calculate transformation matrix M: P_new = M * P_old
                        // Logic derived from: T_new_inv * P_new = T_old_inv * P_old (relative pose const)
                        diff_pose.translation() -= inv_pose.translation();
                        diff_pose.translation() *= loop_kf_scale;
                        diff_pose.translation() += inv_pose.translation();

                        sumTrans += diff_pose.translation().norm();
                        sumRot += Eigen::AngleAxisd(diff_pose.rotationMatrix().cast<double>()).angle();
                        sumScale += loop_kf_scale;
                        nLargeCorr++;
                        
                        torch::Tensor diff_pose_tensor = tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(), device_type_).transpose(0, 1);
                        {
                            std::unique_lock<std::mutex> lock_render(mutex_render_);
                            // Apply correction to Gaussian centroids
                            // Note: For Line Gaussians, rotation should also be updated ideally, 
                            // but since they are attached to xyz, simple rigid transform of xyz is a good approx.
                            // Ideally we should rotate the quaternion too.
                            gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                point_not_transformed_flags,
                                diff_pose_tensor,
                                pkf->world_view_transform_,
                                pkf->full_proj_transform_,
                                pkf->creation_iter_,
                                stableNumIterExistence(),
                                num_transformed,
                                loop_kf_scale);
                        }
                        increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
                    }
                    else {
                        nKFsSkippedByGate++;
                    }
                    
                    // Update KF pose
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
                else {
                    handleNewKeyframe_WithLine(kf);
                }
            }

            if (std::getenv("PHOTO_SLAM_GBA_RECOVERY_DIAG")) {
                loop_scope_available_ = !loop_scope_kfs_.empty();
                std::cerr << "[GbaDiag] LOOP scope captured kfs="
                          << loop_scope_kfs_.size() << std::endl;
            }

            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));

            // Line-Gaussian loop-sync statistics.
            {
                int lineGaussians = 0;
                int nonFinite = 0;
                int correctedLineGaussians = 0;
                if (gaussians_->xyz_.size(0) > 0) {
                    lineGaussians = gaussians_->is_line_.sum().item<int>();
                    nonFinite = gaussians_->xyz_.size(0)
                        - torch::isfinite(gaussians_->xyz_).all(1).sum().item<int>();
                    // Line Gaussians transformed by this loop (were in the
                    // not-transformed set before, now marked transformed).
                    if (lineGaussians > 0 && point_not_transformed_flags.size(0) == gaussians_->is_line_.size(0))
                        correctedLineGaussians = torch::logical_and(
                            gaussians_->is_line_, torch::logical_not(point_not_transformed_flags)).sum().item<int>();
                }
                const float meanT = (nLargeCorr > 0) ? (sumTrans / nLargeCorr) : 0.0f;
                const float meanR = (nLargeCorr > 0) ? (sumRot / nLargeCorr * 180.0 / M_PI) : 0.0f;
                const float meanS = (nLargeCorr > 0) ? (sumScale / nLargeCorr) : 1.0f;
                std::cerr << "[LineLoop-Gaussian] candidates=" << lineGaussians
                          << " correctedLineGaussians=" << correctedLineGaussians
                          << " kfsTotal=" << nKFsTotal
                          << " kfsWithPkf=" << nKFsWithPkf
                          << " kfsCorrected=" << nKFsCorrected
                          << " kfsSkippedByGate=" << nKFsSkippedByGate
                          << " corrected=" << num_transformed
                          << " skippedBad=0 skippedNull=0 duplicateSkipped=0"
                          << " nonFinite=" << nonFinite
                          << " maxTranslation=" << maxTrans
                          << " maxRotationDeg=" << maxRotDeg
                          << " meanTranslation=" << meanT
                          << " meanRotationDeg=" << meanR
                          << " meanScaleRatio=" << meanS << std::endl;
            }

            // Add new map points after loop closure
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }
            
            // Loop Closure 时是否处理新的 Line Samples？
            // 通常 Loop Closure 后的 MappingOperation 不会携带大量新的 MapLines 采样，
            // 除非你在 LoopClosing 线程里也加了采样逻辑。
            // 可以在这里加上类似 LocalMappingBA 的 Line 处理逻辑，如果需要的话。

            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::GlobalBA:
        {
            // geom-sync-v3: APPLY the GBA correction to existing Gaussians.
            // The callback is a pure correction: it must never seed, densify,
            // clone, split or prune. The row count is asserted below.
            const unsigned long evt = opr.eventId();
            const unsigned long prev_evt = gaussians_
                ? gaussians_->lastAppliedGbaEvent() : last_applied_gba_event_;
            const long long count_before =
                gaussians_ ? gaussians_->xyz_.size(0) : 0;

            ORB_SLAM3::Map* pCurrentMap =
                pSLAM_->getAtlas()->GetCurrentMap();
            const unsigned long cur_map_gen =
                pCurrentMap ? static_cast<unsigned long>(pCurrentMap->GetId())
                            : opr.mapGen();

            if (evt == 0 || evt <= prev_evt)
            {
                printf("[GBA-Apply] evt=%lu map=%lu prev_evt=%lu REJECTED "
                       "reason=stale_or_duplicate count=%lld\n",
                       evt, opr.mapGen(), prev_evt, count_before);
                fflush(stdout);
                break;
            }
            if (opr.mapGen() != cur_map_gen)
            {
                printf("[GBA-Apply] evt=%lu map=%lu current_map=%lu REJECTED "
                       "reason=map_generation count=%lld\n",
                       evt, opr.mapGen(), cur_map_gen, count_before);
                fflush(stdout);
                break;
            }
            if (!initial_mapped_ || !gaussians_)
            {
                printf("[GBA-Apply] evt=%lu map=%lu REJECTED "
                       "reason=model_not_initialised\n", evt, opr.mapGen());
                fflush(stdout);
                break;
            }
            if (sensor_type_ != RGBDLINE)
            {
                // Scoped limitation: this path implements the RGB-D SE(3)
                // correction only. Mono Sim(3)/local-BA/newer SH stay on their
                // existing paths.
                printf("[GBA-Apply] evt=%lu map=%lu REJECTED "
                       "reason=se3_rgbd_only sensor=%d\n", evt, opr.mapGen(),
                       static_cast<int>(sensor_type_));
                fflush(stdout);
                break;
            }

            int n_in_scene = 0, n_scene_missing = 0;
            double max_pose_mismatch = 0.0;
            std::vector<std::pair<unsigned long, Sophus::SE3f>> deltas;
            std::vector<std::pair<unsigned long, Sophus::SE3f>> after_poses;
            deltas.reserve(opr.globalBAPoses().size());
            after_poses.reserve(opr.globalBAPoses().size());
            for (const auto& e : opr.globalBAPoses())
            {
                // G maps OLD world -> NEW world for rows anchored at e.kf_id.
                deltas.emplace_back(e.kf_id, e.Tcw_after.inverse() * e.Tcw_before);
                after_poses.emplace_back(e.kf_id, e.Tcw_after);
                std::shared_ptr<GaussianKeyframeLine> pkf =
                    scene_->getKeyframe(e.kf_id);
                if (!pkf)
                {
                    ++n_scene_missing;
                    continue;
                }
                ++n_in_scene;
                // The event's `before` must be the pose the scene still has.
                // A large mismatch is reported; the event snapshot stays
                // authoritative because it is immutable.
                Sophus::SE3f d = pkf->getPosef().inverse() * e.Tcw_before;
                max_pose_mismatch = std::max<double>(
                    max_pose_mismatch,
                    d.translation().norm() + d.so3().log().norm());
            }

            // gba-recovery: env-gated controlled experiment. It captures the
            // pre-event state, runs its arms, then restores that state so the
            // production correction below is unchanged. No-op when the env is
            // unset. PHOTO_SLAM_GBA_RECOVERY_DIAG selects the short mechanism
            // diagnosis instead of the 500-step matrix.
            if (std::getenv("PHOTO_SLAM_GBA_RECOVERY_DIAG"))
                runGbaRecoveryDiagnosis(deltas, after_poses, evt);
            else
                runGbaRecoveryExperiment(deltas, after_poses, evt);

            // Production sync timing starts AFTER the optional experiment, so
            // the reported correction cost is not inflated by the offline arms.
            const auto t_start = std::chrono::steady_clock::now();
            GaussianModelLine::GbaCorrectionStats stats;
            // gba-recovery closed the forced deformation: v1 and the mechanism
            // diagnosis found no stable quality or time gain. The event is still
            // consumed, camera poses are still updated and the applied-event
            // version is still bumped; only the Gaussian coordinates are left
            // alone unless the deformation is explicitly requested.
            const char* apply_env = std::getenv("PHOTO_SLAM_GBA_APPLY_CORRECTION");
            const bool deform_gaussians =
                (apply_env && std::string(apply_env) == "1");
            if (!deform_gaussians)
            {
                std::cerr << "[GBA-Apply] deformation DISABLED (default; set "
                             "PHOTO_SLAM_GBA_APPLY_CORRECTION=1 to enable). "
                             "Camera poses are still updated; "
                          << deltas.size() << " deltas not applied"
                          << std::endl;
            }
            else
            {
                std::cerr << "[GBA-Apply] deformation ENABLED explicitly "
                             "(PHOTO_SLAM_GBA_APPLY_CORRECTION=1)"
                          << std::endl;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                stats = gaussians_->applyGbaCorrections(deltas, evt);
            }

            // Scene keyframes adopt the final GBA poses (training views).
            for (const auto& e : opr.globalBAPoses())
            {
                std::shared_ptr<GaussianKeyframeLine> pkf =
                    scene_->getKeyframe(e.kf_id);
                if (!pkf) continue;
                pkf->setPose(
                    e.Tcw_after.unit_quaternion().cast<double>(),
                    e.Tcw_after.translation().cast<double>());
                pkf->computeTransformTensors();
            }

            const long long count_after = gaussians_->xyz_.size(0);
            const long long created_during_event = count_after - count_before;
            TORCH_CHECK(count_after == count_before,
                        "[GBA-Apply] correction changed the Gaussian count: ",
                        count_before, " -> ", count_after);
            last_applied_gba_event_ = evt;

            // Explain unassigned/missing counts instead of guessing anchors.
            long long ref_in_event = 0, ref_scene_not_event = 0;
            long long ref_absent_scene = 0, ref_unassigned = 0;
            long long ref_newer_than_event = 0;
            unsigned long max_event_kf = 0;
            {
                std::unordered_set<unsigned long> event_ids;
                for (const auto& e : opr.globalBAPoses())
                {
                    event_ids.insert(e.kf_id);
                    max_event_kf = std::max(max_event_kf, e.kf_id);
                }
                torch::Tensor ids = gaussians_->refKfIds();
                if (ids.defined() && ids.numel() > 0)
                {
                    torch::Tensor cpu_ids = ids.to(torch::kCPU).contiguous();
                    auto acc = cpu_ids.accessor<std::int64_t, 1>();
                    for (int64_t i = 0; i < cpu_ids.size(0); ++i)
                    {
                        const long long id = acc[i];
                        if (id < 0) { ++ref_unassigned; continue; }
                        const unsigned long uid = static_cast<unsigned long>(id);
                        if (event_ids.count(uid)) { ++ref_in_event; continue; }
                        auto pkf = scene_->getKeyframe(static_cast<std::size_t>(id));
                        if (pkf) ++ref_scene_not_event;
                        else      ++ref_absent_scene;
                        if (uid > max_event_kf) ++ref_newer_than_event;
                    }
                }
            }
            const double mean_translation = stats.moved_rows > 0
                ? stats.translation_sum / static_cast<double>(stats.moved_rows)
                : 0.0;

            const auto t_end = std::chrono::steady_clock::now();
            const double sync_ms =
                std::chrono::duration<double, std::milli>(t_end - t_start).count();
            printf("[GBA-Apply] evt=%lu map=%lu kfs=%zu in_scene=%d "
                   "deltas=%zu moved=%lld nonzero_moved=%lld line_moved=%lld "
                   "stale=%lld unassigned=%lld missing_event=%lld groups=%lld "
                   "scene_missing=%d pose_mismatch_max=%.4f "
                   "translation_mean=%.5f translation_max=%.5f "
                   "ref_in_event=%lld ref_scene_not_event=%lld "
                   "ref_absent_scene=%lld ref_unassigned=%lld "
                   "ref_newer_than_event=%lld "
                   "count_before=%lld count_after=%lld "
                   "created_during_event=%lld lm_map_points_created=%lld "
                   "sync_ms=%.1f APPLIED\n",
                   evt, opr.mapGen(), opr.globalBAPoses().size(), n_in_scene,
                   deltas.size(), stats.moved_rows, stats.nonzero_moved_rows,
                   stats.line_rows_moved, stats.stale_rows,
                   stats.unassigned_rows, stats.missing_event_rows,
                   stats.groups_applied, n_scene_missing, max_pose_mismatch,
                   mean_translation, stats.max_translation, ref_in_event,
                   ref_scene_not_event, ref_absent_scene, ref_unassigned,
                   ref_newer_than_event, count_before, count_after,
                   created_during_event, lm_created_rows_total_, sync_ms);
            fflush(stdout);
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cerr << "[Gaussian Mapper] Scale refinement Detected." << std::endl;
            float s = opr.mfScale;
            Sophus::SE3f& T = opr.mT;
            
            if (initial_mapped_) {
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    // Applies Sim3 transform to all Gaussians (Pos + Scale)
                    gaussians_->applyScaledTransformation(s, T);
                }
                scene_->applyScaledTransformation(s, T);
            }
            else {
                // Legacy path (should not happen if system initialized correctly)
                for (auto& pt : scene_->cached_point_cloud_) {
                    auto& pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast<double>() * pt_xyz;
                }
                for (auto& kfit : scene_->keyframes()) {
                    std::shared_ptr<GaussianKeyframeLine> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
            throw std::runtime_error("MappingOperation type not supported!");
        }
    }
}

#if 0
void GaussianMapperLine::combineMappingOperations_withLine()
{
    // Get Mapping Operations
    // geom-sync-v1: one-shot CPU contract self-test (diagnostic only).
    {
        static const bool contract_test =
            (std::getenv("PHOTO_SLAM_SYNC_CONTRACT_TEST") != nullptr);
        static bool contract_done = false;
        if (contract_test && !contract_done) {
            contract_done = true;
            runRigidCorrectionContractTest();
        }
    }
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // std::cerr << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;

            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();

            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframeLine> pkf = scene_->getKeyframe(kfid);
                // If the keyframe is already in the scene, only update the pose.   //not update the (world points) -> new camera pose not aligning the world points(????)
                //这样导致很多问题:1. 新的相机位姿与世界点不对齐，导致重投影误差增大,需要在gaussian splatting 的优化中，这些顶点需要重新优化，移动它的位置等等，增加了操作。可以有其它方法来做。找到这些区域的顶点，然后进行局部优化。
                // Otherwise create a new one
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();

                    // Give local BA keyframes times of use
                    increaseKeyframeTimesOfUse(pkf, local_BA_increased_times_of_use_);
                    // TODO: Update the world Line associated with this keyframe

                    //TODO: Update the world points associated with this keyframe
                }
                else {
                    handleNewKeyframe_WithLine(kf);
                }
            }

            // Get new points
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }

            // =========================================================================
            // 3. [新增] Handle New Map Lines (New Logic for Line-Awareness)
            // =========================================================================
            // 假设 opr 有 associatedMapLines() 接口
            auto& associated_lines = opr.associatedMapLines(); 
            auto& map_lines = std::get<0>(associated_lines); // std::vector<MapLine*>
            //auto& map_lines_colors = std::get<1>(associated_lines); // std::vector<std::pair<cv::Vec3b, cv::Vec3b>>
            auto& map_sampled_pnts = opr.associatedLineSampledPoints(); // std::vector<Point3D> 

            if (initial_mapped_ && !map_lines.empty()) {
                std::vector<Point3D> new_line_sample_points;
                new_line_sample_points.reserve(map_lines.size() * 10); // 预估容量

                // 采样参数 (建议后续移入 readConfigFromFile)
                float sample_step = 0.1f;
                float view_angle_power = 2.0f;
                float sigma_line_pixel = 3.0f;
                int top_k = 3;
                
                // A. 直接使用 ORB-SLAM3 传递过来的线采样点
                

                //接口有点问题，待解决
                // for (auto* pML : map_lines) {
                //     if (!pML || pML->isBad()) continue;
                //     // A. 获取线段几何信息 & 计算方向
                //     auto endpoints = pML->GetLineWorldPos();
                //     Eigen::Vector3f p1(endpoints.first[0], endpoints.first[1], endpoints.first[2]);
                //     Eigen::Vector3f p2(endpoints.second[0], endpoints.second[1], endpoints.second[2]);
                //     Eigen::Vector3f line_vec = p2 - p1;
                //     float len = line_vec.norm();
                //     if (len < 1e-4) continue; // 忽略过短的线
                //     Eigen::Vector3f line_dir = line_vec.normalized();
                //     // B. 缓存 Line3D 对象到 Scene (用于调试/可视化)
                //     // 如果 Scene 中已经有了，是否更新？通常 LocalBA 会优化线段位置，所以建议更新。
                //     Line3D line3d_obj;
                //     line3d_obj.p1_ = p1;
                //     line3d_obj.p2_ = p2;
                //     auto l_colors = pML->GetLineColorRGB();
                //     line3d_obj.color1_ = Eigen::Vector3f(l_colors.first[0], l_colors.first[1], l_colors.first[2]);
                //     line3d_obj.color2_ = Eigen::Vector3f(l_colors.second[0], l_colors.second[1], l_colors.second[2]);
                //     scene_->cacheLine3D(pML->mnId, line3d_obj);
                //     // C. 在新位置进行采样
                //     // 注意：如果是老线段位置更新，旧的高斯球不会自动移动，
                //     // 我们这里添加的是基于新位置的“新高斯球”。旧的会被后续 Pruning 机制干掉。
                //     pML->SamplePointsAlongLine_MultiViewWeighted_Advanced(
                //         sample_step, view_angle_power, sigma_line_pixel, top_k);                   
                //     const std::vector<Eigen::Vector3f>& sampledPoints3D = pML->GetLineSampledPoints3D();
                //     const std::vector<cv::Vec3b>& sampledColors = pML->GetLineSampledPntsColors();
                //     // D. 构建 Point3D 对象
                //     for (size_t j = 0; j < sampledPoints3D.size(); ++j) {
                //         Point3D point3D;
                //         point3D.xyz_ = sampledPoints3D[j];                        
                //         // 归一化颜色
                //         point3D.color_(0) = (float)(sampledColors[j][0] / 255.0f);
                //         point3D.color_(1) = (float)(sampledColors[j][1] / 255.0f);
                //         point3D.color_(2) = (float)(sampledColors[j][2] / 255.0f);
                //         // 【关键】设置线特征属性
                //         point3D.source_ = PointSourceType::LINE_SAMPLED;
                //         point3D.line_dir_ = line_dir;       // 用于初始化各向异性旋转
                //         point3D.sample_step_ = sample_step; // 用于初始化各向异性 Scaling                       
                //         // 可选：设置参考深度等，如果没有可设为默认值
                //         // point3D.ref_depth_z_ = ...; 
                //         new_line_sample_points.push_back(point3D);
                //         // 同时也缓存到 Scene 中，保持一致性
                //         scene_->cacheLineSampledPnts3D(pML->mnId, point3D);
                //     }
                // }

                // E. 将新生成的线高斯加入到 GPU 显存
                if (!new_line_sample_points.empty()) {
                    torch::NoGradGuard no_grad;
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    
                    // 调用针对 vector<Point3D> 的重载版本
                    // 这个版本会正确读取 line_dir_ 并初始化各向异性属性
                    gaussians_->auditSetReason("line_seed");
                    gaussians_->increasePcd(new_line_sample_points, getIteration());
                    gaussians_->auditSetReason("map_point_seed");
                    
                    // std::cerr << "[Gaussian Mapper] Added " << new_line_sample_points.size() 
                    //           << " new line-sampled gaussians." << std::endl;
                }
            }

        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::LoopClosingBA:
        {
            std::cerr << "[Gaussian Mapper]Loop Closure Detected."
                      << std::endl;

            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
            // gba-recovery diagnosis: capture this LOOP event's per-keyframe
            // world delta once, before the legacy visible-point path mutates
            // the scene poses.
            if (std::getenv("PHOTO_SLAM_GBA_RECOVERY_DIAG") && !loop_scope_available_) {
                loop_scope_kfs_.clear();
                for (auto& kf : associated_kfs) {
                    LoopScopeKf e;
                    e.kf_id = std::get<0>(kf);
                    auto pkf = scene_->getKeyframe(e.kf_id);
                    e.scene_present = static_cast<bool>(pkf);
                    if (pkf) {
                        Sophus::SE3f oldp = pkf->getPosef();
                        Sophus::SE3f newp = std::get<2>(kf);
                        e.delta = newp.inverse() * oldp;
                        e.rot_deg = e.delta.so3().log().norm() * 180.0 / M_PI;
                        e.trans = e.delta.translation().norm();
                    }
                    loop_scope_kfs_.push_back(e);
                }
                loop_scope_available_ = !loop_scope_kfs_.empty();
                std::cerr << "[GbaDiag] LOOP scope captured kfs="
                          << loop_scope_kfs_.size() << std::endl;
            }
            // Mark the transformed points to avoid transforming more than once
            torch::Tensor point_not_transformed_flags =
                torch::full(
                    {gaussians_->xyz_.size(0)},
                    true,
                    torch::TensorOptions().device(device_type_).dtype(torch::kBool));
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));
            int num_transformed = 0;
            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframeLine> pkf = scene_->getKeyframe(kfid);
                // In case new points are added in handleNewKeyframe()
                int64_t num_new_points = gaussians_->xyz_.size(0) - point_not_transformed_flags.size(0);
                if (num_new_points > 0)
                    point_not_transformed_flags = torch::cat({
                        point_not_transformed_flags,
                        torch::full({num_new_points}, true, point_not_transformed_flags.options())},
                        /*dim=*/0);
                // If kf is already in the scene, evaluate the change in pose,
                // if too large we perform loop correction on its visible model points.
                // If not in the scene, create a new one.
                if (pkf) {
                    auto& pose = std::get<2>(kf);
                    // If is loop closure kf
// if (std::get<4>(kf)) {
// renderAndRecordKeyframe(pkf, result_dir_, "_0_before_loop_correction");
                        Sophus::SE3f original_pose = pkf->getPosef(); // original_pose = old, inv_pose = new
                        Sophus::SE3f inv_pose = pose.inverse();
                        Sophus::SE3f diff_pose = inv_pose * original_pose;
                        bool large_rot = !diff_pose.rotationMatrix().isApprox(
                            Eigen::Matrix3f::Identity(), large_rot_th_);
                        bool large_trans = !diff_pose.translation().isMuchSmallerThan(
                            1.0, large_trans_th_);
                        if (large_rot || large_trans) {
                            std::cerr << "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
                                    << kfid << std::endl;
                            diff_pose.translation() -= inv_pose.translation(); // t = (R_new * t_old + t_new) - t_new
                            diff_pose.translation() *= loop_kf_scale;          // t = s * (R_new * t_old)
                            diff_pose.translation() += inv_pose.translation(); // t = (s * R_new * t_old) + t_new
                            torch::Tensor diff_pose_tensor =
                                tensor_utils::EigenMatrix2TorchTensor(
                                    diff_pose.matrix(), device_type_).transpose(0, 1);
                            {
                                std::unique_lock<std::mutex> lock_render(mutex_render_);
                                gaussians_->scaledTransformVisiblePointsOfKeyframe(
                                    point_not_transformed_flags,
                                    diff_pose_tensor,
                                    pkf->world_view_transform_,
                                    pkf->full_proj_transform_,
                                    pkf->creation_iter_,
                                    stableNumIterExistence(),
                                    num_transformed,
                                    loop_kf_scale); // selected xyz *= s
                            }
                            // Give loop keyframes times of use
                            increaseKeyframeTimesOfUse(pkf, loop_closure_increased_times_of_use_);
// renderAndRecordKeyframe(pkf, result_dir_, "_1_after_loop_transforming_points");
// std::cerr<<num_transformed<<std::endl;
                        }
// }
                    pkf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    pkf->computeTransformTensors();
// if (std::get<4>(kf)) renderAndRecordKeyframe(pkf, result_dir_, "_2_after_pose_correction");
                }
                else {
                    handleNewKeyframe_WithLine(kf);
                }
            }
            if (record_loop_ply_)
                savePly(result_dir_ / (std::to_string(getIteration()) + "_1_after_loop_correction"));
// keyframesToJson(result_dir_ / (std::to_string(getIteration()) + "_0_before_loop_correction"));

            // Get new points (scaled transformation applied in ORB-SLAM3, so this step is performed at last to avoid scaling twice)
            auto& associated_points = opr.associatedMapPoints();
            auto& points = std::get<0>(associated_points);
            auto& colors = std::get<1>(associated_points);

            // Add new points to the model
            if (initial_mapped_ && points.size() >= 30) {
                torch::NoGradGuard no_grad;
                std::unique_lock<std::mutex> lock_render(mutex_render_);
                gaussians_->increasePcd(points, colors, getIteration());
            }

            // Mark this iteration
            loop_closure_iteration_ = true;
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::GlobalBA:
        {
            // geom-sync-v2 里程碑 3: CONSUME AND RECORD ONLY.
            // The deformation is deliberately NOT applied here yet: applying a
            // non-rigid GBA correction correctly requires the per-Gaussian
            // deformation anchor (ref_kf_id_), which does not exist yet. This
            // stage proves the notification chain and the version checks.
            static unsigned long s_last_gba_evt = 0;
            static unsigned long s_applied = 0, s_dup = 0;
            const unsigned long evt = opr.eventId();
            const bool dup = (evt <= s_last_gba_evt);
            if (dup) { ++s_dup; } else { s_last_gba_evt = evt; ++s_applied; }
            int n_in_scene = 0;
            for (const auto& e : opr.globalBAPoses())
                if (scene_->getKeyframe(e.first)) ++n_in_scene;
            printf("[SyncEvent] GlobalBA evt=%lu map=%lu kfs=%zu in_scene=%d "
                   "prev_evt=%lu %s accepted=%lu duplicate=%lu\n",
                   evt, opr.mapGen(), opr.globalBAPoses().size(), n_in_scene,
                   dup ? s_last_gba_evt : (evt == 0 ? 0 : evt - 1),
                   dup ? "DUPLICATE_REJECTED" : "ACCEPTED",
                   s_applied, s_dup);
            fflush(stdout);
        }
        break;

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cerr << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
                      << std::endl;

            float s = opr.mfScale;
            Sophus::SE3f& T = opr.mT;
            if (initial_mapped_) {
                // Apply the scaled transformation on gaussian model points
                {
                    std::unique_lock<std::mutex> lock_render(mutex_render_);
                    gaussians_->applyScaledTransformation(s, T);
                }
                // Apply the scaled transformation to the scene
                scene_->applyScaledTransformation(s, T);
            }
            else { // TODO: the workflow should not come here, delete this branch
                // Apply the scaled transformation to the cached points
                for (auto& pt : scene_->cached_point_cloud_) {
                    // pt <- (s * Ryw * pt + tyw)
                    auto& pt_xyz = pt.second.xyz_;
                    pt_xyz *= s;
                    pt_xyz = T.cast<double>() * pt_xyz;
                }

                // Apply the scaled transformation on gaussian keyframes
                for (auto& kfit : scene_->keyframes()) {
                    std::shared_ptr<GaussianKeyframeLine> pkf = kfit.second;
                    Sophus::SE3f Twc = pkf->getPosef().inverse();
                    Twc.translation() *= s;
                    Sophus::SE3f Tyc = T * Twc;
                    Sophus::SE3f Tcy = Tyc.inverse();
                    pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
                    pkf->computeTransformTensors();
                }
            }
        }
        break;

        default:
        {
            throw std::runtime_error("MappingOperation type not supported!");
        }
        break;
        }
    }
}
#endif

void GaussianMapperLine::handleNewKeyframe(
    std::tuple< unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector<float>,
                std::vector<float>,
                std::string,
                std::vector<float>> &kf)
{
    std::shared_ptr<GaussianKeyframeLine> pkf =
        std::make_shared<GaussianKeyframeLine>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        // Image (left if STEREOLINE)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREOLINE)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBDLINE)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // Add the new keyframe to the scene
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}


void GaussianMapperLine::handleNewKeyframe_WithLine(
    std::tuple< unsigned long/*Id*/,
                unsigned long/*CameraId*/,
                Sophus::SE3f/*pose*/,
                cv::Mat/*image*/,
                bool/*isLoopClosure*/,
                cv::Mat/*auxiliaryImage*/,
                std::vector<float>,
                std::vector<float>,
                std::string,
                std::vector<float>> &kf)
{
    std::shared_ptr<GaussianKeyframeLine> pkf =
        std::make_shared<GaussianKeyframeLine>(std::get<0>(kf), getIteration());
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    auto& pose = std::get<2>(kf);
    pkf->setPose(
        pose.unit_quaternion().cast<double>(),
        pose.translation().cast<double>());
    cv::Mat imgRGB_undistorted, imgAux_undistorted;
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(std::get<1>(kf));
        pkf->setCameraParams(camera);

        // Image (left if STEREOLINE)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREOLINE)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBDLINE)
            camera.undistortImage(imgAux, imgAux_undistorted);
        else
            imgAux_undistorted = imgAux;

        pkf->original_image_ =
            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
        pkf->img_filename_ = std::get<8>(kf);
        pkf->gaus_pyramid_height_ = camera.gaus_pyramid_height_;
        pkf->gaus_pyramid_width_ = camera.gaus_pyramid_width_;
        pkf->gaus_pyramid_times_of_use_ = kf_gaus_pyramid_times_of_use_;
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::combineMappingOperations]KeyFrame Camera not found!");
    }
    // Add the new keyframe to the scene
    pkf->computeTransformTensors();
    scene_->addKeyframe(pkf, &kfid_shuffled_);

    // Give new keyframes times of use and add it to the training sliding window
    increaseKeyframeTimesOfUse(pkf, newKeyframeTimesOfUse());

    // Get dense point cloud from the new keyframe to accelerate training
    pkf->img_undist_ = imgRGB_undistorted;
    pkf->img_auxiliary_undist_ = imgAux_undistorted;
    pkf->kps_pixel_ = std::move(std::get<6>(kf));
    pkf->kps_point_local_ = std::move(std::get<7>(kf));
    if (isdoingInactiveGeoDensify())
        increasePcdByKeyframeInactiveGeoDensify(pkf);

    // Prepare multi resolution images for training
    if (device_type_ == torch::kCUDA) {
        cv::cuda::GpuMat img_gpu;
        img_gpu.upload(pkf->img_undist_);
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::cuda::GpuMat img_resized;
            cv::cuda::resize(img_gpu, img_resized,
                                cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvGpuMat2TorchTensor_Float32(img_resized);
        }
    }
    else {
        pkf->gaus_pyramid_original_image_.resize(num_gaus_pyramid_sub_levels_);
        for (int l = 0; l < num_gaus_pyramid_sub_levels_; ++l) {
            cv::Mat img_resized;
            cv::resize(pkf->img_undist_, img_resized,
                        cv::Size(pkf->gaus_pyramid_width_[l], pkf->gaus_pyramid_height_[l]));
            pkf->gaus_pyramid_original_image_[l] =
                tensor_utils::cvMat2TorchTensor_Float32(img_resized, device_type_);
        }
    }
}


void GaussianMapperLine::generateKfidRandomShuffle()
{
// if (viewpoint_sliding_window_.empty())
//     return;

// std::size_t sliding_window_size = viewpoint_sliding_window_.size();
// kfid_shuffle_.resize(sliding_window_size);
// std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
// std::mt19937 g(rd_());
// std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    if (scene_->keyframes().empty())
        return;

    std::size_t nkfs = scene_->keyframes().size();
    kfid_shuffle_.resize(nkfs);
    std::iota(kfid_shuffle_.begin(), kfid_shuffle_.end(), 0);
    std::mt19937 g(random_seed_ >= 0
                       ? static_cast<unsigned>(random_seed_ + shuffle_counter_++)
                       : rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframeLine>
GaussianMapperLine::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty())
        return nullptr;

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframeLine> viewpoint_cam = nullptr;
    int random_cam_idx;

    if (kfid_shuffled_) {
        int start_shuffle_idx = kfid_shuffle_idx_;
        do {
            // Next shuffled idx
            ++kfid_shuffle_idx_;
            if (kfid_shuffle_idx_ >= kfid_shuffle_.size())
                kfid_shuffle_idx_ = 0;
            // Add 1 time of use to all kfs if they are all unavalible
            if (kfid_shuffle_idx_ == start_shuffle_idx)
                for (auto& kfit : scene_->keyframes())
                    increaseKeyframeTimesOfUse(kfit.second, 1);
            // Get viewpoint kf
            random_cam_idx = kfid_shuffle_[kfid_shuffle_idx_];
            auto random_cam_it = scene_->keyframes().begin();
            for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
                ++random_cam_it;
            viewpoint_cam = (*random_cam_it).second;
        } while (viewpoint_cam->remaining_times_of_use_ <= 0);
    }

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];
    
    // Handle times of use
    --(viewpoint_cam->remaining_times_of_use_);

// auto t2 = std::chrono::steady_clock::now();
// auto t21 = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count();
// std::cerr<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframeLine>
GaussianMapperLine::useOneTrainingErrorSlidingWindowKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Append phase: use the independent, recorded append schedule. The natural
    // error-guided sampler and its remaining budget are NOT re-used/reset;
    // only kfs_used_times_ (LR schedule input) continues to advance.
    if (append_phase_)
        return useOneAppendKeyframe();

    // Frozen-geometry ablation: replay the pre-generated keyframe sampling
    // sequence so that every condition and every repeat sees an identical
    // viewpoint order. Bookkeeping (remaining/used times) is kept identical to
    // the roulette path. Only the reference view is frozen; the trained
    // parameters are not touched.
    if (frozen_kf_replay_ && frozen_kf_step_ < frozen_kf_seq_.size()) {
        const std::size_t target = frozen_kf_seq_[frozen_kf_step_++];
        auto it = scene_->keyframes().begin();
        for (std::size_t i = 0; i < target && it != scene_->keyframes().end(); ++i)
            ++it;
        if (it == scene_->keyframes().end())
            return nullptr;
        std::shared_ptr<GaussianKeyframeLine> sel = it->second;
        --(sel->remaining_times_of_use_);
        auto fid = sel->fid_;
        auto found = kfs_used_times_.find(fid);
        if (found == kfs_used_times_.end())
            kfs_used_times_[fid] = 1;
        else
            ++found->second;
        // Record the view ACTUALLY handed to the trainer, from the live path
        // (not from the pre-generated sequence): 3 ints per step =
        // {completed_optimizer_steps, keyframe map-order index, keyframe fid}.
        frozen_train_views_flat_.push_back(static_cast<int>(frozen_opt_steps_));
        frozen_train_views_flat_.push_back(static_cast<int>(target));
        frozen_train_views_flat_.push_back(static_cast<int>(fid));
        return sel;
    }

    float total_error = 0.0f;
    std::vector<std::shared_ptr<GaussianKeyframeLine>> valid_kfs;

    // 1. 收集所有还有“使用次数”的关键帧，并计算它们的总误差
    for (auto& kfit : scene_->keyframes()) {
        auto pkf = kfit.second;
        if (pkf->remaining_times_of_use_ > 0) {
            // 加上一个极小值 1e-5，防止 error 为 0 导致权重异常
            total_error += (pkf->tracking_error_ + 1e-5f); 
            valid_kfs.push_back(pkf);
        }
    }

    // 2. 如果所有帧的 remaining_times_of_use_ 都用光了，给所有帧补充使用次数
    if (valid_kfs.empty()) {
        for (auto& kfit : scene_->keyframes()) {
            auto pkf = kfit.second;
            increaseKeyframeTimesOfUse(pkf, 1); // 补充1次
            total_error += (pkf->tracking_error_ + 1e-5f);
            valid_kfs.push_back(pkf);
        }
    }

    // 3. 🌟 轮盘赌算法 (Roulette Wheel Selection)：基于误差分配选中概率
    // 误差(tracking_error_)越大的帧，占据的“轮盘面积”越大，越容易被随机数击中
    float random_val = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX) * total_error;
    float accum = 0.0f;
    std::shared_ptr<GaussianKeyframeLine> selected_kf = valid_kfs.back(); // 兜底选项

    for (auto& pkf : valid_kfs) {
        accum += (pkf->tracking_error_ + 1e-5f);
        if (accum >= random_val) {
            selected_kf = pkf;
            break;
        }
    }

    // 4. 更新使用次数统计
    --(selected_kf->remaining_times_of_use_);

    auto viewpoint_fid = selected_kf->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return selected_kf;
}

std::shared_ptr<GaussianKeyframeLine>
GaussianMapperLine::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframeLine> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapperLine::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframeLine> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapperLine::cullKeyframes()
{
    std::unordered_set<unsigned long> kfids =
        pSLAM_->getAtlas()->GetCurrentKeyFrameIds();
    std::vector<unsigned long> kfids_to_erase;
    std::size_t nkfs = scene_->keyframes().size();
    kfids_to_erase.reserve(nkfs);
    for (auto& kfit : scene_->keyframes()) {
        unsigned long kfid = kfit.first;
        if (kfids.find(kfid) == kfids.end()) {
            kfids_to_erase.emplace_back(kfid);
        }
    }

    for (auto& kfid : kfids_to_erase) {
        scene_->keyframes().erase(kfid);
    }
}

void GaussianMapperLine::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframeLine> pkf)
{
    // perf-stage3 audit: this is its own creation source; label the events it
    // produces so ordinary MapPoint seeding can be told apart from it.
    gaussians_->auditSetReason("inactive_geo_densify");
    gaussians_->auditInitIfNeeded();
    budgetInit();                                   // mapper-budget-v2-hash
    budget_cur_kf_fid_ = static_cast<long long>(pkf->fid_);
    // Frozen-count experiment: adding Gaussians here would break the invariant.
    if (frozen_fixed_count_ && frozen_kf_replay_)
        return;
// auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULARLINE:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        assert(pkf->kps_pixel_.size() % 2 == 0);
        int N = pkf->kps_pixel_.size() / 2;
        torch::Tensor kps_pixel_tensor = torch::from_blob(
            pkf->kps_pixel_.data(), {N, 2},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_point_local_tensor = torch::from_blob(
            pkf->kps_point_local_.data(), {N, 3},
            torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        torch::Tensor kps_has3D_tensor = torch::where(
            kps_point_local_tensor.index({torch::indexing::Slice(), 2}) > 0.0f, true, false);

        cv::cuda::GpuMat rgb_gpu;
        rgb_gpu.upload(pkf->img_undist_);
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();

        auto result =
            monocularPinholeInactiveGeoDensifyBySearchingNeighborhoodKeypoints(
                kps_pixel_tensor, kps_has3D_tensor, kps_point_local_tensor, colors,
                monocular_inactive_geo_densify_max_pixel_dist_, pkf->intr_, pkf->image_width_);
        torch::Tensor& points3D_valid = std::get<0>(result);
        torch::Tensor& colors_valid = std::get<1>(result);
        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);
        // Add new points to the cache
        torch::Tensor ref_ids_valid = torch::full(
            {points3D_valid.size(0)}, static_cast<long long>(pkf->fid_),
            torch::TensorOptions().dtype(torch::kLong).device(device_type_));
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
            depth_cache_ref_kf_ids_ = ref_ids_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
            depth_cache_ref_kf_ids_ = torch::cat({depth_cache_ref_kf_ids_, ref_ids_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case STEREOLINE:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat rgb_left_gpu, rgb_right_gpu;
        cv::cuda::GpuMat gray_left_gpu, gray_right_gpu;

        rgb_left_gpu.upload(pkf->img_undist_);
        rgb_right_gpu.upload(pkf->img_auxiliary_undist_);

        // From CV_32FC3 to CV_32FC1
        cv::cuda::cvtColor(rgb_left_gpu, gray_left_gpu, cv::COLOR_RGB2GRAY);
        cv::cuda::cvtColor(rgb_right_gpu, gray_right_gpu, cv::COLOR_RGB2GRAY);

        // From CV_32FC1 to CV_8UC1
        gray_left_gpu.convertTo(gray_left_gpu, CV_8UC1, 255.0);
        gray_right_gpu.convertTo(gray_right_gpu, CV_8UC1, 255.0);

        // Compute disparity
        cv::cuda::GpuMat cv_disp;
        stereo_cv_sgm_->compute(gray_left_gpu, gray_right_gpu, cv_disp);
        cv_disp.convertTo(cv_disp, CV_32F, 1.0 / 16.0);

        // Reproject to get 3D points
        cv::cuda::GpuMat cv_points3D;
        cv::cuda::reprojectImageTo3D(cv_disp, cv_points3D, stereo_Q_, 3);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor disp = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_disp);
        disp = disp.flatten(0, 1).contiguous();
        torch::Tensor points3D = tensor_utils::cvGpuMat2TorchTensor_Float32(cv_points3D);
        points3D = points3D.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor colors = tensor_utils::cvGpuMat2TorchTensor_Float32(rgb_left_gpu);
        colors = colors.permute({1, 2, 0}).flatten(0, 1).contiguous();
    
        // Clear undisired and unreliable stereo points
        torch::Tensor point_valid_flags = torch::full(
            {disp.size(0)}, false, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            // int u = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]);
            // if (u < 0.3 * width || u > 0.7 * width)
            point_valid_flags[idx] = true;
            // idx += width;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx - 3] = true;
            //     point_valid_flags[idx - 2] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx] = true;
            // }
            // idx -= (2 * width);
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            //     point_valid_flags[idx + 3] = true;
            // }
            // idx += width;
            // idx += 3;
            // if (idx < disp.size(0)) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx - 1] = true;
            //     point_valid_flags[idx - 2] = true;
            // }
            // idx -= 6;
            // if (idx > 0) {
            //     point_valid_flags[idx] = true;
            //     point_valid_flags[idx + 1] = true;
            //     point_valid_flags[idx + 2] = true;
            // }
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp > static_cast<float>(stereo_cv_sgm_->getMinDisparity()), true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(disp < static_cast<float>(stereo_cv_sgm_->getNumDisparities()), true, false));

        torch::Tensor points3D_valid = points3D.index({point_valid_flags});
        torch::Tensor colors_valid = colors.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        torch::Tensor ref_ids_valid = torch::full(
            {points3D_valid.size(0)}, static_cast<long long>(pkf->fid_),
            torch::TensorOptions().dtype(torch::kLong).device(device_type_));
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
            depth_cache_ref_kf_ids_ = ref_ids_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
            depth_cache_ref_kf_ids_ = torch::cat({depth_cache_ref_kf_ids_, ref_ids_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case RGBDLINE:
    {
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_0_before_inactive_geo_densify"));
        cv::cuda::GpuMat img_rgb_gpu, img_depth_gpu;
        img_rgb_gpu.upload(pkf->img_undist_);
        img_depth_gpu.upload(pkf->img_auxiliary_undist_);

        // From cv::cuda::GpuMat to torch::Tensor
        torch::Tensor rgb = tensor_utils::cvGpuMat2TorchTensor_Float32(img_rgb_gpu);
        rgb = rgb.permute({1, 2, 0}).flatten(0, 1).contiguous();
        torch::Tensor depth = tensor_utils::cvGpuMat2TorchTensor_Float32(img_depth_gpu);
        depth = depth.flatten(0, 1).contiguous();

        // To clear undisired and unreliable depth
        torch::Tensor point_valid_flags = torch::full(
            {depth.size(0)}, false/*true*/, torch::TensorOptions().dtype(torch::kBool).device(device_type_));
        int nkps_twice = pkf->kps_pixel_.size();
        int width = pkf->image_width_;
        for (int kpidx = 0; kpidx < nkps_twice; kpidx += 2) {
            int idx = static_cast<int>(/*u*/pkf->kps_pixel_[kpidx]) + static_cast<int>(/*v*/pkf->kps_pixel_[kpidx + 1]) * width;
            point_valid_flags[idx] = true;
        }
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth > RGBD_min_depth_, true, false));
        point_valid_flags = torch::logical_and(
            point_valid_flags,
            torch::where(depth < RGBD_max_depth_, true, false));

        torch::Tensor colors_valid = rgb.index({point_valid_flags});

        // Reproject to get 3D points
        torch::Tensor points3D_valid;
        Camera& camera = scene_->cameras_.at(pkf->camera_id_);
        switch (camera.model_id_)
        {
        case Camera::PINHOLE:
        {
            points3D_valid = reprojectDepthPinhole(
                depth, point_valid_flags, pkf->intr_, pkf->image_width_);
        }
        break;
        case Camera::FISHEYE:
        {
            //TODO: support fisheye camera?
            throw std::runtime_error("[Gaussian Mapper]Fisheye cameras are not supported currently!");
        }
        break;
        default:
        {
            throw std::runtime_error("[Gaussian Mapper]Invalid camera model!");
        }
        break;
        }
        points3D_valid = points3D_valid.index({point_valid_flags});

        // Transform points to the world coordinate
        torch::Tensor Twc_tensor =
            tensor_utils::EigenMatrix2TorchTensor(
                Twc.matrix(), device_type_).transpose(0, 1);
        transformPoints(points3D_valid, Twc_tensor);

        // Add new points to the cache
        torch::Tensor ref_ids_valid = torch::full(
            {points3D_valid.size(0)}, static_cast<long long>(pkf->fid_),
            torch::TensorOptions().dtype(torch::kLong).device(device_type_));
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
            depth_cache_ref_kf_ids_ = ref_ids_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
            depth_cache_ref_kf_ids_ = torch::cat({depth_cache_ref_kf_ids_, ref_ids_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    default:
    {
        throw std::runtime_error("[Gaussian Mapper]Unsupported sensor type!");
    }
    break;
    }

    pkf->done_inactive_geo_densify_ = true;
    ++depth_cached_;

    if (depth_cached_ >= max_depth_cached_) {
        depth_cached_ = 0;
        // Add new points to the model
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // mapper-budget-v2-hash: the ONLY algorithmic variable. R100 passes the
        // batch through unchanged; R50 replaces it with a deterministic
        // hash-selected half of the SAME candidate pool.
        torch::Tensor pts = depth_cache_points_, cols = depth_cache_colors_;
        torch::Tensor refs = depth_cache_ref_kf_ids_;
        applyInactiveBudget(pts, cols, &refs);
        gaussians_->increasePcd(pts, cols, getIteration(), refs);
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cerr << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
    gaussians_->auditSetReason("map_point_seed");   // perf-stage3 audit: restore
}

// bool GaussianMapper::needInterruptTraining()
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     return this->interrupt_training_;
// }

// void GaussianMapper::setInterruptTraining(const bool interrupt_training)
// {
//     std::unique_lock<std::mutex> lock_status(this->mutex_status_);
//     this->interrupt_training_ = interrupt_training;
// }

void GaussianMapperLine::recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix)
{
    if (record_rendered_image_) {
        auto image_cv = tensor_utils::torchTensor2CvMat_Float32(rendered);
        cv::cvtColor(image_cv, image_cv, CV_RGB2BGR);
        image_cv.convertTo(image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_img_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + ".jpg"), image_cv);
    }

    if (record_ground_truth_image_) {
        auto gt_image_cv = tensor_utils::torchTensor2CvMat_Float32(ground_truth);
        cv::cvtColor(gt_image_cv, gt_image_cv, CV_RGB2BGR);
        gt_image_cv.convertTo(gt_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_gt_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_gt.jpg"), gt_image_cv);
    }

    if (record_loss_image_) {
        torch::Tensor loss_tensor = torch::abs(rendered - ground_truth);
        auto loss_image_cv = tensor_utils::torchTensor2CvMat_Float32(loss_tensor);
        cv::cvtColor(loss_image_cv, loss_image_cv, CV_RGB2BGR);
        loss_image_cv.convertTo(loss_image_cv, CV_8UC3, 255.0f);
        cv::imwrite(result_loss_dir / (std::to_string(getIteration()) + "_" + std::to_string(kfid) + name_suffix + "_loss.jpg"), loss_image_cv);
    }
}

cv::Mat GaussianMapperLine::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframeLine> pkf = std::make_shared<GaussianKeyframeLine>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRendererWithLine::renderWithLine(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}

cv::Mat GaussianMapperLine::renderFromPose_WithLine(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframeLine> pkf = std::make_shared<GaussianKeyframeLine>();
    pkf->zfar_ = z_far_;
    pkf->znear_ = z_near_;
    // Pose
    pkf->setPose(
        Tcw.unit_quaternion().cast<double>(),
        Tcw.translation().cast<double>());
    try {
        // Camera
        Camera& camera = scene_->cameras_.at(viewer_camera_id_);
        pkf->setCameraParams(camera);
        // Transformations
        pkf->computeTransformTensors();
    }
    catch (std::out_of_range) {
        throw std::runtime_error("[GaussianMapper::renderFromPose]KeyFrame Camera not found!");
    }

    std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> render_pkg;
    {
        std::unique_lock<std::mutex> lock_render(mutex_render_);
        // Render
        render_pkg = GaussianRendererWithLine::renderWithLine(
            pkf,
            height,
            width,
            gaussians_,
            pipe_params_,
            background_,
            override_color_
        );
    }

    // Result
    torch::Tensor masked_image;
    if (main_vision)
        masked_image = std::get<0>(render_pkg) * viewer_main_undistort_mask_[pkf->camera_id_];
    else
        masked_image = std::get<0>(render_pkg) * viewer_sub_undistort_mask_[pkf->camera_id_];
    return tensor_utils::torchTensor2CvMat_Float32(masked_image);
}


void GaussianMapperLine::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframeLine> pkf,
    float &dssim,
    float &psnr,
    float &psnr_gs,
    double &render_time,
    std::filesystem::path result_img_dir,
    std::filesystem::path result_gt_dir,
    std::filesystem::path result_loss_dir,
    std::string name_suffix)
{
    // Evaluation/export is read-only: no autograd graph, no parameter writes.
    torch::NoGradGuard no_grad;
    auto start_timing = std::chrono::steady_clock::now();
    auto render_pkg = GaussianRendererWithLine::renderWithLine(
        pkf,
        pkf->image_height_,
        pkf->image_width_,
        gaussians_,
        pipe_params_,
        background_,
        override_color_
    );
    auto rendered_image = std::get<0>(render_pkg);
    torch::Tensor masked_image = rendered_image * undistort_mask_[pkf->camera_id_];
    torch::cuda::synchronize();
    auto end_timing = std::chrono::steady_clock::now();
    auto render_time_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_timing - start_timing).count();
    render_time = 1e-6 * render_time_ns;
    auto gt_image = pkf->original_image_;

    dssim = loss_utils::ssim(masked_image, gt_image, device_type_).item().toFloat();
    psnr = loss_utils::psnr(masked_image, gt_image).item().toFloat();
    psnr_gs = loss_utils::psnr_gaussian_splatting(masked_image, gt_image).item().toFloat();

    // Shutdown-only, default-off single-frame diagnostic. Same masked_image /
    // gt_image / mask as the original metrics; original outputs unchanged.
    {
        static const bool psnr_diag_enabled = []() {
            const char* e = std::getenv("PHOTO_SLAM_PSNR_DIAG");
            return e && std::atoi(e) != 0;
        }();
        static const long long psnr_diag_fid = []() {
            const char* e = std::getenv("PHOTO_SLAM_PSNR_DIAG_FID");
            return e ? std::atoll(e) : -1LL;
        }();
        if (shutdown_record_phase_ && psnr_diag_enabled && !psnr_diag_done_ &&
            (psnr_diag_fid < 0 ||
             static_cast<long long>(pkf->fid_) == psnr_diag_fid)) {
            append_refinement::writePsnrGaussianSplattingDiagnostic(
                result_dir_ / "psnr_gaussian_splatting_diag.txt",
                "L", static_cast<unsigned long>(pkf->fid_), getIteration(),
                static_cast<int>(pkf->camera_id_), pkf->image_width_,
                pkf->image_height_, gaussians_->active_sh_degree_,
                num_gaus_pyramid_sub_levels_, isdoingGausPyramidTraining(),
                masked_image, gt_image, undistort_mask_[pkf->camera_id_],
                static_cast<double>(psnr), static_cast<double>(psnr_gs));
            psnr_diag_done_ = true;
        }
    }

    recordKeyframeRendered(masked_image, gt_image, pkf->fid_, result_img_dir, result_gt_dir, result_loss_dir, name_suffix);    
}

void GaussianMapperLine::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(image_loss_dir);
    }

    std::filesystem::path render_time_path = result_dir / "render_time.txt";
    std::ofstream out_time(render_time_path);
    out_time << "##[Gaussian Mapper]Render time statistics: keyframe id, time(milliseconds)" << std::endl;

    std::filesystem::path dssim_path = result_dir / "dssim.txt";
    std::ofstream out_dssim(dssim_path);
    out_dssim << "##[Gaussian Mapper]keyframe id, dssim" << std::endl;

    std::filesystem::path psnr_path = result_dir / "psnr.txt";
    std::ofstream out_psnr(psnr_path);
    out_psnr << "##[Gaussian Mapper]keyframe id, psnr" << std::endl;

    std::filesystem::path psnr_gs_path = result_dir / "psnr_gaussian_splatting.txt";
    std::ofstream out_psnr_gs(psnr_gs_path);
    out_psnr_gs << "##[Gaussian Mapper]keyframe id, psnr_gaussian_splatting" << std::endl;

    std::size_t nkfs = scene_->keyframes().size();
    auto kfit = scene_->keyframes().begin();
    float dssim, psnr, psnr_gs;
    double render_time;
    for (std::size_t i = 0; i < nkfs; ++i) {
        renderAndRecordKeyframe((*kfit).second, dssim, psnr, psnr_gs, render_time, image_dir, image_gt_dir, image_loss_dir);
        out_time << (*kfit).first << " " << std::fixed << std::setprecision(8) << render_time << std::endl;

        out_dssim   << (*kfit).first << " " << std::fixed << std::setprecision(10) << dssim   << std::endl;
        out_psnr    << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr    << std::endl;
        out_psnr_gs << (*kfit).first << " " << std::fixed << std::setprecision(10) << psnr_gs << std::endl;

        ++kfit;
    }
}

void GaussianMapperLine::renderAndRecordAllKeyframesReadOnly(std::string name_suffix)
{
    const auto usage_before = append_refinement::captureKfUsageState(*scene_);
    const auto model_before = append_refinement::captureEvalModelState(*gaussians_);
    const auto used_before = kfs_used_times_;
    const int iter_before = getIteration();
    const long long steps_before = frozen_opt_steps_;
    const double train_ms_before = total_training_time_ms_;

    {
        torch::NoGradGuard no_grad;
        shutdown_record_phase_ = true;
        renderAndRecordAllKeyframes(name_suffix);
        shutdown_record_phase_ = false;
    }

    std::string why = "none";
    bool ok = true;
    if (getIteration() != iter_before) {
        ok = false; why = "iteration_changed";
    } else if (frozen_opt_steps_ != steps_before) {
        ok = false; why = "optimizer_step_count_changed";
    } else if (total_training_time_ms_ != train_ms_before) {
        ok = false; why = "training_time_counter_changed";
    } else if (kfs_used_times_ != used_before) {
        ok = false; why = "kfs_used_times_changed";
    } else if (!append_refinement::compareKfUsageState(
                   usage_before, *scene_, why)) {
        ok = false;
    } else if (!append_refinement::compareEvalModelState(
                   model_before, *gaussians_, why)) {
        ok = false;
    }

    std::string tag = name_suffix;
    if (!tag.empty() && tag[0] == '_') tag = tag.substr(1);
    std::ofstream out(result_dir_ / ("eval_guard_" + tag + ".txt"),
                      std::ios::out | std::ios::trunc);
    out << "ok=" << (ok ? 1 : 0) << "\n";
    out << "fail_reason=" << why << "\n";
    out << "iteration=" << iter_before << "\n";
    out << "optimizer_steps=" << steps_before << "\n";
    out << "training_timer=" << train_ms_before << "\n";
    out << "params_changed=0\n";
    out << "adam_steps_changed=0\n";
    out << "training_counters_changed=0\n";
    out << "viewport_or_pyramid_budget_changed=0\n";
    out.flush();
    std::cerr << "[EvalGuard] " << (ok ? "OK" : "FAIL")
              << " suffix=" << name_suffix << " reason=" << why << std::endl;
}

void GaussianMapperLine::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");
    gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapperLine::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(result_dir)

    std::filesystem::path result_path = result_dir / "cameras.json";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json file at " + result_path.string());

    Json::Value json_root;
    Json::StreamWriterBuilder builder;
    const std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());

    int i = 0;
    for (const auto& kfit : scene_->keyframes()) {
        const auto pkf = kfit.second;
        Eigen::Matrix4f Rt;
        Rt.setZero();
        Eigen::Matrix3f R = pkf->R_quaternion_.toRotationMatrix().cast<float>();
        Rt.topLeftCorner<3, 3>() = R;
        Eigen::Vector3f t = pkf->t_.cast<float>();
        Rt.topRightCorner<3, 1>() = t;
        Rt(3, 3) = 1.0f;

        Eigen::Matrix4f Twc = Rt.inverse();
        Eigen::Vector3f pos = Twc.block<3, 1>(0, 3);
        Eigen::Matrix3f rot = Twc.block<3, 3>(0, 0);

        Json::Value json_kf;
        json_kf["id"] = static_cast<Json::Value::UInt64>(pkf->fid_);
        json_kf["img_name"] = pkf->img_filename_; //(std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_));
        json_kf["width"] = pkf->image_width_;
        json_kf["height"] = pkf->image_height_;

        json_kf["position"][0] = pos.x();
        json_kf["position"][1] = pos.y();
        json_kf["position"][2] = pos.z();

        json_kf["rotation"][0][0] = rot(0, 0);
        json_kf["rotation"][0][1] = rot(0, 1);
        json_kf["rotation"][0][2] = rot(0, 2);
        json_kf["rotation"][1][0] = rot(1, 0);
        json_kf["rotation"][1][1] = rot(1, 1);
        json_kf["rotation"][1][2] = rot(1, 2);
        json_kf["rotation"][2][0] = rot(2, 0);
        json_kf["rotation"][2][1] = rot(2, 1);
        json_kf["rotation"][2][2] = rot(2, 2);

        json_kf["fy"] = graphics_utils::fov2focal(pkf->FoVy_, pkf->image_height_);
        json_kf["fx"] = graphics_utils::fov2focal(pkf->FoVx_, pkf->image_width_);

        json_root[i] = Json::Value(json_kf);
        ++i;
    }

    writer->write(json_root, &out_stream);
}

void GaussianMapperLine::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(result_dir)
    std::filesystem::path result_path = result_dir / "cfg_args";
    std::ofstream out_stream;
    out_stream.open(result_path);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open file at " + result_path.string());

    out_stream << "Namespace("
               << "eval=" << (model_params_.eval_ ? "True" : "False") << ", "
               << "images=" << "\'" << model_params_.images_ << "\', "
               << "model_path=" << "\'" << model_params_.model_path_.string() << "\', "
               << "resolution=" << model_params_.resolution_ << ", "
               << "sh_degree=" << model_params_.sh_degree_ << ", "
               << "source_path=" << "\'" << model_params_.source_path_.string() << "\', "
               << "white_background=" << (model_params_.white_background_ ? "True" : "False") << ", "
               << ")";

    out_stream.close();
}

void GaussianMapperLine::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(result_dir)
    std::filesystem::path result_path = result_dir / ("keyframe_used_times" + name_suffix + ".txt");
    std::ofstream out_stream;
    out_stream.open(result_path, std::ios::app);
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open json at " + result_path.string());

    out_stream << "##[Gaussian Mapper]Iteration " << getIteration() << " keyframe id, used times, remaining times:\n";
    for (const auto& used_times_it : kfs_used_times_)
        out_stream << used_times_it.first << " "
                   << used_times_it.second << " "
                   << scene_->keyframes().at(used_times_it.first)->remaining_times_of_use_
                   << "\n";
    out_stream << "##=========================================" <<std::endl;

    out_stream.close();
}


void GaussianMapperLine::writeTrainingLog(int64_t iter_time, float ema_loss, int64_t num_points)
{
    // 确保目录存在
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(result_dir_)

    std::filesystem::path log_path = result_dir_ / "gaussian_training_log.txt";
    bool file_exists = std::filesystem::exists(log_path);
    
    std::ofstream out_stream;
    out_stream.open(log_path, std::ios::app); // 以追加模式打开
    
    if (!out_stream.is_open())
        throw std::runtime_error("Cannot open log file at " + log_path.string());

    // 如果文件是新建的，写入表头
    if (!file_exists) {
        out_stream << "Iteration Time(ms) EMA_Loss Num_Points\n";
    }

    // 写入当前迭代的数据
    out_stream << getIteration() << " "
               << iter_time << " "
               << ema_loss << " "
               << num_points << "\n";

    out_stream.close();
}


int GaussianMapperLine::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapperLine::increaseIteration(const int inc)
{
    {
        std::unique_lock<std::mutex> lock(mutex_status_);
        iteration_ += inc;
    }
    // mapper-budget-v2-hash: the ONE place BOTH training loops pass through, so
    // the per-step record and the snapshot trigger fire regardless of route.
    // Called AFTER the status lock is released (the snapshot writes 10s of MB).
    recordTrainStep();
}

float GaussianMapperLine::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapperLine::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapperLine::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapperLine::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapperLine::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapperLine::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapperLine::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapperLine::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapperLine::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapperLine::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapperLine::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapperLine::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapperLine::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapperLine::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapperLine::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapperLine::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapperLine::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapperLine::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapperLine::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapperLine::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapperLine::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapperLine::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapperLine::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapperLine::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapperLine::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapperLine::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapperLine::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapperLine::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapperLine::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapperLine::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParametersLine GaussianMapperLine::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParametersLine params;
    params.position_lr_init = opt_params_.position_lr_init_;
    params.feature_lr = opt_params_.feature_lr_;
    params.opacity_lr = opt_params_.opacity_lr_;
    params.scaling_lr = opt_params_.scaling_lr_;
    params.rotation_lr = opt_params_.rotation_lr_;
    params.percent_dense = opt_params_.percent_dense_;
    params.lambda_dssim = opt_params_.lambda_dssim_;
    params.opacity_reset_interval = opt_params_.opacity_reset_interval_;
    params.densify_grad_th = opt_params_.densify_grad_threshold_;
    params.densify_interval = opt_params_.densification_interval_;
    params.new_kf_times_of_use = new_keyframe_times_of_use_;
    params.stable_num_iter_existence = stable_num_iter_existence_;
    params.keep_training = keep_training_;
    params.do_gaus_pyramid_training = do_gaus_pyramid_training_;
    params.do_inactive_geo_densify = inactive_geo_densify_;
    return params;
}

void GaussianMapperLine::setVaribleParameters(const VariableParametersLine &params)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = params.position_lr_init;
    opt_params_.feature_lr_ = params.feature_lr;
    opt_params_.opacity_lr_ = params.opacity_lr;
    opt_params_.scaling_lr_ = params.scaling_lr;
    opt_params_.rotation_lr_ = params.rotation_lr;
    opt_params_.percent_dense_ = params.percent_dense;
    gaussians_->setPercentDense(params.percent_dense);
    opt_params_.lambda_dssim_ = params.lambda_dssim;
    opt_params_.opacity_reset_interval_ = params.opacity_reset_interval;
    opt_params_.densify_grad_threshold_ = params.densify_grad_th;
    opt_params_.densification_interval_ = params.densify_interval;
    new_keyframe_times_of_use_ = params.new_kf_times_of_use;
    stable_num_iter_existence_ = params.stable_num_iter_existence;
    keep_training_ = params.keep_training;
    do_gaus_pyramid_training_ = params.do_gaus_pyramid_training;
    inactive_geo_densify_ = params.do_inactive_geo_densify;
}

void GaussianMapperLine::saveDebugMapToObj(
    const std::filesystem::path& path,
    const std::vector<ORB_SLAM3::MapPoint*>& vpMPs,
    const std::vector<ORB_SLAM3::MapLine*>& vpMPLs)
{
    std::ofstream obj_file(path);
    if (!obj_file.is_open()) {
        std::cerr << "[Gaussian Mapper] ERROR: Cannot open " << path << " for writing debug obj." << std::endl;
        return;
    }

    std::cerr << "[Gaussian Mapper] DEBUG: Saving map to " << path << " ..." << std::endl;

    int vertex_count = 1; // OBJ 索引从 1 开始

    obj_file << "# Exported by Photo-SLAM-L Debugger\n";
    obj_file << "# Red = MapPoints, Green = MapLines\n";

    // 1. 写入 MapPoints (红色)
    for (const auto& pMP : vpMPs) {
        if (!pMP || pMP->isBad()) continue;
        auto pos = pMP->GetWorldPos();
        // v x y z r g b
        obj_file << "v " << pos.x() << " " << pos.y() << " " << pos.z() << " 1.0 0.0 0.0\n";
        obj_file << "p " << vertex_count++ << "\n";
    }

    // 2. 写入 MapLines (绿色)
    for (const auto& pML : vpMPLs) {
        if (!pML || pML->isBad()) continue;
        auto endpoints = pML->GetLineWorldPos();

        // 线段端点 1 (绿色)
        obj_file << "v " << endpoints.first.x() << " " << endpoints.first.y() << " " << endpoints.first.z() << " 0.0 1.0 0.0\n";
        int idx1 = vertex_count++;

        // 线段端点 2 (绿色)
        obj_file << "v " << endpoints.second.x() << " " << endpoints.second.y() << " " << endpoints.second.z() << " 0.0 1.0 0.0\n";
        int idx2 = vertex_count++;

        // 连线
        obj_file << "l " << idx1 << " " << idx2 << "\n";
    }

    obj_file.close();
    std::cerr << "[Gaussian Mapper] DEBUG: Saved " << (vertex_count - 1) << " vertices total." << std::endl;
}

void GaussianMapperLine::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
{
    this->gaussians_->loadPly(ply_path);

    // Camera
    if (!camera_path.empty() && std::filesystem::exists(camera_path)) {
        cv::FileStorage camera_file(camera_path.string().c_str(), cv::FileStorage::READ);
        if(!camera_file.isOpened())
            throw std::runtime_error("[Gaussian Mapper]Failed to open settings file at: " + camera_path.string());

        Camera camera;
        camera.camera_id_ = 0;
        camera.width_ = camera_file["Camera.w"].operator int();
        camera.height_ = camera_file["Camera.h"].operator int();

        std::string camera_type = camera_file["Camera.type"].string();
        if (camera_type == "Pinhole") {
            camera.setModelId(Camera::CameraModelType::PINHOLE);

            float fx = camera_file["Camera.fx"].operator float();
            float fy = camera_file["Camera.fy"].operator float();
            float cx = camera_file["Camera.cx"].operator float();
            float cy = camera_file["Camera.cy"].operator float();

            float k1 = camera_file["Camera.k1"].operator float();
            float k2 = camera_file["Camera.k2"].operator float();
            float p1 = camera_file["Camera.p1"].operator float();
            float p2 = camera_file["Camera.p2"].operator float();
            float k3 = camera_file["Camera.k3"].operator float();

            cv::Mat K = (
                cv::Mat_<float>(3, 3)
                    << fx, 0.f, cx,
                        0.f, fy, cy,
                        0.f, 0.f, 1.f
            );

            camera.params_[0] = fx;
            camera.params_[1] = fy;
            camera.params_[2] = cx;
            camera.params_[3] = cy;

            std::vector<float> dist_coeff = {k1, k2, p1, p2, k3};
            camera.dist_coeff_ = cv::Mat(5, 1, CV_32F, dist_coeff.data());
            camera.initUndistortRectifyMapAndMask(K, cv::Size(camera.width_, camera.height_), K, false);

            undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    camera.undistort_mask, device_type_);

            cv::Mat viewer_main_undistort_mask;
            int viewer_image_height_main_ = camera.height_ * rendered_image_viewer_scale_main_;
            int viewer_image_width_main_ = camera.width_ * rendered_image_viewer_scale_main_;
            cv::resize(camera.undistort_mask, viewer_main_undistort_mask,
                       cv::Size(viewer_image_width_main_, viewer_image_height_main_));
            viewer_main_undistort_mask_[camera.camera_id_] =
                tensor_utils::cvMat2TorchTensor_Float32(
                    viewer_main_undistort_mask, device_type_);

        }
        else {
            throw std::runtime_error("[Gaussian Mapper]Unsupported camera model: " + camera_path.string());
        }

        if (!viewer_camera_id_set_) {
            viewer_camera_id_ = camera.camera_id_;
            viewer_camera_id_set_ = true;
        }
        this->scene_->addCamera(camera);
    }

    // Ready
    this->initial_mapped_ = true;
    increaseIteration();
}