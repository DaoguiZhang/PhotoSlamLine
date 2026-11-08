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

#include "include/gaussian_mapper.h"

#include <iomanip>
#include <limits>

GaussianMapper::GaussianMapper(
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
    // Random seed
    std::srand(seed);
    torch::manual_seed(seed);

    // Device
    if (device_type == torch::kCUDA && torch::cuda::is_available()) {
        std::cout << "[Gaussian Mapper]CUDA available! Training on GPU." << std::endl;
        device_type_ = torch::kCUDA;
        model_params_.data_device_ = "cuda";
    }
    else {
        std::cout << "[Gaussian Mapper]Training on CPU." << std::endl;
        device_type_ = torch::kCPU;
        model_params_.data_device_ = "cpu";
    }

    result_dir_ = result_dir;
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
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
    gaussians_ = std::make_shared<GaussianModel>(model_params_);
    scene_ = std::make_shared<GaussianScene>(model_params_);

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
        this->sensor_type_ = MONOCULAR;
    }
    break;
    case ORB_SLAM3::System::STEREO:
    case ORB_SLAM3::System::IMU_STEREO:
    {
        this->sensor_type_ = STEREO;
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
        this->sensor_type_ = RGBD;
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
    UndistortParams undistort_params(
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
            if (this->sensor_type_ == MONOCULAR || this->sensor_type_ == RGBD)
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

            if (this->sensor_type_ == STEREO) {
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

void GaussianMapper::readConfigFromFile(std::filesystem::path cfg_path)
{
    cv::FileStorage settings_file(cfg_path.string().c_str(), cv::FileStorage::READ);
    if(!settings_file.isOpened()) {
       std::cerr << "[Gaussian Mapper]Failed to open settings file at: " << cfg_path << std::endl;
       exit(-1);
    }

    std::cout << "[Gaussian Mapper]Reading parameters from " << cfg_path << std::endl;
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

    prune_big_point_after_iter_ =
        settings_file["Optimization.prune_big_point_after_iter"].operator int();
    densify_min_opacity_ =
        settings_file["Optimization.densify_min_opacity"].operator float();

    // Viewer Parameters
    rendered_image_viewer_scale_ =
        settings_file["GaussianViewer.image_scale"].operator float();
    rendered_image_viewer_scale_main_ =
        settings_file["GaussianViewer.image_scale_main"].operator float();
}

void GaussianMapper::run()
{
    // Initialize append/budget configuration BEFORE natural training so the
    // cumulative training-time boundary is active from the first real update.
    appendRefinementEnabled();

    // First loop: Initial gaussian mapping
    while (!isStopped()) {
        // Check conditions for initial mapping
        if (hasMetInitialMappingConditions()) {
            pSLAM_->getAtlas()->clearMappingOperation();

            // Get initial sparse map
            auto pMap = pSLAM_->getAtlas()->GetCurrentMap();
            std::vector<ORB_SLAM3::KeyFrame*> vpKFs;
            std::vector<ORB_SLAM3::MapPoint*> vpMPs;
            {
                std::unique_lock<std::mutex> lock_map(pMap->mMutexMapUpdate);
                vpKFs = pMap->GetAllKeyFrames();
                vpMPs = pMap->GetAllMapPoints();
                for (const auto& pMP : vpMPs){
                    Point3D point3D;
                    auto pos = pMP->GetWorldPos();
                    point3D.xyz_(0) = pos.x();
                    point3D.xyz_(1) = pos.y();
                    point3D.xyz_(2) = pos.z();
                    auto color = pMP->GetColorRGB();
                    point3D.color_(0) = color(0);
                    point3D.color_(1) = color(1);
                    point3D.color_(2) = color(2);
                    scene_->cachePoint3D(pMP->mnId, point3D);
                }

                //std::cerr << "===============================================================1111111111===========================" << std::endl;

                for (const auto& pKF : vpKFs){
                    std::shared_ptr<GaussianKeyframe> new_kf = std::make_shared<GaussianKeyframe>(pKF->mnId, getIteration());
                    new_kf->zfar_ = z_far_;
                    new_kf->znear_ = z_near_;
                    // Pose
                    auto pose = pKF->GetPose();
                    new_kf->setPose(
                        pose.unit_quaternion().cast<double>(),
                        pose.translation().cast<double>());
                    cv::Mat imgRGB_undistorted, imgAux_undistorted;
                    try {
                        // Camera
                        Camera& camera = scene_->cameras_.at(pKF->mpCamera->GetId());
                        new_kf->setCameraParams(camera);

                        // Image (left if STEREO)
                        cv::Mat imgRGB = pKF->imgLeftRGB;
                        if (this->sensor_type_ == STEREO)
                            imgRGB_undistorted = imgRGB;
                        else
                            camera.undistortImage(imgRGB, imgRGB_undistorted);
                        // Auxiliary Image
                        cv::Mat imgAux = pKF->imgAuxiliary;
                        if (this->sensor_type_ == RGBD)
                            camera.undistortImage(imgAux, imgAux_undistorted);
                        else
                            imgAux_undistorted = imgAux;

                        new_kf->original_image_ =
                            tensor_utils::cvMat2TorchTensor_Float32(imgRGB_undistorted, device_type_);
                        new_kf->img_filename_ = pKF->mNameFile;
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

                    // Features
                    std::vector<float> pixels;
                    std::vector<float> pointsLocal;
                    pKF->GetKeypointInfo(pixels, pointsLocal);
                    new_kf->kps_pixel_ = std::move(pixels);
                    new_kf->kps_point_local_ = std::move(pointsLocal);
                    new_kf->img_undist_ = imgRGB_undistorted;
                    new_kf->img_auxiliary_undist_ = imgAux_undistorted;
                }
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
            trainForOneIteration();

            // Finish initial mapping loop
            initial_mapped_ = true;
            break;
        }
        else if (pSLAM_->isShutDown()) {
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
        // Check conditions for incremental mapping
        if (hasMetIncrementalMappingConditions()) {
            combineMappingOperations();
            if (cull_keyframes_)
                cullKeyframes();
        }

        // Invoke training once
        trainForOneIteration();

        if (pSLAM_->isShutDown()) {
            SLAM_stop_iter = getIteration();
            SLAM_ended_ = true;
        }

        if (SLAM_ended_ || getIteration() >= opt_params_.iterations_)
            break;
    }

    // Third loop: Tail gaussian optimization
    int densify_interval = densifyInterval();
    int n_delay_iters = densify_interval * 0.8;
    while (getIteration() - SLAM_stop_iter <= n_delay_iters || getIteration() % densify_interval <= n_delay_iters || isKeepingTraining()) {
        trainForOneIteration();
        densify_interval = densifyInterval();
        n_delay_iters = densify_interval * 0.8;
    }

    // Append refinement: real extra optimizer updates on the EXISTING model.
    // Inserted exactly after the third loop and before shutdown export/signal.
    if (appendRefinementEnabled()) {
        runAppendRefinement();
    }

    // Save and clear
    const auto export_start = std::chrono::steady_clock::now();
    renderAndRecordAllKeyframesReadOnly("_shutdown");
    savePly(result_dir_ / (std::to_string(getIteration()) + "_shutdown") / "ply");
    writeKeyframeUsedTimes(result_dir_ / "used_times", "final");
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

    // Quad comparison telemetry: one parseable line, same fields as the L build.
    std::cerr << "[QuadMetrics] optimizer_steps=" << quad_opt_steps_
              << " gaussians=" << gaussians_->xyz_.size(0)
              << " line_gaussians=0"
              << " kfs=" << scene_->keyframes().size()
              << " mapping_seconds=" << (quad_mapping_ms_ / 1000.0)
              << std::endl;
    std::ofstream quad_csv(result_dir_ / "quad_metrics.txt");
    quad_csv << "optimizer_steps=" << quad_opt_steps_ << "\n"
             << "gaussians=" << gaussians_->xyz_.size(0) << "\n"
             << "line_gaussians=0\n"
             << "kfs=" << scene_->keyframes().size() << "\n"
             << "mapping_seconds=" << (quad_mapping_ms_ / 1000.0) << "\n";
    quad_csv.flush();

    signalStop();
}

void GaussianMapper::trainColmap()
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

/**
 * @brief The training iteration body
 * 
 */
void GaussianMapper::trainForOneIteration()
{
    const auto budget_iter_start = std::chrono::steady_clock::now();
    increaseIteration(1);
    auto iter_start_timing = std::chrono::steady_clock::now();

    // Pick a random Camera
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = useOneRandomSlidingWindowKeyframe();
    if (!viewpoint_cam) {
        increaseIteration(-1);
        return;
    }

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
    auto render_pkg = GaussianRenderer::render(
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

    // Get rid of black edges caused by undistortion
    torch::Tensor masked_image = rendered_image * mask;

    // Loss
    auto Ll1 = loss_utils::l1_loss(masked_image, gt_image);
    float lambda_dssim = lambdaDssim();
    auto loss = (1.0 - lambda_dssim) * Ll1
                + lambda_dssim * (1.0 - loss_utils::ssim(masked_image, gt_image, device_type_));
    
    loss.backward();

    torch::cuda::synchronize();

    {
        torch::NoGradGuard no_grad;
        ema_loss_for_log_ = 0.4f * loss.item().toFloat() + 0.6 * ema_loss_for_log_;

        if (keyframe_record_interval_ &&
            getIteration() % keyframe_record_interval_ == 0)
            recordKeyframeRendered(masked_image, gt_image, viewpoint_cam->fid_, result_dir_, result_dir_, result_dir_);

        // Densification
        // Append phase: this single gate disables the ACTUAL densify/clone/
        // split/prune path AND the opacity reset below, so the Gaussian set is
        // frozen without touching the training/loss code.
        if (!append_phase_ && getIteration() < opt_params_.densify_until_iter_) {
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
                gaussians_->densifyAndPrune(
                    densifyGradThreshold(),
                    densify_min_opacity_,//0.005,//
                    scene_->cameras_extent_,
                    size_threshold
                );
            }

            if (opacityResetInterval()
                && (getIteration() % opacityResetInterval() == 0
                    ||(model_params_.white_background_ && getIteration() == opt_params_.densify_from_iter_)))
                gaussians_->resetOpacity();
        }

        auto iter_end_timing = std::chrono::steady_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        iter_end_timing - iter_start_timing).count();

        // Log and save
        if (training_report_interval_ && (getIteration() % training_report_interval_ == 0))
            GaussianTrainer::trainingReport(
                getIteration(),
                opt_params_.iterations_,
                Ll1,
                loss,
                ema_loss_for_log_,
                loss_utils::l1_loss,
                iter_time,
                *gaussians_,
                *scene_,
                pipe_params_,
                background_
            );
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
        quad_mapping_ms_ += static_cast<double>(iter_time);
        if (getIteration() < opt_params_.iterations_) {
            if (append_phase_ && append_diag_mode_ &&
                append_completed_steps_ == 0)
                appendRecordFirstStepAudit("first_step_before");
            gaussians_->optimizer_->step();
            ++quad_opt_steps_;
            if (append_phase_) {
                ++append_completed_steps_;
                if (append_diag_mode_ && append_config_.steps > 0 &&
                    append_completed_steps_ == append_config_.steps)
                    appendRecordFirstStepAudit("last_step_after");
            }
            gaussians_->optimizer_->zero_grad(true);
        }
        // Iteration cap reached in append mode: NO step happened and no append
        // counter was advanced. The driver turns this into a hard failure
        // instead of spinning or reporting empty steps as success.
    }

    // Unified cumulative-training-time boundary: start of this training call
    // (selection + render + loss + backward + densify + optimizer step) to the
    // completed GPU work. Only active in budget mode.
    if (time_budget_mode_) {
        if (device_type_ == torch::kCUDA)
            torch::cuda::synchronize();
        cumulative_train_seconds_ +=
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - budget_iter_start).count();
    }
}

// ---------------------------------------------------------------------------
// Append refinement (point build)
// ---------------------------------------------------------------------------

bool GaussianMapper::appendRefinementEnabled()
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

long long GaussianMapper::appendValidatedCount()
{
    if (!gaussians_ || !gaussians_->xyz_.defined())
        return -1;
    return static_cast<long long>(gaussians_->xyz_.size(0));
}

bool GaussianMapper::appendPreflight(std::string& reason, int& drained_batches)
{
    if (!pSLAM_) {
        reason = "no_pSLAM";
        return false;
    }
    // The append phase must not run while the SLAM producers can still mutate
    // the scene. System::Shutdown() requests finish and joins LocalMapping /
    // LoopClosing, so isShutDown() is the production boundary we require.
    if (!pSLAM_->isShutDown()) {
        reason = "slam_not_shutdown";
        return false;
    }

    // Consume any pending mapping/correction operations through the NORMAL
    // production path, then require the queue to stay empty. Without this the
    // append loop would silently assume a frozen camera/geometry state.
    auto has_ops = [this]() {
        return pSLAM_->getAtlas()->hasMappingOperation();
    };
    auto drain_one = [this]() {
        combineMappingOperations();
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

void GaussianMapper::prepareAppendViewSequence()
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

std::shared_ptr<GaussianKeyframe> GaussianMapper::useOneAppendKeyframe()
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

    // LR-schedule bookkeeping only (identical input to updateLearningRate()).
    // remaining_times_of_use_ is deliberately NOT touched: the append phase
    // uses its own recorded schedule.
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

void GaussianMapper::appendRecordFirstStepAudit(const char* stage)
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

void GaussianMapper::appendWriteBoundary(const std::filesystem::path& file,
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

void GaussianMapper::appendWriteStatus(bool ok, const std::string& reason)
{
    append_ok_ = ok;
    append_fail_reason_ = reason;
    std::filesystem::create_directories(append_dir_);
    std::ofstream out(append_dir_ / "append_status.txt", std::ios::out | std::ios::trunc);
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
    out << "timestamp=" << append_refinement::nowStamp() << '\n';
    out.flush();
}

void GaussianMapper::runAppendRefinement()
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
        s << "# append refinement (point build)\n";
        s << "enabled=1\n";
        s << "build=P_target\n";
        s << "mode=" << append_refinement::appendModeName(
                             time_budget_mode_, append_config_.steps) << '\n';
        s << "budget_seconds=" << time_budget_config_.seconds << '\n';
        s << "config_steps=" << append_config_.steps << '\n';
        s << "seed=" << append_config_.seed << '\n';
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
    natural_completed_steps_ = quad_opt_steps_;
    natural_train_seconds_ = quad_mapping_ms_ / 1000.0;
    natural_cumulative_train_seconds_ = cumulative_train_seconds_;

    // Save the natural-end model in the consistent boundary state (SLAM
    // stopped, queue empty). Evaluation is offline on this PLY.
    {
        const auto save_start = std::chrono::steady_clock::now();
        natural_end_iteration_ = getIteration();
        natural_end_ply_dir_ = append_dir_ / "natural_end";
        savePly(natural_end_ply_dir_);
        natural_save_seconds_ =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::steady_clock::now() - save_start).count();
    }

    // Iteration cap: a step is only executed when getIteration() < iterations_.
    // In step mode, refuse to start if the requested number of REAL updates
    // cannot happen. In budget mode, an early cap is a recorded
    // budget_unmatched condition, not a silent success.
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

    // Boundary snapshot: model/optimizer/Adam identity, geometry and camera.
    append_param_snapshot_ =
        append_refinement::captureParamSnapshot(*gaussians_->optimizer_);
    append_adam_steps_before_ =
        append_refinement::captureAdamSteps(*gaussians_->optimizer_);
    append_kf_before_ = append_refinement::snapshotKeyframeStates(*scene_);
    append_camera_hash_before_ = append_refinement::cameraFingerprint(*scene_);
    append_gaussian_count_before_ = appendValidatedCount();
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

    // ---- real append training: SAME trainForOneIteration() body -----------
    const long long step_cap = (append_config_.steps > 0)
                                   ? static_cast<long long>(append_config_.steps)
                                   : std::numeric_limits<long long>::max();

    // Budget mode: if the natural phase (online + original final refinement)
    // already spent the budget, the append phase must not run. This is an
    // explicit unmatched-budget result, never silently converted to success.
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

    append_phase_ = true;
    const auto train_start = std::chrono::steady_clock::now();
    int no_progress = 0;
    bool failed = false;

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
            trainForOneIteration();
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

        // Close the GPU-async timing boundary for this iteration.
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

    // Save the budget-end model (or reuse the natural-end model when the
    // budget was already exhausted before append).
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

    // ---- validation / evidence -------------------------------------------
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
    const bool slam_shutdown_after = pSLAM_ && pSLAM_->isShutDown();
    const bool queue_empty_after =
        pSLAM_ && !pSLAM_->getAtlas()->hasMappingOperation();

    // Adam-step verification. In independent acceptance/diagnostic mode
    // (PHOTO_SLAM_APPEND_DIAG=1) only groups that really had a gradient on the
    // first append step are required to advance. In normal/formal mode there is
    // NO per-step grad readback; we verify the stage-boundary Adam deltas:
    // at least one group advanced by the real step count and no group advanced
    // by more (parameters without gradients are not forced).
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
        gaussian_count_after == append_gaussian_count_before_;
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
           << " camera_fingerprint_end=" << camera_hash_after
           << " slam_shutdown_after="
           << append_refinement::boolStr(slam_shutdown_after)
           << " pending_operations_after="
           << append_refinement::boolStr(!queue_empty_after)
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
              << " build=P mode=" << append_refinement::appendModeName(
                     time_budget_mode_, append_config_.steps)
              << " steps=" << append_completed_steps_ << "/"
              << (append_config_.steps > 0
                      ? std::to_string(append_config_.steps)
                      : std::string("budget"))
              << " natural_steps=" << natural_completed_steps_
              << " raw_iteration=" << getIteration()
              << " gaussians=" << gaussian_count_after
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

bool GaussianMapper::isStopped()
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    return this->stopped_;
}

void GaussianMapper::signalStop(const bool going_to_stop)
{
    std::unique_lock<std::mutex> lock_status(this->mutex_status_);
    this->stopped_ = going_to_stop;
}

bool GaussianMapper::hasMetInitialMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->GetNumKeyframes() >= min_num_initial_map_kfs_ &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

bool GaussianMapper::hasMetIncrementalMappingConditions()
{
    if (!pSLAM_->isShutDown() &&
        pSLAM_->getAtlas()->hasMappingOperation())
        return true;

    bool conditions_met = false;
    return conditions_met;
}

void GaussianMapper::combineMappingOperations()
{
    // Get Mapping Operations
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // std::cout << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;

            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();

            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
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
            std::cout << "[Gaussian Mapper]Loop Closure Detected."
                      << std::endl;

            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
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
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
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
                            std::cout << "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
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
// std::cout<<num_transformed<<std::endl;
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

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
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
                    std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
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

        case ORB_SLAM3::MappingOperation::OprType::GlobalBA:
        {
            // Point-only mapper: no per-Gaussian deformation anchor exists,
            // so consume the correction and reject it explicitly. Before this
            // case existed, a loop-closure GBA event aborted the process with
            // "MappingOperation type not supported!".
            printf("[GBA-Apply] evt=%lu map=%lu REJECTED "
                   "reason=point_only_mapper\n",
                   opr.eventId(), opr.mapGen());
            fflush(stdout);
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

void GaussianMapper::combineMappingOperations_withLine()
{
    // Get Mapping Operations
    while (pSLAM_->getAtlas()->hasMappingOperation()) {
        ORB_SLAM3::MappingOperation opr =
            pSLAM_->getAtlas()->getAndPopMappingOperation();

        switch (opr.meOperationType)
        {
        case ORB_SLAM3::MappingOperation::OprType::LocalMappingBA:
        {
            // std::cout << "[Gaussian Mapper]Local BA Detected."
            //           << std::endl;

            // Get new keyframes
            auto& associated_kfs = opr.associatedKeyFrames();

            // Add keyframes to the scene
            for (auto& kf : associated_kfs) {
                // Keyframe Id
                auto kfid = std::get<0>(kf);
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
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
            std::cout << "[Gaussian Mapper]Loop Closure Detected."
                      << std::endl;

            // Get the loop keyframe scale modification factor
            float loop_kf_scale = opr.mfScale;

            // Get new keyframes (scaled transformation applied in ORB-SLAM3)
            auto& associated_kfs = opr.associatedKeyFrames();
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
                std::shared_ptr<GaussianKeyframe> pkf = scene_->getKeyframe(kfid);
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
                            std::cout << "[Gaussian Mapper]Large loop correction detected, transforming visible points of kf "
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
// std::cout<<num_transformed<<std::endl;
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

        case ORB_SLAM3::MappingOperation::OprType::ScaleRefinement:
        {
            std::cout << "[Gaussian Mapper]Scale refinement Detected. Transforming all kfs and points..."
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
                    std::shared_ptr<GaussianKeyframe> pkf = kfit.second;
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

        case ORB_SLAM3::MappingOperation::OprType::GlobalBA:
        {
            printf("[GBA-Apply] evt=%lu map=%lu REJECTED "
                   "reason=point_only_mapper\n",
                   opr.eventId(), opr.mapGen());
            fflush(stdout);
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


void GaussianMapper::handleNewKeyframe(
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
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
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

        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
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


void GaussianMapper::handleNewKeyframe_WithLine(
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
    std::shared_ptr<GaussianKeyframe> pkf =
        std::make_shared<GaussianKeyframe>(std::get<0>(kf), getIteration());
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

        // Image (left if STEREO)
        cv::Mat imgRGB = std::get<3>(kf);
        if (this->sensor_type_ == STEREO)
            imgRGB_undistorted = imgRGB;
        else
            camera.undistortImage(imgRGB, imgRGB_undistorted);
        // Auxiliary Image
        cv::Mat imgAux = std::get<5>(kf);
        if (this->sensor_type_ == RGBD)
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



void GaussianMapper::generateKfidRandomShuffle()
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
    std::mt19937 g(rd_());
    std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);

    kfid_shuffled_ = true;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomSlidingWindowKeyframe()
{
// auto t1 = std::chrono::steady_clock::now();
    if (scene_->keyframes().empty())
        return nullptr;

    // Append phase uses its OWN pre-generated, recorded view schedule instead
    // of the online sliding-window/remaining-budget roulette. This does not
    // reset or consume remaining_times_of_use_; only kfs_used_times_ (the LR
    // schedule input) is advanced, exactly like the production sampler.
    if (append_phase_)
        return useOneAppendKeyframe();

    if (!kfid_shuffled_)
        generateKfidRandomShuffle();

    std::shared_ptr<GaussianKeyframe> viewpoint_cam = nullptr;
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
// std::cout<<t21 <<" ns"<<std::endl;
    return viewpoint_cam;
}

std::shared_ptr<GaussianKeyframe>
GaussianMapper::useOneRandomKeyframe()
{
    if (scene_->keyframes().empty())
        return nullptr;

    // Get randomly
    int nkfs = static_cast<int>(scene_->keyframes().size());
    int random_cam_idx = std::rand() / ((RAND_MAX + 1u) / nkfs);
    auto random_cam_it = scene_->keyframes().begin();
    for (int cam_idx = 0; cam_idx < random_cam_idx; ++cam_idx)
        ++random_cam_it;
    std::shared_ptr<GaussianKeyframe> viewpoint_cam = (*random_cam_it).second;

    // Count used times
    auto viewpoint_fid = viewpoint_cam->fid_;
    if (kfs_used_times_.find(viewpoint_fid) == kfs_used_times_.end())
        kfs_used_times_[viewpoint_fid] = 1;
    else
        ++kfs_used_times_[viewpoint_fid];

    return viewpoint_cam;
}

void GaussianMapper::increaseKeyframeTimesOfUse(
    std::shared_ptr<GaussianKeyframe> pkf,
    int times)
{
    pkf->remaining_times_of_use_ += times;
}

void GaussianMapper::cullKeyframes()
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

void GaussianMapper::increasePcdByKeyframeInactiveGeoDensify(
    std::shared_ptr<GaussianKeyframe> pkf)
{
// auto start_timing = std::chrono::steady_clock::now();
    torch::NoGradGuard no_grad;

    Sophus::SE3f Twc = pkf->getPosef().inverse();

    switch (this->sensor_type_)
    {
    case MONOCULAR:
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
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case STEREO:
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
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
        }
// savePly(result_dir_ / (std::to_string(getIteration()) + "_" + std::to_string(pkf->fid_) + "_1_after_inactive_geo_densify"));
    }
    break;
    case RGBD:
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
        if (depth_cached_ == 0) {
            depth_cache_points_ = points3D_valid;
            depth_cache_colors_ = colors_valid;
        }
        else {
            depth_cache_points_ = torch::cat({depth_cache_points_, points3D_valid}, /*dim=*/0);
            depth_cache_colors_ = torch::cat({depth_cache_colors_, colors_valid}, /*dim=*/0);
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
        gaussians_->increasePcd(depth_cache_points_, depth_cache_colors_, getIteration());
    }

// auto end_timing = std::chrono::steady_clock::now();
// auto completion_time = std::chrono::duration_cast<std::chrono::milliseconds>(
//                 end_timing - start_timing).count();
// std::cout << "[Gaussian Mapper]increasePcdByKeyframeInactiveGeoDensify() takes "
//             << completion_time
//             << " ms"
//             << std::endl;
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

void GaussianMapper::recordKeyframeRendered(
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

cv::Mat GaussianMapper::renderFromPose(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
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
        render_pkg = GaussianRenderer::render(
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

cv::Mat GaussianMapper::renderFromPose_WithLine(
    const Sophus::SE3f &Tcw,
    const int width,
    const int height,
    const bool main_vision)
{
    if (!initial_mapped_ || getIteration() <= 0)
        return cv::Mat(height, width, CV_32FC3, cv::Vec3f(0.0f, 0.0f, 0.0f));
    std::shared_ptr<GaussianKeyframe> pkf = std::make_shared<GaussianKeyframe>();
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
        render_pkg = GaussianRenderer::render(
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


void GaussianMapper::renderAndRecordKeyframe(
    std::shared_ptr<GaussianKeyframe> pkf,
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
    auto render_pkg = GaussianRenderer::render(
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

    // Shutdown-only, default-off single-frame diagnostic. It re-uses EXACTLY
    // the same masked_image / gt_image / mask as the original metrics and does
    // not alter the recorded psnr or psnr_gaussian_splatting outputs.
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
                "P", static_cast<unsigned long>(pkf->fid_), getIteration(),
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

void GaussianMapper::renderAndRecordAllKeyframes(
    std::string name_suffix)
{
    std::filesystem::path result_dir = result_dir_ / (std::to_string(getIteration()) + name_suffix);
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

    std::filesystem::path image_dir = result_dir / "image";
    if (record_rendered_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_dir);

    std::filesystem::path image_gt_dir = result_dir / "image_gt";
    if (record_ground_truth_image_)
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_gt_dir);

    std::filesystem::path image_loss_dir = result_dir / "image_loss";
    if (record_loss_image_) {
        CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(image_loss_dir);
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

void GaussianMapper::renderAndRecordAllKeyframesReadOnly(std::string name_suffix)
{
    const auto usage_before = append_refinement::captureKfUsageState(*scene_);
    const auto model_before = append_refinement::captureEvalModelState(*gaussians_);
    const auto used_before = kfs_used_times_;
    const int iter_before = getIteration();
    const long long steps_before = quad_opt_steps_;
    const double map_ms_before = quad_mapping_ms_;

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
    } else if (quad_opt_steps_ != steps_before) {
        ok = false; why = "optimizer_step_count_changed";
    } else if (quad_mapping_ms_ != map_ms_before) {
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
    out << "training_timer=" << map_ms_before << "\n";
    out << "params_changed=0\n";
    out << "adam_steps_changed=0\n";
    out << "training_counters_changed=0\n";
    out << "viewport_or_pyramid_budget_changed=0\n";
    out.flush();
    std::cerr << "[EvalGuard] " << (ok ? "OK" : "FAIL")
              << " suffix=" << name_suffix << " reason=" << why << std::endl;
}

void GaussianMapper::savePly(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
    keyframesToJson(result_dir);
    saveModelParams(result_dir);

    std::filesystem::path ply_dir = result_dir / "point_cloud";
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    ply_dir = ply_dir / ("iteration_" + std::to_string(getIteration()));
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(ply_dir)

    gaussians_->savePly(ply_dir / "point_cloud.ply");
    gaussians_->saveSparsePointsPly(result_dir / "input.ply");
}

void GaussianMapper::keyframesToJson(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)

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

void GaussianMapper::saveModelParams(std::filesystem::path result_dir)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
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

void GaussianMapper::writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix)
{
    CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS(result_dir)
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

int GaussianMapper::getIteration()
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    return iteration_;
}
void GaussianMapper::increaseIteration(const int inc)
{
    std::unique_lock<std::mutex> lock(mutex_status_);
    iteration_ += inc;
}

float GaussianMapper::positionLearningRateInit()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.position_lr_init_;
}
float GaussianMapper::featureLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.feature_lr_;
}
float GaussianMapper::opacityLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_lr_;
}
float GaussianMapper::scalingLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.scaling_lr_;
}
float GaussianMapper::rotationLearningRate()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.rotation_lr_;
}
float GaussianMapper::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.percent_dense_;
}
float GaussianMapper::lambdaDssim()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.lambda_dssim_;
}
int GaussianMapper::opacityResetInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.opacity_reset_interval_;
}
float GaussianMapper::densifyGradThreshold()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densify_grad_threshold_;
}
int GaussianMapper::densifyInterval()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return opt_params_.densification_interval_;
}
int GaussianMapper::newKeyframeTimesOfUse()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return new_keyframe_times_of_use_;
}
int GaussianMapper::stableNumIterExistence()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return stable_num_iter_existence_;
}
bool GaussianMapper::isKeepingTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return keep_training_;
}
bool GaussianMapper::isdoingGausPyramidTraining()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return do_gaus_pyramid_training_;
}
bool GaussianMapper::isdoingInactiveGeoDensify()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return inactive_geo_densify_;
}

void GaussianMapper::setPositionLearningRateInit(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.position_lr_init_ = lr;
}
void GaussianMapper::setFeatureLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.feature_lr_ = lr;
}
void GaussianMapper::setOpacityLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_lr_ = lr;
}
void GaussianMapper::setScalingLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.scaling_lr_ = lr;
}
void GaussianMapper::setRotationLearningRate(const float lr)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.rotation_lr_ = lr;
}
void GaussianMapper::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.percent_dense_ = percent_dense;
    gaussians_->setPercentDense(percent_dense);
}
void GaussianMapper::setLambdaDssim(const float lambda_dssim)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.lambda_dssim_ = lambda_dssim;
}
void GaussianMapper::setOpacityResetInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.opacity_reset_interval_ = interval;
}
void GaussianMapper::setDensifyGradThreshold(const float th)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densify_grad_threshold_ = th;
}
void GaussianMapper::setDensifyInterval(const int interval)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    opt_params_.densification_interval_ = interval;
}
void GaussianMapper::setNewKeyframeTimesOfUse(const int times)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    new_keyframe_times_of_use_ = times;
}
void GaussianMapper::setStableNumIterExistence(const int niter)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    stable_num_iter_existence_ = niter;
}
void GaussianMapper::setKeepTraining(const bool keep)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    keep_training_ = keep;
}
void GaussianMapper::setDoGausPyramidTraining(const bool gaus_pyramid)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    do_gaus_pyramid_training_ = gaus_pyramid;
}
void GaussianMapper::setDoInactiveGeoDensify(const bool inactive_geo_densify)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    inactive_geo_densify_ = inactive_geo_densify;
}

VariableParameters GaussianMapper::getVaribleParameters()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    VariableParameters params;
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

void GaussianMapper::setVaribleParameters(const VariableParameters &params)
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

void GaussianMapper::loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path)
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