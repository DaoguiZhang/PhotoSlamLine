/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by zdg 2025
 * as part of Photo-SLAM_Line
 * 
 */

#pragma once

#include <memory>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <unordered_map>

#include <torch/torch.h>
#include <c10/cuda/CUDACachingAllocator.h>

#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "third_party/simple-knn/spatial.h"
#include "third_party/tinyply/tinyply.h"
#include "types.h"
#include "point3d.h"
#include "line_3d.h"
#include "operate_points.h"
#include "general_utils.h"
#include "sh_utils.h"
#include "tensor_utils.h"
#include "gaussian_parameters_line.h"

//这里是线段的顶点是和其他点一起优化，还是单独优化（线段顶点有自己的优化目标）后续再做
#define GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE                        \
    this->Tensor_vec_xyz_ = {this->xyz_};                    \
    this->Tensor_vec_feature_dc_ = {this->features_dc_};     \
    this->Tensor_vec_feature_rest_ = {this->features_rest_}; \
    this->Tensor_vec_opacity_ = {this->opacity_};            \
    this->Tensor_vec_scaling_ = {this->scaling_};            \
    this->Tensor_vec_rotation_ = {this->rotation_};

#define GAUSSIAN_MODEL_LINE_INIT_TENSORS(device_type)                                             \
    this->xyz_ = torch::empty(0, torch::TensorOptions().device(device_type));                \
    this->features_dc_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->features_rest_ = torch::empty(0, torch::TensorOptions().device(device_type));      \
    this->scaling_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->rotation_ = torch::empty(0, torch::TensorOptions().device(device_type));           \
    this->opacity_ = torch::empty(0, torch::TensorOptions().device(device_type));            \
    this->max_radii2D_ = torch::empty(0, torch::TensorOptions().device(device_type));        \
    this->xyz_gradient_accum_ = torch::empty(0, torch::TensorOptions().device(device_type)); \
    this->denom_ = torch::empty(0, torch::TensorOptions().device(device_type));              \
    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

class GaussianModelLine
{
public:
    GaussianModelLine(const int sh_degree);
    GaussianModelLine(const GaussianModelParamsLine& model_params);

    torch::Tensor getScalingActivation();
    torch::Tensor getScaling();
    torch::Tensor getRotationActivation();
    torch::Tensor getXYZ();
    torch::Tensor getFeatures();
    torch::Tensor getOpacityActivation();
    torch::Tensor getCovarianceActivation(int scaling_modifier = 1);

    // along-line: 0.5~1.0
    // normal:     0.3~0.6
    static float minWorldScaleFromPixelFootprint(float z, float focal);
    static Eigen::Vector3f initLineSampleLogScale(float sample_step, float ref_depth_z,float ref_focal, float k_t = 0.7f,float k_n = 0.4f);
    static Eigen::Vector4f initQuatAlignXToDir(const Eigen::Vector3f& dir_unit);
    torch::Tensor computeLineCoherenceLoss(float lambda_line = 1.0f);

    void oneUpShDegree();
    void setShDegree(const int sh);

    void createFromPcd(
        std::map<point3D_id_t, Point3D> pcd,
        const float spatial_lr_scale);
    void increasePcd(const std::vector<Point3D>& new_points, const int iteration);
    void increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration);
    void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration);

    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));
    void scaledTransformationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_scaling);

    void scaledTransformVisiblePointsOfKeyframe(
        torch::Tensor& point_not_transformed_flags,
        torch::Tensor& diff_pose,
        torch::Tensor& kf_world_view_transform,
        torch::Tensor& kf_full_proj_transform,
        const int kf_creation_iter,
        const int stable_num_iter_existence,
        int& num_transformed,
        const float scale = 1.0f);

    void trainingSetup(const GaussianOptimizationParamsLine& training_args);
    float updateLearningRate(int step);
    void setPositionLearningRate(float position_lr);
    void setFeatureLearningRate(float feature_lr);
    void setOpacityLearningRate(float opacity_lr);
    void setScalingLearningRate(float scaling_lr);
    void setRotationLearningRate(float rot_lr);

    void resetOpacity();
    torch::Tensor replaceTensorToOptimizer(torch::Tensor& t, int tensor_idx);

    void prunePoints(torch::Tensor& mask);
    void prunePointsWithLineAwareness(torch::Tensor& mask);

    void pruneExceedinglyAnisotropic(float threshold);

    void densificationPostfix(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_exist_since_iter);
    void densificationPostfixWithLineAwareness(
        torch::Tensor& new_xyz,
        torch::Tensor& new_features_dc,
        torch::Tensor& new_features_rest,
        torch::Tensor& new_opacities,
        torch::Tensor& new_scaling,
        torch::Tensor& new_rotation,
        torch::Tensor& new_exist_since_iter,
        // -------- 新增：line-aware --------
        torch::Tensor& new_is_line,        // [N_new]
        torch::Tensor& new_line_dir_w      // [N_new,3]
    );

    torch::Tensor computeLineLevelPruneMaskGPU(
        const torch::Tensor& base_prune_mask,
        float dir_thresh_deg,
        float dist_thresh,
        float min_line_opacity_sum);

    void densifyAndSplit(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent,
        int N = 2);

    void densifyAndSplitWithLineAwareness(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent,
        int N);

    void densifyAndClone(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent);

    void densifyAndCloneWithLineAwareness(
        torch::Tensor& grads,
        float grad_threshold,
        float scene_extent);

    void densifyAndPrune(
        float max_grad,
        float min_opacity,
        float extent,
        int max_screen_size);

    void densifyAndPruneWithLineAwareness(
        float max_grad,
        float min_opacity,
        float extent,
        int max_screen_size);
    
    torch::Tensor computeLineLevelPruneMask(
        const torch::Tensor& base_prune_mask,
        float dir_thresh_deg,
        float dist_thresh,
        float min_line_opacity_sum
    );

    void addDensificationStats(
        torch::Tensor& viewspace_point_tensor,
        torch::Tensor& update_filter);
    
    torch::Tensor computeGroupedLineLoss(const std::unordered_multimap<line3D_id_t, point3D_id_t>& line_to_sample_ids,
        const std::map<point3D_id_t, int>& pnt_id_to_tensor_idx, // 预先建立点ID到Tensor行号的映射
        float lambda_coherence);

    torch::Tensor computeLineShapeConstraint(float lambda_ecc, float lambda_ori);

// void increasePointsIterationsOfExistence(const int i = 1);

    void loadPly(std::filesystem::path ply_path);
    void savePly(std::filesystem::path result_path);
    void saveSparsePointsPly(std::filesystem::path result_path);

    float percentDense();
    void setPercentDense(const float percent_dense);

protected:
    float exponLrFunc(int step);

public:
    torch::DeviceType device_type_;

    int active_sh_degree_;
    int max_sh_degree_;
    float max_aniso_threshold_;

    torch::Tensor xyz_;
    torch::Tensor is_line_;   // [P] bool, check whether point is line sampled or not
    // 可选：每个 line Gaussian 的方向（世界系）
    torch::Tensor line_dir_w_;  // N x 3, float
    torch::Tensor point_ids_; // [N] 类型为 kLong，存储每个点的唯一 ID （用于后续处理）
    torch::Tensor features_dc_;
    torch::Tensor features_rest_;
    torch::Tensor scaling_;
    torch::Tensor rotation_;
    torch::Tensor opacity_;
    torch::Tensor max_radii2D_;
    torch::Tensor xyz_gradient_accum_;
    torch::Tensor denom_;
    torch::Tensor exist_since_iter_;

    std::vector<torch::Tensor> Tensor_vec_xyz_,
                               Tensor_vec_feature_dc_,
                               Tensor_vec_feature_rest_,
                               Tensor_vec_opacity_,
                               Tensor_vec_scaling_ ,
                               Tensor_vec_rotation_;

    std::shared_ptr<torch::optim::Adam> optimizer_;
    float percent_dense_;
    float spatial_lr_scale_;

    torch::Tensor sparse_points_xyz_;
    torch::Tensor sparse_points_color_;

    //Debug
    // GaussianModelLine.h 用于统计：在一次 forward render 中，
    //统计: 1: 每个 Gaussian 累计 alpha; 2: 被命中的像素数量
    torch::Tensor debug_hit_count_;     // [N]
    torch::Tensor debug_alpha_accum_;   // [N]


    // Line-specific tensors
    //torch::Tensor line_features_; // TO DO: optimize line gaussian splatting(point in line)

protected:
    float lr_init_;
    float lr_final_;
    int lr_delay_steps_;
    float lr_delay_mult_;
    int max_steps_;

    std::mutex mutex_settings_;
};
