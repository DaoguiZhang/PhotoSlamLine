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
 * created BY zdg in 2025,
 * as part of Photo-SLAM.
 */

#pragma once

#include <string>
#include <filesystem>

//TO MODIFY: add another constructor or variables
class GaussianModelParamsLine
{
public:
    GaussianModelParamsLine(
        std::filesystem::path source_path = "",
        std::filesystem::path model_path = "",
        std::filesystem::path exec_path = "",
        int sh_degree = 3,
        std::string images = "images",
        float resolution = -1.0f,
        bool white_background = false,
        std::string data_device = "cuda",
        bool eval = false);

public:
    int sh_degree_;
    std::filesystem::path source_path_;
    std::filesystem::path model_path_;
    std::string images_;
    float resolution_;
    bool white_background_;
    std::string data_device_;
    bool eval_;
};

class GaussianPipelineParamsLine
{
public:
    GaussianPipelineParamsLine(
        bool convert_SHs = false,
        bool compute_cov3D = false);

public:
    bool convert_SHs_;
    bool compute_cov3D_;
};

class GaussianOptimizationParamsLine
{
public:
    GaussianOptimizationParamsLine(
        int iterations = 30'000,
        float position_lr_init = 0.00016f,
        float position_lr_final = 0.0000016f,
        float position_lr_delay_mult = 0.01f,
        int position_lr_max_steps = 30'000,
        float feature_lr = 0.0025f,
        float opacity_lr = 0.05f,
        float scaling_lr = 0.005f,
        float rotation_lr = 0.001f,
        float percent_dense = 0.01f,
        float lambda_dssim = 0.2f,
        int densification_interval = 100,
        int opacity_reset_interval = 3000,
        int densify_from_iter = 500,
        int densify_until_iter = 15'000,
        float densify_grad_threshold = 0.0002f);

public:
    int iterations_;
    float position_lr_init_;
    float position_lr_final_;
    float position_lr_delay_mult_;
    int position_lr_max_steps_;
    float feature_lr_;
    float opacity_lr_;
    float scaling_lr_;
    float rotation_lr_;
    float percent_dense_;
    float lambda_dssim_;
    int densification_interval_;
    int opacity_reset_interval_;
    int densify_from_iter_;
    int densify_until_iter_;
    float densify_grad_threshold_;
    //added by zdg()
    float weight_line_coherence_=0.0;
    float weight_line_shape_ori_=0.0;
    float weight_line_shape_ecc_=0.0;
};
