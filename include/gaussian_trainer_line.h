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
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#pragma once

#include <torch/torch.h>

#include <iomanip>
#include <random>
#include <chrono>
#include <memory>
#include <thread>
#include <mutex>
#include <vector>
#include <unordered_map>

#include <opencv2/opencv.hpp>

#include "ORB-SLAM3/include/System.h"

#include "loss_utils.h"
#include "gaussian_parameters_line.h"
#include "gaussian_model_line.h"
#include "gaussian_scene_line.h"
#include "gaussian_renderer_line.h"


class GaussianTrainerLine
{
public:
    GaussianTrainerLine();

    static void trainingOnce(
        std::shared_ptr<GaussianSceneLine> scene,
        std::shared_ptr<GaussianModelLine> gaussians,
        GaussianModelParamsLine& dataset,
        GaussianOptimizationParamsLine& opt,
        GaussianPipelineParamsLine& pipe,
        torch::DeviceType device_type = torch::kCUDA,
        std::vector<int> testing_iterations = {},
        std::vector<int> saving_iterations = {},
        std::vector<int> checkpoint_iterations = {}/*, checkpoint*/);

    static void trainingReport(
        int iteration,
        int num_iterations,
        torch::Tensor& Ll1,
        torch::Tensor& loss,
        float ema_loss_for_log,
        std::function<torch::Tensor(torch::Tensor&, torch::Tensor&)> l1_loss,
        int64_t elapsed_time,
        GaussianModelLine& gaussians,
        GaussianSceneLine& scene,
        GaussianPipelineParamsLine& pipe,
        torch::Tensor& background);

};
