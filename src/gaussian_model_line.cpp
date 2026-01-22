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

#include "include/gaussian_model_line.h"


// 由 CUDA 逻辑反推：保证 my_radius>=1 的最小 world scale 近似：scale >= z/(3f)
inline float GaussianModelLine::minWorldScaleFromPixelFootprint(float z, float focal)
{
    if (z <= 0.f || focal <= 0.f) return 0.f;
    return z / (3.0f * focal);
}

// 线采样点的 anisotropic scale 初始化（严格对齐 preprocessCUDA 的 my_radius 门槛）
// 返回的是 “log-space scale”，对应你模型里的 scaling_
inline Eigen::Vector3f GaussianModelLine::initLineSampleLogScale(
    float sample_step,
    float ref_depth_z,
    float ref_focal,
    float k_t = 0.7f,      // along-line: 0.5~1.0
    float k_n = 0.4f)      // normal:     0.3~0.6
{
    // 1) 沿线方向：由采样步长控制
    float scale_t = std::max(1e-6f, k_t * sample_step);

    // 2) 法向方向：至少保证 1px footprint（对齐 CUDA my_radius）
    float scale_n = std::max(1e-6f, k_n * sample_step);

    if (ref_depth_z > 0.f && ref_focal > 0.f) {
        float scale_n_min = minWorldScaleFromPixelFootprint(ref_depth_z, ref_focal);
        scale_n = std::max(scale_n, scale_n_min);
    }

    Eigen::Vector3f s(scale_t, scale_n, scale_n);
    return s.array().log().matrix();
}

// 初始化 rotation：让局部 X 轴对齐线方向（这样 anisotropic 的 scale_t 真正沿线）
// 返回 (w,x,y,z) 且与你 CUDA computeCov3D 的读取方式一致：rot.x=r(w), rot.y=x, rot.z=y, rot.w=z
inline Eigen::Vector4f GaussianModelLine::initQuatAlignXToDir(const Eigen::Vector3f& dir_unit)
{
    Eigen::Vector3f d = dir_unit;
    float n = d.norm();
    if (n < 1e-8f) d = Eigen::Vector3f::UnitX();
    else d /= n;

    // Eigen::Quaternionf::FromTwoVectors 把 UnitX 旋到 d
    Eigen::Quaternionf q = Eigen::Quaternionf::FromTwoVectors(Eigen::Vector3f::UnitX(), d);
    q.normalize();

    // CUDA computeCov3D: glm::vec4 rot; q.x=r, q.y=x, q.z=y, q.w=z
    Eigen::Vector4f out;
    out[0] = q.w();
    out[1] = q.x();
    out[2] = q.y();
    out[3] = q.z();
    return out;
}

torch::Tensor GaussianModelLine::computeLineCoherenceLoss(float lambda_line)
{
    using namespace torch::indexing;

    // 1. 筛选出属于线特征的高斯点
    auto line_mask = this->is_line_; // [N] bool
    if (!line_mask.any().item<bool>()) {
        return torch::zeros({}, torch::TensorOptions().device(device_type_));
    }

    auto xyz_l = this->xyz_.index({line_mask});           // [Nl, 3]
    auto dir_l = this->line_dir_w_.index({line_mask});    // [Nl, 3]

    // 2. 这里的关键挑战是如何区分“不同的线段”
    // 在 SLAM 场景下，通常每个点属于一个特定的 LineID。
    // 如果你没有存储 LineID，我们可以利用空间邻近性和方向一致性做一个简单的 Local Window 约束。
    // 这里演示一个局部重心约束逻辑：
    
    // 计算当前所有线点的重心 (或者你可以按 LineID 分组计算)
    // 假设我们简化处理：约束点与其方向向量的投影一致性
    
    // q_i = (P_i - P_ref) 
    // r_i = q_i - (q_i \cdot d_i) * d_i  (即垂直于方向向量的分量)
    
    // 为了简单且高效（无需分组），我们计算点 i 的位置与其方向向量的偏离
    // 这种 Loss 会抑制高斯点向垂直于线方向漂移
    
    // 我们需要一个参考点，这里简单使用滑动窗口或全局分组。
    // 如果你的 Point3D 结构里有 line_id，建议在这里 cat 一个 line_id_tensor 进来。
    
    // 简易版：惩罚每个线高斯点的 XYZ 变化在法向方向上的分量
    // 假设 xyz_0 是初始位置（detach状态）
    auto xyz_init = xyz_l.detach(); 
    auto diff = xyz_l - xyz_init; // [Nl, 3] 优化过程中的位移
    
    // 计算位移在法向上的投影: residual = diff - (diff \cdot dir) * dir
    auto dot = (diff * dir_l).sum(1, true); // [Nl, 1]
    auto projection = dot * dir_l;          // [Nl, 3]
    auto normal_component = diff - projection;
    
    torch::Tensor loss = normal_component.pow(2).sum();

    return lambda_line * loss;
}


GaussianModelLine::GaussianModelLine(const int sh_degree)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    this->max_sh_degree_ = sh_degree;

    // Device
    if (torch::cuda::is_available())
        this->device_type_ = torch::kCUDA;
    else
        this->device_type_ = torch::kCPU;

    GAUSSIAN_MODEL_LINE_INIT_TENSORS(this->device_type_)
}

GaussianModelLine::GaussianModelLine(const GaussianModelParamsLine &model_params)
    : active_sh_degree_(0), spatial_lr_scale_(0.0),
      lr_delay_steps_(0), lr_delay_mult_(1.0), max_steps_(1000000)
{
    this->max_sh_degree_ = model_params.sh_degree_;

    // Device
    if (model_params.data_device_ == "cuda")
        this->device_type_ = torch::kCUDA;
    else
        this->device_type_ = torch::kCPU;

    GAUSSIAN_MODEL_LINE_INIT_TENSORS(this->device_type_)
}

torch::Tensor GaussianModelLine::getScalingActivation()
{
    return torch::exp(this->scaling_);
}

torch::Tensor GaussianModelLine::getRotationActivation()
{
    return torch::nn::functional::normalize(this->rotation_);
}

torch::Tensor GaussianModelLine::getXYZ()
{
    return this->xyz_;
}

torch::Tensor GaussianModelLine::getFeatures()
{
    return torch::cat({this->features_dc_.clone(), this->features_rest_.clone()}, /*dim=*/1);
}

torch::Tensor GaussianModelLine::getOpacityActivation()
{
    return torch::sigmoid(this->opacity_);
}

torch::Tensor GaussianModelLine::getCovarianceActivation(int scaling_modifier)
{
    // build_rotation
    auto r = this->rotation_;
    auto R = general_utils::build_rotation(r);

    // build_scaling_rotation(scaling_modifier * scaling(Activation), rotation(_))
    auto s = scaling_modifier * this->getScalingActivation();
    auto L = torch::zeros({s.size(0), 3, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    L.select(1, 0).select(1, 0).copy_(s.index({torch::indexing::Slice(), 0}));
    L.select(1, 1).select(1, 1).copy_(s.index({torch::indexing::Slice(), 1}));
    L.select(1, 2).select(1, 2).copy_(s.index({torch::indexing::Slice(), 2}));
    L = R.matmul(L); // L = R @ L

    // build_covariance_from_scaling_rotation
    auto actual_covariance = L.matmul(L.transpose(1, 2));
    // strip_symmetric
    // strip_lowerdiag
    auto symm_uncertainty = torch::zeros({actual_covariance.size(0), 6}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    symm_uncertainty.select(1, 0).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 0}));
    symm_uncertainty.select(1, 1).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 1}));
    symm_uncertainty.select(1, 2).copy_(actual_covariance.index({torch::indexing::Slice(), 0, 2}));
    symm_uncertainty.select(1, 3).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 1}));
    symm_uncertainty.select(1, 4).copy_(actual_covariance.index({torch::indexing::Slice(), 1, 2}));
    symm_uncertainty.select(1, 5).copy_(actual_covariance.index({torch::indexing::Slice(), 2, 2}));

    return symm_uncertainty;
}

void GaussianModelLine::oneUpShDegree()
{
    if (this->active_sh_degree_ < this->max_sh_degree_)
        this->active_sh_degree_ += 1;
}

void GaussianModelLine::setShDegree(const int sh)
{
    this->active_sh_degree_ = (sh > this->max_sh_degree_ ? this->max_sh_degree_ : sh);
}

void GaussianModelLine::createFromPcd(
    std::map<point3D_id_t, Point3D> pcd,
    const float spatial_lr_scale)
{
    this->spatial_lr_scale_ = spatial_lr_scale;
    int num_points = static_cast<int>(pcd.size());
    torch::Tensor fused_point_cloud = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor color = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    is_line_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    line_dir_w_ = torch::zeros({num_points, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));  //Set line direction

    auto is_line_cpu = is_line_.cpu();
    auto line_acc = is_line_cpu.accessor<bool, 1>();    //这个函数作用
    auto pcd_it = pcd.begin();
    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
        auto& point = (*pcd_it).second;
        fused_point_cloud.index({point_idx, 0}) = point.xyz_(0);
        fused_point_cloud.index({point_idx, 1}) = point.xyz_(1);
        fused_point_cloud.index({point_idx, 2}) = point.xyz_(2);
        color.index({point_idx, 0}) = point.color_(0);
        color.index({point_idx, 1}) = point.color_(1);
        color.index({point_idx, 2}) = point.color_(2);
        line_acc[point_idx] = (point.source_ == PointSourceType::LINE_SAMPLED); //check it point is line sampled
        ++pcd_it;
    }

    torch::Tensor fused_color = sh_utils::RGB2SH(color);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {fused_color.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = fused_color;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    // std::cout << "[Gaussian Model]Number of points at initialization : " << fused_point_cloud.size(0) << std::endl;

    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;
    
    // ---- init scales: 分两类 ----
    // 先给普通点按原始策略算一个 scales_dist（log-space）
    torch::Tensor scales = torch::zeros({num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat));

    {
        // distCUDA2 需要 CUDA tensor
        torch::Tensor point_cloud_copy = fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor scales_dist = torch::log(torch::sqrt(dist2));          // [N]
        scales_dist = scales_dist.unsqueeze(1).repeat({1, 3});               // [N,3]
        scales.copy_(scales_dist);
    }

    // 再把 line-sampled 点覆盖成 anisotropic log-scale，并把 rotation 对齐线方向
    // 这里用 CPU 循环写回 scales/rots（不重），点数再大也没你训练耗时大
    // 注意：pts[i].sample_step_ / ref_depth_z_ / ref_focal_ 要尽量填对
    // 3) 在 CPU 上改 line-sampled 点的 scales/rots（避免 GPU 上逐点 index_put_ 慢）
    //    关键：先保存 cpu tensor，再 accessor
    torch::Tensor scales_cpu = scales.to(torch::kCPU).contiguous();
    torch::Tensor rots_cpu   = rots.to(torch::kCPU).contiguous();

    torch::Tensor line_dir_cpu = line_dir_w_.cpu().contiguous();
    auto dir_acc = line_dir_cpu.accessor<float, 2>();

    auto scales_acc = scales_cpu.accessor<float, 2>();
    auto rots_acc   = rots_cpu.accessor<float, 2>();
    pcd_it = pcd.begin();
    for (int i = 0; i < num_points; ++i) {
        const Point3D& p = (*pcd_it).second;
        //Set sampled points' line direction
        if (p.source_ == PointSourceType::MAP_POINT)
        {
            // Do something for MAP_POINT
             // 对 point gaussian：给个合法但无关的值
            dir_acc[i][0] = 1.f;
            dir_acc[i][1] = 0.f;
            dir_acc[i][2] = 0.f;
            continue;
        }
        
        Eigen::Vector3f d = p.line_dir_.normalized();
        dir_acc[i][0] = d.x();
        dir_acc[i][1] = d.y();
        dir_acc[i][2] = d.z();


        // --- scale ---
        const float step  = (p.sample_step_ > 0.f) ? p.sample_step_ : 0.05f; // fallback
        Eigen::Vector3f log_s = initLineSampleLogScale(step, p.ref_depth_z_, p.ref_focal_);
        scales_acc[i][0] = log_s[0];
        scales_acc[i][1] = log_s[1];
        scales_acc[i][2] = log_s[2];

        // --- rotation: align local X to line_dir ---
        Eigen::Vector4f q = initQuatAlignXToDir(p.line_dir_);
        rots_acc[i][0] = q[0];
        rots_acc[i][1] = q[1];
        rots_acc[i][2] = q[2];
        rots_acc[i][3] = q[3];
    }

    

    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    this->exist_since_iter_ = torch::zeros(
        {fused_point_cloud.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    this->xyz_ = fused_point_cloud.requires_grad_();
    this->features_dc_ = features.index({torch::indexing::Slice(),
                                         torch::indexing::Slice(),
                                         torch::indexing::Slice(0, 1)})
                             .transpose(1, 2)
                             .contiguous()
                             .requires_grad_();
    this->features_rest_ = features.index({torch::indexing::Slice(),
                                           torch::indexing::Slice(),
                                           torch::indexing::Slice(1, features.size(2))})
                               .transpose(1, 2)
                               .contiguous()
                               .requires_grad_();
    this->scaling_ = scales.requires_grad_();
    this->rotation_ = rots.requires_grad_();
    this->opacity_ = opacities.requires_grad_();

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    //debug
    this->debug_hit_count_ =
        torch::zeros({this->getXYZ().size(0)},
                 torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    this->debug_alpha_accum_ =
        torch::zeros({this->getXYZ().size(0)},
                 torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    this->max_radii2D_ = torch::zeros({this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
    this->is_line_ = is_line_cpu.to(device_type_);  //cp back to gpu
}

void GaussianModelLine::increasePcd(
    const std::vector<Point3D>& new_points,
    const int iteration)
{
    const int N = static_cast<int>(new_points.size());
    if (N == 0) return;

    // ------------------------------------------------------------------
    // 1) pack xyz / color on CPU (safe for accessor)
    // ------------------------------------------------------------------
    torch::Tensor xyz_cpu = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor col_cpu = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

    {
        auto xyz_acc = xyz_cpu.accessor<float, 2>();
        auto col_acc = col_cpu.accessor<float, 2>();
        for (int i = 0; i < N; ++i) {
            const auto& p = new_points[i];
            xyz_acc[i][0] = p.xyz_.x();
            xyz_acc[i][1] = p.xyz_.y();
            xyz_acc[i][2] = p.xyz_.z();
            col_acc[i][0] = p.color_.x();
            col_acc[i][1] = p.color_.y();
            col_acc[i][2] = p.color_.z();
        }
    }

    // Move to target device (CUDA/CPU)
    torch::Tensor new_xyz   = xyz_cpu.to(device_type_, /*non_blocking=*/false).contiguous();
    torch::Tensor new_color = col_cpu.to(device_type_, /*non_blocking=*/false).contiguous();

    torch::Tensor new_is_line =
        torch::zeros({new_xyz.size(0)},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    new_is_line.fill_(false);

    // ------------------------------------------------------------------
    // 2) SH features (same as createFromPcd)
    // ------------------------------------------------------------------
    torch::Tensor fused_color = sh_utils::RGB2SH(new_color);
    const int sh_dim = (max_sh_degree_ + 1) * (max_sh_degree_ + 1);

    torch::Tensor features = torch::zeros(
        {N, 3, sh_dim},
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;

    torch::Tensor new_features_dc =
        features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)})
                .transpose(1, 2).contiguous();
    torch::Tensor new_features_rest =
        features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, sh_dim)})
                .transpose(1, 2).contiguous();

    // ------------------------------------------------------------------
    // 3) scale & rotation init (aligned with createFromPcd)
    // ------------------------------------------------------------------
    // rotation: identity quat
    torch::Tensor new_rotation = torch::zeros({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    new_rotation.index({torch::indexing::Slice(), 0}) = 1.f;

    // baseline scale: dist-based isotropic (log-space)
    torch::Tensor new_scaling;
    {
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(new_xyz.clone()), 1e-7);
        torch::Tensor s = torch::log(torch::sqrt(dist2)).unsqueeze(1).repeat({1, 3}); // [N,3] log-scale
        new_scaling = s.contiguous();
    }

    // overwrite line-sampled on CPU (safe)
    torch::Tensor scaling_cpu  = new_scaling.to(torch::kCPU).contiguous();
    torch::Tensor rotation_cpu = new_rotation.to(torch::kCPU).contiguous();
    torch::Tensor new_is_line_cpu = new_is_line.to(torch::kCPU).contiguous();

    {
        auto s_acc = scaling_cpu.accessor<float, 2>();
        auto r_acc = rotation_cpu.accessor<float, 2>();
        auto new_is_line_acc = new_is_line_cpu.accessor<bool, 1>();

        for (int i = 0; i < N; ++i) {
            const auto& p = new_points[i];
            if (p.source_ != PointSourceType::LINE_SAMPLED) continue;

            const float step = (p.sample_step_ > 0.f) ? p.sample_step_ : 0.05f;

            // anisotropic log-scale (your function must output log-scale!)
            Eigen::Vector3f log_s = initLineSampleLogScale(step, p.ref_depth_z_, p.ref_focal_);
            s_acc[i][0] = log_s[0];
            s_acc[i][1] = log_s[1];
            s_acc[i][2] = log_s[2];

            // rotation align local X -> line_dir
            Eigen::Vector4f q = initQuatAlignXToDir(p.line_dir_);
            r_acc[i][0] = q[0];
            r_acc[i][1] = q[1];
            r_acc[i][2] = q[2];
            r_acc[i][3] = q[3];

            new_is_line_acc[i] = true;
        }
    }

    this->point_ids_ = torch::arange(0, num_points, 
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));

    new_scaling  = scaling_cpu.to(device_type_).contiguous();
    new_rotation = rotation_cpu.to(device_type_).contiguous();

    // ------------------------------------------------------------------
    // 4) opacity & existence
    // ------------------------------------------------------------------
    torch::Tensor new_opacity =
        general_utils::inverse_sigmoid(
            0.1f * torch::ones({N, 1},
                torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)));

    torch::Tensor new_exist_since_iter =
        torch::full({N}, iteration,
            torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    // ------------------------------------------------------------------
    // 5) densification postfix (unchanged)
    // ------------------------------------------------------------------
    densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
}


void GaussianModelLine::increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration)
{
// auto time1 = std::chrono::steady_clock::now();
    assert(points.size() == colors.size());
    assert(points.size() % 3 == 0);
    auto num_new_points = static_cast<int>(points.size() / 3);
    if (num_new_points == 0)
        return;

    torch::Tensor new_point_cloud = torch::from_blob(
        points.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        // torch::zeros({num_new_points, 3}, xyz_.options());
    torch::Tensor new_colors = torch::from_blob(
        colors.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
        // torch::zeros({num_new_points, 3}, xyz_.options());

    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
    }

    torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {new_fused_colors.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = new_fused_colors;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    // std::cout << "[Gaussian Model]Number of points increase : "
    //           << num_new_points << std::endl;

    torch::Tensor dist2 = torch::clamp_min(
        distCUDA2(new_point_cloud.clone()), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    torch::Tensor rots = torch::zeros(
        {new_point_cloud.size(0), 4},
         torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {new_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    torch::Tensor new_exist_since_iter = torch::full(
        {new_point_cloud.size(0)},
        iteration,
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    auto new_xyz = new_point_cloud;
    auto new_features_dc = features.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(),
                                                    torch::indexing::Slice(0, 1)})
                                        .transpose(1, 2)
                                        .contiguous();
    auto new_features_rest = features.index({torch::indexing::Slice(),
                                                      torch::indexing::Slice(),
                                                      torch::indexing::Slice(1, features.size(2))})
                                          .transpose(1, 2)
                                          .contiguous();
    auto new_opacities = opacities;
    auto new_scaling = scales;
    auto new_rotation = rots;

// auto time2 = std::chrono::steady_clock::now();
// auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
// std::cout << "increasePcd(umap) preparation time: " << time << " ms" <<std::endl;

    densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
// auto time3 = std::chrono::steady_clock::now();
// time = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
// std::cout << "increasePcd(umap) postfix time: " << time << " ms" <<std::endl;
}

void GaussianModelLine::increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration)
{
// auto time1 = std::chrono::steady_clock::now();
    auto num_new_points = new_point_cloud.size(0);
    if (num_new_points == 0)
        return;

    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
    }

    torch::Tensor new_fused_colors = sh_utils::RGB2SH(new_colors);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {new_fused_colors.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(0, 3),
         0}) = new_fused_colors;
    features.index(
        {torch::indexing::Slice(),
         torch::indexing::Slice(3, features.size(1)),
         torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    // std::cout << "[Gaussian Model]Number of points increase : "
    //           << num_new_points << std::endl;

    torch::Tensor dist2 = torch::clamp_min(
        distCUDA2(new_point_cloud.clone()), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    torch::Tensor rots = torch::zeros(
        {new_point_cloud.size(0), 4},
         torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;
    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {new_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    torch::Tensor new_exist_since_iter = torch::full(
        {new_point_cloud.size(0)},
        iteration,
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    auto new_xyz = new_point_cloud;
    auto new_features_dc = features.index({torch::indexing::Slice(),
                                                    torch::indexing::Slice(),
                                                    torch::indexing::Slice(0, 1)})
                                        .transpose(1, 2)
                                        .contiguous();
    auto new_features_rest = features.index({torch::indexing::Slice(),
                                                      torch::indexing::Slice(),
                                                      torch::indexing::Slice(1, features.size(2))})
                                          .transpose(1, 2)
                                          .contiguous();
    auto new_opacities = opacities;
    auto new_scaling = scales;
    auto new_rotation = rots;

// auto time2 = std::chrono::steady_clock::now();
// auto time = std::chrono::duration_cast<std::chrono::milliseconds>(time2-time1).count();
// std::cout << "increasePcd(tensor) preparation time: " << time << " ms" <<std::endl;

    densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );

    c10::cuda::CUDACachingAllocator::emptyCache();

// auto time3 = std::chrono::steady_clock::now();
// time = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
// std::cout << "increasePcd(tensor) postfix time: " << time << " ms" <<std::endl;
}

void GaussianModelLine::applyScaledTransformation(
    const float s,
    const Sophus::SE3f T)
{
    torch::NoGradGuard no_grad;
    // pt <- (s * Ryw * pt + tyw)
    this->xyz_ *= s;
    torch::Tensor T_tensor =
        tensor_utils::EigenMatrix2TorchTensor(T.matrix(), device_type_).transpose(0, 1);
    transformPoints(this->xyz_, T_tensor);
    // 在 applyScaledTransformation 中同步变换线方向(由于 line_dir_w_ 目前在你的代码里是 requires_grad(false)，如果线本身的位置发生了旋转，line_dir_w_ 就失效了)
    torch::Tensor R_tensor = T_tensor.slice(0, 0, 3).slice(1, 0, 3); // 提取旋转部分
    this->line_dir_w_ = torch::matmul(R_tensor, this->line_dir_w_.transpose(0,1)).transpose(0,1);

// torch::Tensor scales;
// torch::Tensor point_cloud_copy = this->xyz_.clone();
// torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
// scales = torch::log(torch::sqrt(dist2));
// auto scales_ndimension = scales.ndimension();
// scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    this->scaling_ *= s;
    scaledTransformationPostfix(this->xyz_, this->scaling_);
}

void GaussianModelLine::scaledTransformationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_scaling)
{
    // param_groups[0] = xyz_
    torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(new_xyz, 0);
    // param_groups[4] = scaling_
    torch::Tensor optimizable_scaling = this->replaceTensorToOptimizer(new_scaling, 4);

    this->xyz_ = optimizable_xyz;
    this->scaling_ = optimizable_scaling;

    this->Tensor_vec_xyz_ = {this->xyz_};
    this->Tensor_vec_scaling_ = {this->scaling_};
}

void GaussianModelLine::scaledTransformVisiblePointsOfKeyframe(
    torch::Tensor& point_not_transformed_flags,
    torch::Tensor& diff_pose,
    torch::Tensor& kf_world_view_transform,
    torch::Tensor& kf_full_proj_transform,
    const int kf_creation_iter,
    const int stable_num_iter_existence,
    int& num_transformed,
    const float scale)
{
    torch::NoGradGuard no_grad;

    torch::Tensor points = this->getXYZ();
    torch::Tensor rots = this->getRotationActivation();
    // torch::Tensor scales = this->scaling_;// * scale;

    torch::Tensor point_unstable_flags = torch::where(
        torch::abs(this->exist_since_iter_ - kf_creation_iter) < stable_num_iter_existence,
        true,
        false);

    scaleAndTransformThenMarkVisiblePoints(
        points,
        rots,
        point_not_transformed_flags,
        point_unstable_flags,
        diff_pose,
        kf_world_view_transform,
        kf_full_proj_transform,
        num_transformed,
        scale
    );

// torch::Tensor point_cloud_copy = points.clone();
// torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
// torch::Tensor scales = torch::log(torch::sqrt(dist2));
// auto scales_ndimension = scales.ndimension();
// scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});

    // Postfix
    // ==================================
    // param_groups[0] = xyz_
    // param_groups[1] = feature_dc_
    // param_groups[2] = feature_rest_
    // param_groups[3] = opacity_
    // param_groups[4] = scaling_
    // param_groups[5] = rotation_
    // ==================================
    torch::Tensor optimizable_xyz = this->replaceTensorToOptimizer(points, 0);
    // torch::Tensor optimizable_scaling = this->replaceTensorToOptimizer(scales, 4);
    torch::Tensor optimizable_rots = this->replaceTensorToOptimizer(rots, 5);

    this->xyz_ = optimizable_xyz;
    // this->scaling_ = optimizable_scaling;
    this->rotation_ = optimizable_rots;

    this->Tensor_vec_xyz_ = {this->xyz_};
    // this->Tensor_vec_scaling_ = {this->scaling_};
    this->Tensor_vec_rotation_ = {this->rotation_};
}

void GaussianModelLine::trainingSetup(const GaussianOptimizationParamsLine& training_args)
{
    setPercentDense(training_args.percent_dense_);
    this->xyz_gradient_accum_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
    this->denom_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));

    torch::optim::AdamOptions adam_options;
    adam_options.set_lr(0.0);
    adam_options.eps() = 1e-15;

    this->optimizer_.reset(new torch::optim::Adam(Tensor_vec_xyz_, adam_options));
    optimizer_->param_groups()[0].options().set_lr(training_args.position_lr_init_ * this->spatial_lr_scale_);

    optimizer_->add_param_group(Tensor_vec_feature_dc_);
    optimizer_->param_groups()[1].options().set_lr(training_args.feature_lr_);

    optimizer_->add_param_group(Tensor_vec_feature_rest_);
    optimizer_->param_groups()[2].options().set_lr(training_args.feature_lr_ / 20.0);

    optimizer_->add_param_group(Tensor_vec_opacity_);
    optimizer_->param_groups()[3].options().set_lr(training_args.opacity_lr_);

    optimizer_->add_param_group(Tensor_vec_scaling_);
    optimizer_->param_groups()[4].options().set_lr(training_args.scaling_lr_);

    optimizer_->add_param_group(Tensor_vec_rotation_);
    optimizer_->param_groups()[5].options().set_lr(training_args.rotation_lr_);

    // get_expon_lr_func
    lr_init_ = training_args.position_lr_init_ * this->spatial_lr_scale_;
    lr_final_ = training_args.position_lr_final_ * this->spatial_lr_scale_;
    lr_delay_mult_ = training_args.position_lr_delay_mult_;
    max_steps_ = training_args.position_lr_max_steps_;
}

float GaussianModelLine::updateLearningRate(int step)
{
    // def update_learning_rate(self, iteration):
    //     ''' Learning rate scheduling per step '''
    //     for param_group in self.optimizer.param_groups:
    //         if param_group["name"] == "xyz":
    //             lr = self.xyz_scheduler_args(iteration)
    //             param_group['lr'] = lr
    //             return lr
    float lr = this->exponLrFunc(step);
    optimizer_->param_groups()[0].options().set_lr(lr); // Tensor_vec_xyz_
    return lr;
}

// ==================================
// param_groups[0] = xyz_
// param_groups[1] = feature_dc_
// param_groups[2] = feature_rest_
// param_groups[3] = opacity_
// param_groups[4] = scaling_
// param_groups[5] = rotation_
// ==================================
void GaussianModelLine::setPositionLearningRate(float position_lr)
{
    optimizer_->param_groups()[0].options().set_lr(position_lr * this->spatial_lr_scale_);
}
void GaussianModelLine::setFeatureLearningRate(float feature_lr)
{
    optimizer_->param_groups()[1].options().set_lr(feature_lr);
    optimizer_->param_groups()[2].options().set_lr(feature_lr / 20.0);
}
void GaussianModelLine::setOpacityLearningRate(float opacity_lr)
{
    optimizer_->param_groups()[3].options().set_lr(opacity_lr);
}
void GaussianModelLine::setScalingLearningRate(float scaling_lr)
{
    optimizer_->param_groups()[4].options().set_lr(scaling_lr);
}
void GaussianModelLine::setRotationLearningRate(float rot_lr)
{
    optimizer_->param_groups()[5].options().set_lr(rot_lr);
}

void GaussianModelLine::resetOpacity()
{
    torch::Tensor opacities_new = general_utils::inverse_sigmoid(
        torch::min(
            this->getOpacityActivation(),
            torch::ones_like(this->getOpacityActivation() * 0.01)));
    torch::Tensor optimizable_tensors = this->replaceTensorToOptimizer(opacities_new, 3); // "opacity"
    this->opacity_ = optimizable_tensors;
    this->Tensor_vec_opacity_ = {this->opacity_};
}

torch::Tensor GaussianModelLine::replaceTensorToOptimizer(torch::Tensor& tensor, int tensor_idx)
{
    auto& param = this->optimizer_->param_groups()[tensor_idx].params()[0];
    auto& state = optimizer_->state();
    //auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
    auto key = param.unsafeGetTensorImpl();
    auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    new_state->step(stored_state.step());
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed only when options.amsgrad(true), which is false by default

    state.erase(key);
    param = tensor.requires_grad_();
    //key = c10::guts::to_string(param.unsafeGetTensorImpl());
    key = param.unsafeGetTensorImpl();
    state[key] = std::move(new_state);

    auto optimizable_tensors = param;
    return optimizable_tensors;
}

void GaussianModelLine::prunePoints(torch::Tensor& mask)
{
    auto valid_points_mask = ~mask;

    // _prune_optimizer
    std::vector<torch::Tensor> optimizable_tensors(6);
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
        auto& param = param_groups[group_idx].params()[0];
        //auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        auto key = param.unsafeGetTensorImpl();
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(stored_state.exp_avg().index({valid_points_mask}).clone());
            new_state->exp_avg_sq(stored_state.exp_avg_sq().index({valid_points_mask}).clone());
            // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed only when options.amsgrad(true), which is false by default

            state.erase(key);
            param = param.index({valid_points_mask}).requires_grad_();
            //key = c10::guts::to_string(param.unsafeGetTensorImpl());
            key = param.unsafeGetTensorImpl();
            state[key] = std::move(new_state);
            optimizable_tensors[group_idx] = param;
        }
        else {
            param = param.index({valid_points_mask}).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    // ==================================
    // param_groups[0] = xyz_
    // param_groups[1] = feature_dc_
    // param_groups[2] = feature_rest_
    // param_groups[3] = opacity_
    // param_groups[4] = scaling_
    // param_groups[5] = rotation_
    // ==================================
    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    this->exist_since_iter_ = this->exist_since_iter_.index({valid_points_mask});

    this->xyz_gradient_accum_ = this->xyz_gradient_accum_.index({valid_points_mask});

    this->denom_ = this->denom_.index({valid_points_mask});
    this->max_radii2D_ = this->max_radii2D_.index({valid_points_mask});
}

void GaussianModelLine::prunePointsWithLineAwareness(torch::Tensor& mask)
{
    using namespace torch::indexing;

    auto valid_points_mask = ~mask;

    // =====================================================
    // A. Optimizer-managed tensors (Adam state-aware pruning)
    // =====================================================
    std::vector<torch::Tensor> optimizable_tensors(6);
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();

    for (int group_idx = 0; group_idx < 6; ++group_idx)
    {
        auto& param = param_groups[group_idx].params()[0];
        auto key = param.unsafeGetTensorImpl();

        if (state.find(key) != state.end())
        {
            auto& stored_state =
                static_cast<torch::optim::AdamParamState&>(*state[key]);

            auto new_state =
                std::make_unique<torch::optim::AdamParamState>();

            new_state->step(stored_state.step());
            new_state->exp_avg(
                stored_state.exp_avg().index({valid_points_mask}).clone());
            new_state->exp_avg_sq(
                stored_state.exp_avg_sq().index({valid_points_mask}).clone());

            state.erase(key);

            param =
                param.index({valid_points_mask}).requires_grad_();

            key = param.unsafeGetTensorImpl();
            state[key] = std::move(new_state);

            optimizable_tensors[group_idx] = param;
        }
        else
        {
            param =
                param.index({valid_points_mask}).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    // =====================================================
    // B. Assign back optimizer-managed tensors
    // =====================================================
    this->xyz_           = optimizable_tensors[0];
    this->features_dc_   = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_       = optimizable_tensors[3];
    this->scaling_       = optimizable_tensors[4];
    this->rotation_      = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    // ---- 完善 point_ids_ 的维护 ----
    this->point_ids_ = this->point_ids_.index({valid_points_mask});

    // =====================================================
    // C. Structure-only & auxiliary tensors (NO optimizer)
    // =====================================================
    this->is_line_ =
        this->is_line_.index({valid_points_mask});

    this->line_dir_w_ =
        this->line_dir_w_.index({valid_points_mask});

    this->exist_since_iter_ =
        this->exist_since_iter_.index({valid_points_mask});

    this->xyz_gradient_accum_ =
        this->xyz_gradient_accum_.index({valid_points_mask});

    this->denom_ =
        this->denom_.index({valid_points_mask});

    this->max_radii2D_ =
        this->max_radii2D_.index({valid_points_mask});
}

void GaussianModelLine::densificationPostfixWithLineAwareness(
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
)
{
    // =====================================================
    // A. Optimizer-managed tensors (Adam needs to be extended)
    // =====================================================
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };

    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();

    for (int group_idx = 0; group_idx < 6; ++group_idx) {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);

        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];
        auto key = param.unsafeGetTensorImpl();

        if (state.find(key) != state.end()) {
            // ---- extend Adam state ----
            auto& stored_state =
                static_cast<torch::optim::AdamParamState&>(*state[key]);

            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(
                torch::cat({stored_state.exp_avg().clone(),
                            torch::zeros_like(extension_tensor)}, 0));
            new_state->exp_avg_sq(
                torch::cat({stored_state.exp_avg_sq().clone(),
                            torch::zeros_like(extension_tensor)}, 0));

            state.erase(key);

            param = torch::cat({param, extension_tensor}, 0).requires_grad_();
            key = param.unsafeGetTensorImpl();
            state[key] = std::move(new_state);

            optimizable_tensors[group_idx] = param;
        }
        else {
            // no state yet (should rarely happen)
            param = torch::cat({param, extension_tensor}, 0).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    // ---- assign back ----
    this->xyz_           = optimizable_tensors[0];
    this->features_dc_   = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_       = optimizable_tensors[3];
    this->scaling_       = optimizable_tensors[4];
    this->rotation_      = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    // ---- 完善 point_ids_ 的维护 ----
    // 1. 获取当前最大的 ID 值，以确保新生成的 ID 不重复
    int64_t max_id = 0;
    if (this->point_ids_.size(0) > 0) {
        max_id = this->point_ids_.max().item<int64_t>();
    }

    // 2. 为新生成的点分配新 ID
    int64_t num_new = new_xyz.size(0);
    torch::Tensor new_ids = torch::arange(max_id + 1, max_id + 1 + num_new, 
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));

    // 3. 拼接 ID Tensor
    this->point_ids_ = torch::cat({this->point_ids_, new_ids}, 0);

    // =====================================================
    // B. Structure-only tensors (NO optimizer state)
    // =====================================================
    this->is_line_ =
        torch::cat({this->is_line_, new_is_line.to(this->is_line_.device())}, 0);

    this->line_dir_w_ =
        torch::cat({this->line_dir_w_, new_line_dir_w.to(this->line_dir_w_.device())}, 0);

    this->exist_since_iter_ =
        torch::cat({this->exist_since_iter_, new_exist_since_iter}, 0);

    // =====================================================
    // C. Reset auxiliary buffers (size must match new total)
    // =====================================================
    const int N = this->getXYZ().size(0);

    this->xyz_gradient_accum_ =
        torch::zeros({N, 1}, torch::TensorOptions().device(device_type_));

    this->denom_ =
        torch::zeros({N, 1}, torch::TensorOptions().device(device_type_));

    this->max_radii2D_ =
        torch::zeros({N}, torch::TensorOptions().device(device_type_));
}


void GaussianModelLine::densificationPostfix(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_exist_since_iter)
{
    // cat_tensors_to_optimizer
    std::vector<torch::Tensor> optimizable_tensors(6);
    std::vector<torch::Tensor> tensors_dict = {
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation
    };
    auto& param_groups = this->optimizer_->param_groups();
    auto& state = this->optimizer_->state();
    for (int group_idx = 0; group_idx < 6; ++group_idx) {
        auto& group = param_groups[group_idx];
        assert(group.params().size() == 1);
        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];
        //auto key = c10::guts::to_string(param.unsafeGetTensorImpl());
        auto key = param.unsafeGetTensorImpl();
        if (state.find(key) != state.end()) {
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            new_state->step(stored_state.step());
            new_state->exp_avg(torch::cat({stored_state.exp_avg().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            new_state->exp_avg_sq(torch::cat({stored_state.exp_avg_sq().clone(), torch::zeros_like(extension_tensor)}, /*dim=*/0));
            // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone());  // needed only when options.amsgrad(true), which is false by default

            state.erase(key);
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            //key = c10::guts::to_string(param.unsafeGetTensorImpl());
            key = param.unsafeGetTensorImpl();
            state[key] = std::move(new_state);

            optimizable_tensors[group_idx] = param;
        }
        else {
            param = torch::cat({param, extension_tensor}, /*dim=*/0).requires_grad_();
            optimizable_tensors[group_idx] = param;
        }
    }

    // ==================================
    // param_groups[0] = xyz_
    // param_groups[1] = feature_dc_
    // param_groups[2] = feature_rest_
    // param_groups[3] = opacity_
    // param_groups[4] = scaling_
    // param_groups[5] = rotation_
    // ==================================
    this->xyz_ = optimizable_tensors[0];
    this->features_dc_ = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_ = optimizable_tensors[3];
    this->scaling_ = optimizable_tensors[4];
    this->rotation_ = optimizable_tensors[5];

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    this->exist_since_iter_ = torch::cat({this->exist_since_iter_, new_exist_since_iter}, /*dim=*/0);

    this->xyz_gradient_accum_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
    this->denom_ = torch::zeros({this->getXYZ().size(0), 1}, torch::TensorOptions().device(device_type_));
    this->max_radii2D_ = torch::zeros({this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
}


void GaussianModelLine::densifyAndSplitWithLineAwareness(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent,
    int N)
{
    using namespace torch::indexing;

    const int64_t n_init = this->getXYZ().size(0);
    if (n_init == 0) return;

    // =========================================================
    // 1. Gradient-based selection
    // =========================================================
    auto padded_grad = torch::zeros(
        {n_init},
        torch::TensorOptions().device(device_type_).dtype(grads.dtype())
    );

    // grads may be shorter than n_init (same as original GS impl)
    padded_grad.slice(0, 0, grads.size(0)).copy_(grads.squeeze());

    auto selected_mask = padded_grad >= grad_threshold;
    selected_mask = torch::logical_and(
        selected_mask,
        std::get<0>(torch::max(this->getScalingActivation(), 1))
            > percentDense() * scene_extent
    );

    const int64_t M = selected_mask.sum().item<int64_t>();
    if (M == 0) return;

    // =========================================================
    // 2. Gather parameters (selected subset)
    // =========================================================
    auto xyz_sel       = this->getXYZ().index({selected_mask});                 // (M,3)
    auto scale_act_sel = this->getScalingActivation().index({selected_mask});   // (M,3)
    auto rot_sel       = this->rotation_.index({selected_mask});                // (M,*)

    // Structure tensors for selected subset
    auto is_line_sel     = this->is_line_.index({selected_mask});               // (M)
    auto line_dir_w_sel  = this->line_dir_w_.index({selected_mask});            // (M,3)

    auto R = general_utils::build_rotation(rot_sel); // (M,3,3)

    // =========================================================
    // 3. Classify: point-like vs line-like (same rule as before)
    // =========================================================
    auto sorted = std::get<0>(scale_act_sel.sort(1, /*descending=*/true));
    auto sigma0 = sorted.index({Slice(), 0});
    auto sigma1 = sorted.index({Slice(), 1});

    const float line_ratio_thresh = 4.0f;
    auto geom_is_line = sigma0 / (sigma1 + 1e-6) > line_ratio_thresh;

    // Final line mask: (recommended) trust stored is_line_ AND geometry check
    // If you want purely geometric, replace with: auto is_line = geom_is_line;
    auto is_line  = torch::logical_and(is_line_sel, geom_is_line);
    auto is_point = ~is_line;

    // =========================================================
    // 4. Containers for newly generated Gaussians (+ structure)
    // =========================================================
    std::vector<torch::Tensor> new_xyz_list;
    std::vector<torch::Tensor> new_scaling_list;
    std::vector<torch::Tensor> new_rot_list;
    std::vector<torch::Tensor> new_feat_dc_list;
    std::vector<torch::Tensor> new_feat_rest_list;
    std::vector<torch::Tensor> new_opacity_list;
    std::vector<torch::Tensor> new_exist_iter_list;

    // line-aware
    std::vector<torch::Tensor> new_is_line_list;     // (K)
    std::vector<torch::Tensor> new_line_dir_w_list;  // (K,3)

    auto opts_f = torch::TensorOptions().device(device_type_).dtype(scale_act_sel.dtype());

    // =========================================================
    // 5. POINT-AWARE split (original behavior)
    // =========================================================
    if (is_point.any().item<bool>())
    {
        auto xyz_p   = xyz_sel.index({is_point});
        auto scale_p = scale_act_sel.index({is_point});
        auto R_p     = R.index({is_point});
        const int64_t Mp = xyz_p.size(0);

        auto stds = scale_p.repeat({N, 1});
        auto samples = at::normal(
            torch::zeros({Mp * N, 3}, opts_f),
            stds
        );

        auto new_xyz = torch::bmm(
            R_p.repeat({N,1,1}),
            samples.unsqueeze(-1)
        ).squeeze(-1) + xyz_p.repeat({N,1});

        auto new_scaling = torch::log(scale_p.repeat({N,1}) / (0.8f * float(N)));

        new_xyz_list.push_back(new_xyz);
        new_scaling_list.push_back(new_scaling);

        new_rot_list.push_back(rot_sel.index({is_point}).repeat({N,1}));
        new_feat_dc_list.push_back(this->features_dc_.index({selected_mask}).index({is_point}).repeat({N,1,1}));
        new_feat_rest_list.push_back(this->features_rest_.index({selected_mask}).index({is_point}).repeat({N,1,1}));
        new_opacity_list.push_back(this->opacity_.index({selected_mask}).index({is_point}).repeat({N,1}));
        new_exist_iter_list.push_back(this->exist_since_iter_.index({selected_mask}).index({is_point}).repeat({N}));

        // --- structure: point ---
        new_is_line_list.push_back(torch::zeros({Mp * N}, torch::TensorOptions().device(device_type_).dtype(torch::kBool)));
        new_line_dir_w_list.push_back(torch::zeros({Mp * N, 3}, opts_f));
    }

    // =========================================================
    // 6. LINE-AWARE split (1D densification along local x-axis)
    // =========================================================
    if (is_line.any().item<bool>())
    {
        const int axis = 0; // local x-axis = line direction (confirmed)

        auto xyz_l   = xyz_sel.index({is_line});
        auto scale_l = scale_act_sel.index({is_line});
        auto R_l     = R.index({is_line});

        auto rot_l   = rot_sel.index({is_line});
        auto dir_l_w = line_dir_w_sel.index({is_line}); // (Ml,3), should already be unit

        const int64_t Ml = xyz_l.size(0);

        auto sigma_par = scale_l.index({Slice(), axis}).unsqueeze(1);

        auto eps = at::normal(
            torch::zeros({Ml * N, 1}, opts_f),
            sigma_par.repeat({N,1})
        );

        auto local_offset = torch::zeros({Ml * N, 3}, opts_f);
        local_offset.index_put_({Slice(), axis}, eps.squeeze(1));

        auto new_xyz = torch::bmm(
            R_l.repeat({N,1,1}),
            local_offset.unsqueeze(-1)
        ).squeeze(-1) + xyz_l.repeat({N,1});

        auto new_scale_act = scale_l.repeat({N,1});
        new_scale_act.index_put_(
            {Slice(), axis},
            new_scale_act.index({Slice(), axis}) / (0.8f * float(N))
        );

        auto new_scaling = torch::log(new_scale_act);

        new_xyz_list.push_back(new_xyz);
        new_scaling_list.push_back(new_scaling);

        new_rot_list.push_back(rot_l.repeat({N,1}));
        new_feat_dc_list.push_back(this->features_dc_.index({selected_mask}).index({is_line}).repeat({N,1,1}));
        new_feat_rest_list.push_back(this->features_rest_.index({selected_mask}).index({is_line}).repeat({N,1,1}));
        new_opacity_list.push_back(this->opacity_.index({selected_mask}).index({is_line}).repeat({N,1}));
        new_exist_iter_list.push_back(this->exist_since_iter_.index({selected_mask}).index({is_line}).repeat({N}));

        // --- structure: line ---
        new_is_line_list.push_back(torch::ones({Ml * N}, torch::TensorOptions().device(device_type_).dtype(torch::kBool)));
        // repeat world direction (no change during split)
        auto new_dir = dir_l_w.repeat({N,1});
        // safety normalize
        new_dir = new_dir / (new_dir.norm(2, 1, true) + 1e-6);
        new_line_dir_w_list.push_back(new_dir);
    }

    // =========================================================
    // 7. Concatenate & append
    // =========================================================
    if (new_xyz_list.empty()) return; // safety

    auto new_xyz          = torch::cat(new_xyz_list, 0);
    auto new_scaling      = torch::cat(new_scaling_list, 0);
    auto new_rotation     = torch::cat(new_rot_list, 0);
    auto new_features_dc  = torch::cat(new_feat_dc_list, 0);
    auto new_features_rst = torch::cat(new_feat_rest_list, 0);
    auto new_opacity      = torch::cat(new_opacity_list, 0);
    auto new_exist_iter   = torch::cat(new_exist_iter_list, 0);

    auto new_is_line      = torch::cat(new_is_line_list, 0);
    auto new_line_dir_w   = torch::cat(new_line_dir_w_list, 0);

    // size checks (debug-friendly)
    TORCH_CHECK(new_xyz.size(0) == new_is_line.size(0), "new_xyz and new_is_line size mismatch");
    TORCH_CHECK(new_xyz.size(0) == new_line_dir_w.size(0), "new_xyz and new_line_dir_w size mismatch");
    TORCH_CHECK(new_line_dir_w.size(1) == 3, "new_line_dir_w must be [N,3]");

    this->densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rst,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_iter,
        new_is_line,
        new_line_dir_w
    );

    // =========================================================
    // 8. Prune original Gaussians
    // =========================================================
    auto prune_filter = torch::cat({
        selected_mask,
        torch::zeros({new_xyz.size(0)},
            torch::TensorOptions().device(device_type_).dtype(torch::kBool))
    });
    this->prunePointsWithLineAwareness(prune_filter);
}


void GaussianModelLine::densifyAndSplit(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent,
    int N)
{
    int n_init_points = this->getXYZ().size(0);
    // Extract points that satisfy the gradient condition
    auto padded_grad = torch::zeros({n_init_points}, torch::TensorOptions().device(device_type_));
    padded_grad.slice(/*dim=*/0L, /*start=*/0, /*end=*/grads.size(0)).copy_(grads.squeeze());
    auto selected_pts_mask = torch::where(padded_grad >= grad_threshold, true, false);
    selected_pts_mask = torch::logical_and(
        selected_pts_mask,
        std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) > percentDense() * scene_extent
    );

    auto stds = this->getScalingActivation().index({selected_pts_mask}).repeat({N, 1});
    auto means = torch::zeros({stds.size(0), 3}, torch::TensorOptions().device(device_type_));
    auto samples = at::normal(means, stds);
    auto r_masked = this->rotation_.index({selected_pts_mask});
    auto rots = general_utils::build_rotation(r_masked).repeat({N, 1, 1});
    auto new_xyz = torch::bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + this->getXYZ().index({selected_pts_mask}).repeat({N, 1});
    auto new_scaling = torch::log(this->getScalingActivation().index({selected_pts_mask}).repeat({N, 1}) / (0.8 * N)); // scaling_inverse_activation
    auto new_rotation = this->rotation_.index({selected_pts_mask}).repeat({N, 1});
    auto new_features_dc = this->features_dc_.index({selected_pts_mask}).repeat({N, 1, 1});
    auto new_features_rest = this->features_rest_.index({selected_pts_mask}).repeat({N, 1, 1});
    auto new_opacity = this->opacity_.index({selected_pts_mask}).repeat({N, 1});

    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask}).repeat({N});

    this->densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );

    auto prune_filter = torch::cat({
        selected_pts_mask,
        torch::zeros({(N * selected_pts_mask.sum()).item<int>()}, torch::TensorOptions().device(device_type_).dtype(torch::kBool))
    });
    this->prunePoints(prune_filter);
}


void GaussianModelLine::densifyAndCloneWithLineAwareness(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent)
{
    using namespace torch::indexing;

    // =========================================================
    // 1. Gradient-based selection
    // =========================================================
    auto grad_norm = torch::frobenius_norm(grads, /*dim=*/-1);
    auto selected_mask = grad_norm >= grad_threshold;

    selected_mask = torch::logical_and(
        selected_mask,
        std::get<0>(torch::max(this->getScalingActivation(), 1))
            <= percentDense() * scene_extent
    );

    const int64_t M = selected_mask.sum().item<int64_t>();
    if (M == 0) return;

    // =========================================================
    // 2. Gather parameters (selected subset)
    // =========================================================
    auto xyz_sel       = this->xyz_.index({selected_mask});                     // (M,3)
    auto scale_act_sel = this->getScalingActivation().index({selected_mask});   // (M,3)
    auto rot_sel       = this->rotation_.index({selected_mask});                // (M,*)

    // structure tensors (selected subset)
    auto is_line_sel    = this->is_line_.index({selected_mask});                // (M)
    auto line_dir_w_sel = this->line_dir_w_.index({selected_mask});             // (M,3)

    // =========================================================
    // 3. Classify: point-like vs line-like (geometry)
    // =========================================================
    auto sorted = std::get<0>(scale_act_sel.sort(1, /*descending=*/true));
    auto sigma0 = sorted.index({Slice(), 0});
    auto sigma1 = sorted.index({Slice(), 1});

    const float line_ratio_thresh = 4.0f;
    auto geom_is_line = sigma0 / (sigma1 + 1e-6) > line_ratio_thresh;

    // Final line mask: trust stored label AND geometry (recommended)
    // If you prefer pure geometry: auto is_line = geom_is_line;
    auto is_line  = torch::logical_and(is_line_sel, geom_is_line);
    auto is_point = ~is_line;

    // =========================================================
    // 4. Containers (data + structure)
    // =========================================================
    std::vector<torch::Tensor> xyz_list;
    std::vector<torch::Tensor> scale_list;
    std::vector<torch::Tensor> rot_list;
    std::vector<torch::Tensor> feat_dc_list;
    std::vector<torch::Tensor> feat_rest_list;
    std::vector<torch::Tensor> opacity_list;
    std::vector<torch::Tensor> exist_iter_list;

    // line-aware
    std::vector<torch::Tensor> new_is_line_list;      // (K)
    std::vector<torch::Tensor> new_line_dir_w_list;   // (K,3)

    auto opts_f = torch::TensorOptions().device(device_type_).dtype(scale_act_sel.dtype());

    // =========================================================
    // 5. POINT clone (exact copy)
    // =========================================================
    if (is_point.any().item<bool>())
    {
        auto xyz_p   = xyz_sel.index({is_point});
        const int64_t Mp = xyz_p.size(0);

        xyz_list.push_back(xyz_p);
        scale_list.push_back(this->scaling_.index({selected_mask}).index({is_point}));
        rot_list.push_back(rot_sel.index({is_point}));
        feat_dc_list.push_back(this->features_dc_.index({selected_mask}).index({is_point}));
        feat_rest_list.push_back(this->features_rest_.index({selected_mask}).index({is_point}));
        opacity_list.push_back(this->opacity_.index({selected_mask}).index({is_point}));
        exist_iter_list.push_back(this->exist_since_iter_.index({selected_mask}).index({is_point}));

        // structure for point clones
        new_is_line_list.push_back(
            torch::zeros({Mp}, torch::TensorOptions().device(device_type_).dtype(torch::kBool))
        );
        new_line_dir_w_list.push_back(
            torch::zeros({Mp, 3}, opts_f)
        );
    }

    // =========================================================
    // 6. LINE-aware clone
    //    - same scale / rotation
    //    - tiny perturbation along line direction (local x-axis)
    //    - keep line_dir_w unchanged
    // =========================================================
    if (is_line.any().item<bool>())
    {
        const int axis = 0; // line direction = local x-axis

        auto xyz_l   = xyz_sel.index({is_line});
        auto scale_l = scale_act_sel.index({is_line});
        auto rot_l   = rot_sel.index({is_line});

        auto dir_l_w = line_dir_w_sel.index({is_line}); // (Ml,3)

        const int64_t Ml = xyz_l.size(0);

        // tiny noise: 1% of sigma_parallel
        auto sigma_par = scale_l.index({Slice(), axis}).unsqueeze(1);
        auto eps = 0.01f * at::normal(
            torch::zeros({Ml, 1}, opts_f),
            sigma_par
        );

        auto local_offset = torch::zeros({Ml, 3}, opts_f);
        local_offset.index_put_({Slice(), axis}, eps.squeeze(1));

        auto R_l = general_utils::build_rotation(rot_l); // (Ml,3,3)

        auto xyz_new = torch::bmm(
            R_l,
            local_offset.unsqueeze(-1)
        ).squeeze(-1) + xyz_l;

        xyz_list.push_back(xyz_new);
        scale_list.push_back(this->scaling_.index({selected_mask}).index({is_line}));
        rot_list.push_back(rot_l);
        feat_dc_list.push_back(this->features_dc_.index({selected_mask}).index({is_line}));
        feat_rest_list.push_back(this->features_rest_.index({selected_mask}).index({is_line}));
        opacity_list.push_back(this->opacity_.index({selected_mask}).index({is_line}));
        exist_iter_list.push_back(this->exist_since_iter_.index({selected_mask}).index({is_line}));

        // structure for line clones
        new_is_line_list.push_back(
            torch::ones({Ml}, torch::TensorOptions().device(device_type_).dtype(torch::kBool))
        );

        // keep same world direction; normalize for safety
        auto new_dir = dir_l_w / (dir_l_w.norm(2, 1, true) + 1e-6);
        new_line_dir_w_list.push_back(new_dir);
    }

    // =========================================================
    // 7. Concatenate & append
    // =========================================================
    if (xyz_list.empty()) return; // safety

    auto new_xyz          = torch::cat(xyz_list, 0);
    auto new_scaling      = torch::cat(scale_list, 0);
    auto new_rotation     = torch::cat(rot_list, 0);
    auto new_features_dc  = torch::cat(feat_dc_list, 0);
    auto new_features_rst = torch::cat(feat_rest_list, 0);
    auto new_opacity      = torch::cat(opacity_list, 0);
    auto new_exist_iter   = torch::cat(exist_iter_list, 0);

    auto new_is_line      = torch::cat(new_is_line_list, 0);
    auto new_line_dir_w   = torch::cat(new_line_dir_w_list, 0);

    TORCH_CHECK(new_xyz.size(0) == new_is_line.size(0), "new_xyz and new_is_line size mismatch");
    TORCH_CHECK(new_xyz.size(0) == new_line_dir_w.size(0), "new_xyz and new_line_dir_w size mismatch");
    TORCH_CHECK(new_line_dir_w.size(1) == 3, "new_line_dir_w must be [N,3]");

    this->densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rst,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_iter,
        new_is_line,
        new_line_dir_w
    );
}


void GaussianModelLine::densifyAndClone(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent)
{
    // Extract points that satisfy the gradient condition
    auto selected_pts_mask = torch::where(torch::frobenius_norm(grads, /*dim=*/-1) >= grad_threshold, true, false);
    selected_pts_mask = torch::logical_and(
        selected_pts_mask,
        std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) <= percentDense() * scene_extent
    );

    auto new_xyz = this->xyz_.index({selected_pts_mask});
    auto new_features_dc = this->features_dc_.index({selected_pts_mask});
    auto new_features_rest = this->features_rest_.index({selected_pts_mask});
    auto new_opacities = this->opacity_.index({selected_pts_mask});
    auto new_scaling = this->scaling_.index({selected_pts_mask});
    auto new_rotation = this->rotation_.index({selected_pts_mask});

    auto new_exist_since_iter = this->exist_since_iter_.index({selected_pts_mask});

    this->densificationPostfix(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter
    );
}

void GaussianModelLine::densifyAndPrune(
    float max_grad,
    float min_opacity,
    float extent,
    int max_screen_size)
{
    auto grads = this->xyz_gradient_accum_ / this->denom_;
    grads.index_put_({grads.isnan()}, 0.0f);
    this->densifyAndClone(grads, max_grad, extent);
    this->densifyAndSplit(grads, max_grad, extent);

    auto prune_mask = (this->getOpacityActivation() < min_opacity).squeeze();
    if (max_screen_size) {
        auto big_points_vs = this->max_radii2D_ > max_screen_size;
        auto big_points_ws = std::get<0>(this->getScalingActivation().max(/*dim=*/1)) > 0.1f * extent;
        prune_mask = torch::logical_or(torch::logical_or(prune_mask, big_points_vs), big_points_ws);
    }
    this->prunePoints(prune_mask);

    c10::cuda::CUDACachingAllocator::emptyCache(); // torch.cuda.empty_cache()
}

void GaussianModelLine::densifyAndPruneWithLineAwareness(
    float max_grad,
    float min_opacity,
    float extent,
    int max_screen_size)
{
    using namespace torch::indexing;

    // =========================================================
    // 1. Accumulated gradient (photometric-driven)
    // =========================================================
    auto grads = this->xyz_gradient_accum_ / this->denom_;
    grads.index_put_({grads.isnan()}, 0.0f);
    grads.index_put_({grads.isinf()}, 0.0f);

    // =========================================================
    // 2. Structure-aware densification
    // =========================================================
    this->densifyAndCloneWithLineAwareness(grads, max_grad, extent);

    const int split_N = 2; // recommended
    this->densifyAndSplitWithLineAwareness(grads, max_grad, extent, split_N);

    // =========================================================
    // 3. Base pruning criteria (opacity-driven)
    // =========================================================
    auto prune_mask =
        (this->getOpacityActivation() < min_opacity).view({-1});

    // =========================================================
    // 4. Screen-space and world-space pruning
    // =========================================================
    if (max_screen_size > 0)
    {
        auto big_points_vs =
            this->max_radii2D_ > max_screen_size;

        auto big_points_ws =
            std::get<0>(this->getScalingActivation().max(1))
                > 0.1f * extent;

        prune_mask = torch::logical_or(
            prune_mask,
            torch::logical_or(big_points_vs, big_points_ws)
        );
    }

    // =========================================================
    // 5. Prune (line-aware state-safe)
    // =========================================================
    this->prunePointsWithLineAwareness(prune_mask);

    // =========================================================
    // 6. Free cached CUDA memory
    // =========================================================
    c10::cuda::CUDACachingAllocator::emptyCache();
}


//
torch::Tensor GaussianModelLine::computeLineLevelPruneMask(
    const torch::Tensor& base_prune_mask,
    float dir_thresh_deg,
    float dist_thresh,
    float min_line_opacity_sum)
{
    using namespace torch::indexing;

    const int64_t N = this->xyz_.size(0);
    auto prune_mask = base_prune_mask.clone();

    // ---------------------------------------------------------
    // 1. Identify line-like Gaussians
    // ---------------------------------------------------------
    auto scale = this->getScalingActivation(); // (N,3)
    auto sorted = std::get<0>(scale.sort(1, /*descending=*/true));
    auto sigma0 = sorted.index({Slice(), 0});
    auto sigma1 = sorted.index({Slice(), 1});

    const float line_ratio_thresh = 4.0f;
    auto is_line = sigma0 / (sigma1 + 1e-6) > line_ratio_thresh;

    if (!is_line.any().item<bool>())
        return prune_mask;

    // ---------------------------------------------------------
    // 2. Extract line Gaussians
    // ---------------------------------------------------------
    auto idx_line = torch::nonzero(is_line).squeeze();
    auto xyz_l    = this->xyz_.index({idx_line});
    auto opacity  = this->getOpacityActivation().index({idx_line});

    // local x-axis → world direction
    auto rot_l = this->rotation_.index({idx_line});
    auto R_l = general_utils::build_rotation(rot_l);
    auto dir_l = R_l.index({Slice(), Slice(), 0}); // (Nl,3)

    // normalize
    dir_l = dir_l / (dir_l.norm(2, 1, true) + 1e-6);

    const int64_t Nl = xyz_l.size(0);

    // ---------------------------------------------------------
    // 3. Build adjacency by direction + distance
    // ---------------------------------------------------------
    const float cos_thresh =
        std::cos(dir_thresh_deg * M_PI / 180.0f);

    auto visited = torch::zeros({Nl}, torch::kBool);
    auto keep_line = torch::zeros({Nl}, torch::kBool);

    for (int64_t i = 0; i < Nl; ++i)
    {
        if (visited[i].item<bool>())
            continue;

        // BFS / flood fill
        std::vector<int64_t> stack;
        stack.push_back(i);
        visited[i] = true;

        float opacity_sum = 0.0f;

        while (!stack.empty())
        {
            int64_t u = stack.back();
            stack.pop_back();

            opacity_sum += opacity[u].item<float>();

            auto du = dir_l[u];
            auto pu = xyz_l[u];

            for (int64_t v = 0; v < Nl; ++v)
            {
                if (visited[v].item<bool>())
                    continue;

                // direction consistency
                float cos_uv = torch::dot(du, dir_l[v]).item<float>();
                if (std::abs(cos_uv) < cos_thresh)
                    continue;

                // spatial proximity
                float dist = torch::norm(pu - xyz_l[v]).item<float>();
                if (dist > dist_thresh)
                    continue;

                visited[v] = true;
                stack.push_back(v);
            }
        }

        // -----------------------------------------------------
        // 4. Line-level decision
        // -----------------------------------------------------
        if (opacity_sum >= min_line_opacity_sum)
        {
            // mark all visited in this component as keep
            for (int64_t v = 0; v < Nl; ++v)
                if (visited[v].item<bool>())
                    keep_line[v] = true;
        }
    }

    // ---------------------------------------------------------
    // 5. Override prune mask
    // ---------------------------------------------------------
    auto keep_global_idx = idx_line.index({keep_line});
    prune_mask.index_put_({keep_global_idx}, false);

    return prune_mask;
}


torch::Tensor GaussianModelLine::computeLineLevelPruneMaskGPU(
    const torch::Tensor& base_prune_mask,
    float dir_thresh_deg,
    float dist_thresh,
    float min_line_opacity_sum)
{
    using namespace torch::indexing;

    auto prune_mask = base_prune_mask.clone();

    const int64_t N = this->xyz_.size(0);

    // ---------------------------------------------------------
    // 1. Identify line-like Gaussians
    // ---------------------------------------------------------
    auto scale = this->getScalingActivation(); // (N,3)
    auto sorted = std::get<0>(scale.sort(1, /*descending=*/true));
    auto sigma0 = sorted.index({Slice(), 0});
    auto sigma1 = sorted.index({Slice(), 1});

    const float line_ratio_thresh = 4.0f;
    auto is_line = sigma0 / (sigma1 + 1e-6) > line_ratio_thresh;

    if (!is_line.any().item<bool>())
        return prune_mask;

    // ---------------------------------------------------------
    // 2. Extract line Gaussians
    // ---------------------------------------------------------
    auto idx_line = torch::nonzero(is_line).squeeze(1);   // (Nl)
    auto xyz_l    = this->xyz_.index({idx_line});         // (Nl,3)
    auto opacity  = this->getOpacityActivation().index({idx_line}); // (Nl)

    // rotation → direction (local x-axis)
    auto rot_l = this->rotation_.index({idx_line});
    auto R_l   = general_utils::build_rotation(rot_l);
    auto dir_l = R_l.index({Slice(), Slice(), 0}); // (Nl,3)

    // normalize directions
    dir_l = dir_l / (dir_l.norm(2, 1, true) + 1e-6);

    const int64_t Nl = xyz_l.size(0);

    // ---------------------------------------------------------
    // 3. Pairwise direction consistency
    // ---------------------------------------------------------
    // cos_ij = |d_i · d_j|
    auto cos_ij = torch::matmul(dir_l, dir_l.transpose(0,1)).abs(); // (Nl,Nl)

    const float cos_thresh =
        std::cos(dir_thresh_deg * M_PI / 180.0f);

    auto dir_ok = cos_ij >= cos_thresh; // (Nl,Nl)

    // ---------------------------------------------------------
    // 4. Pairwise distance consistency
    // ---------------------------------------------------------
    auto diff = xyz_l.unsqueeze(1) - xyz_l.unsqueeze(0); // (Nl,Nl,3)
    auto dist = diff.norm(2, /*dim=*/2);                 // (Nl,Nl)

    auto dist_ok = dist <= dist_thresh;

    // ---------------------------------------------------------
    // 5. Line affinity mask
    // ---------------------------------------------------------
    auto affinity = torch::logical_and(dir_ok, dist_ok); // (Nl,Nl)

    // ---------------------------------------------------------
    // 6. Line-level opacity aggregation
    // ---------------------------------------------------------
    // support_i = sum_j affinity_ij * opacity_j
    auto support =
        torch::matmul(
            affinity.to(opacity.dtype()),
            opacity.unsqueeze(1)
        ).squeeze(1); // (Nl)

    // ---------------------------------------------------------
    // 7. Keep decision (line-level)
    // ---------------------------------------------------------
    auto keep_line = support >= min_line_opacity_sum; // (Nl)

    // ---------------------------------------------------------
    // 8. Override prune mask
    // ---------------------------------------------------------
    auto keep_global_idx = idx_line.index({keep_line});
    prune_mask.index_put_({keep_global_idx}, false);

    return prune_mask;
}



void GaussianModelLine::addDensificationStats(
    torch::Tensor& viewspace_point_tensor,
    torch::Tensor& update_filter)
{
    this->xyz_gradient_accum_.index_put_(
        {update_filter},
        torch::frobenius_norm(viewspace_point_tensor.grad().index({update_filter, torch::indexing::Slice(0, 2)}),
                              /*dim=*/-1,
                              /*keepdim=*/true),
        /*accumulate=*/true);

    this->denom_.index_put_(
        {update_filter},
        this->denom_.index({update_filter}) + 1);
}

torch::Tensor GaussianModelLine::computeGroupedLineLoss(
    const std::unordered_multimap<line3D_id_t, point3D_id_t>& line_to_sample_ids,
    const std::map<point3D_id_t, int>& pnt_id_to_tensor_idx, // 预先建立点ID到Tensor行号的映射
    float lambda_coherence) 
{
    torch::Tensor total_loss = torch::zeros({}, torch::TensorOptions().device(device_type_));
    int count = 0;

    // 遍历每一条线段
    // 注意：在实际训练循环中，为了性能，建议只对当前帧可见的线段进行计算
    for (auto it = line_to_sample_ids.begin(); it != line_to_sample_ids.end(); ) {
        line3D_id_t curr_line_id = it->first;
        auto range = line_to_sample_ids.equal_range(curr_line_id);

        std::vector<int> indices;
        for (auto r = range.first; r != range.second; ++r) {
            if (pnt_id_to_tensor_idx.count(r->second)) {
                indices.push_back(pnt_id_to_tensor_idx.at(r->second));
            }
        }

        if (indices.size() > 1) {
            // 将属于同一条线的 Gaussian 索引转为 Tensor
            auto idx_tensor = torch::tensor(indices, torch::kLong).to(device_type_);
            
            // 提取这些点的当前位置 [M, 3] 和 初始方向 [M, 3]
            auto p_m = this->xyz_.index({idx_tensor});
            auto d_m = this->line_dir_w_.index({idx_tensor}); // 假设已归一化

            // 计算该组点的几何中心 (作为直线上的一点参考)
            auto center = p_m.mean(0, true); // [1, 3]

            // 计算每个点到直线的距离向量: r = (P - C) - ((P - C) · d) * d
            auto rel_p = p_m - center; // [M, 3]
            auto proj = torch::sum(rel_p * d_m, 1, true) * d_m; // [M, 3]
            auto orthogonal_dist = rel_p - proj; 

            total_loss = total_loss + orthogonal_dist.pow(2).sum();
            count++;
        }
        it = range.second; // 跳到下一条线
    }

    return (count > 0) ? (total_loss * lambda_coherence) : total_loss;
}

torch::Tensor GaussianModelLine::computeLineShapeConstraint(float lambda_ecc, float lambda_ori) {
    auto line_mask = this->is_line_;
    if (!line_mask.any().item<bool>()) return torch::zeros({}, xyz_.options());

    // 1. 提取 Scaling
    // 注意：scaling_ 存储的是 log 空间的值
    auto s = torch::exp(this->scaling_.index({line_mask})); // [N_line, 3]
    auto s_parallel = s.index({torch::indexing::Slice(), 0}); // 沿线方向
    auto s_perpendicular = s.index({torch::indexing::Slice(), torch::indexing::Slice(1, 3)}); // 法向方向

    // 约束 1：各向异性 (希望 s_perp / s_para 趋近于 0)
    // 我们设定一个目标比例，比如希望长宽比至少是 10:1
    auto eccentricity_loss = (s_perpendicular.mean(1) / (s_parallel + 1e-6)).sum();

    // 2. 提取 Rotation 并与初始方向对齐
    // 假设我们在初始化时保存了初始旋转 rotation_init_
    auto q_current = torch::nn::functional::normalize(this->rotation_.index({line_mask}));
    
    // 如果你没有保存初始 q，可以约束当前局部 X 轴与 line_dir_w_ 的夹角
    auto R = general_utils::build_rotation(q_current); // [N_line, 3, 3]
    auto local_x = R.index({torch::indexing::Slice(), torch::indexing::Slice(), 0}); // 变换后的局部 X 轴
    auto d_init = this->line_dir_w_.index({line_mask}); // 初始线方向

    // 约束 2：方向对齐 (余弦相似度接近 1)
    auto cos_sim = torch::abs(torch::sum(local_x * d_init, 1));
    auto orientation_loss = (1.0 - cos_sim).sum();

    return lambda_ecc * eccentricity_loss + lambda_ori * orientation_loss;
}

// void GaussianModel::increasePointsIterationsOfExistence(const int i)
// {
//     this->exist_since_iter_ += i;
// }

void GaussianModelLine::loadPly(std::filesystem::path ply_path)
{
    std::ifstream instream_binary(ply_path, std::ios::binary);
    if (!instream_binary.is_open() || instream_binary.fail())
        throw std::runtime_error("Fail to open ply file at " + ply_path.string());
    instream_binary.seekg(0, std::ios::beg);

    tinyply::PlyFile ply_file;
    ply_file.parse_header(instream_binary);

    std::cout << "\t[ply_header] Type: " << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto & c : ply_file.get_comments())
        std::cout << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto & c : ply_file.get_info())
        std::cout << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : ply_file.get_elements()) {
        std::cout << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        for (const auto &p : e.properties) {
            std::cout << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            if (p.isList)
                std::cout << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            std::cout << std::endl;
        }
    }

    std::shared_ptr<tinyply::PlyData> xyz, f_dc, f_rest, opacity, scales, rot;

    try { xyz = ply_file.request_properties_from_element("vertex", { "x", "y", "z" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { f_dc = ply_file.request_properties_from_element("vertex", { "f_dc_0", "f_dc_1", "f_dc_2" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    int n_f_rest = ((max_sh_degree_ + 1) * (max_sh_degree_ + 1) - 1) * 3;
    if (n_f_rest >= 0) {
        std::vector<std::string> f_rest_element_names(n_f_rest);
        for (int i = 0; i < n_f_rest; ++i)
            f_rest_element_names[i] = "f_rest_" + std::to_string(i);
        try {f_rest = ply_file.request_properties_from_element("vertex", f_rest_element_names); }
        catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }
    }

    try { opacity = ply_file.request_properties_from_element("vertex", { "opacity" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { scales = ply_file.request_properties_from_element("vertex", { "scale_0", "scale_1", "scale_2" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    try { rot = ply_file.request_properties_from_element("vertex", { "rot_0", "rot_1", "rot_2", "rot_3" }); }
    catch (const std::exception & e) { std::cerr << "tinyply exception: " << e.what() << std::endl; }

    ply_file.read(instream_binary);

    if (xyz)     std::cout << "\tRead " << xyz->count     << " total xyz "     << std::endl;
    if (f_dc)    std::cout << "\tRead " << f_dc->count    << " total f_dc "    << std::endl;
    if (f_rest)  std::cout << "\tRead " << f_rest->count  << " total f_rest "  << std::endl;
    if (opacity) std::cout << "\tRead " << opacity->count << " total opacity " << std::endl;
    if (scales)  std::cout << "\tRead " << scales->count  << " total scales "  << std::endl;
    if (rot)     std::cout << "\tRead " << rot->count     << " total rot "     << std::endl;

    // Data to std::vector
    const int num_points = xyz->count;

    const std::size_t n_xyz_bytes = xyz->buffer.size_bytes();
    std::vector<float> xyz_vector(xyz->count * 3);
    std::memcpy(xyz_vector.data(), xyz->buffer.get(), n_xyz_bytes);

    const std::size_t n_f_dc_bytes = f_dc->buffer.size_bytes();
    std::vector<float> f_dc_vector(f_dc->count * 3);
    std::memcpy(f_dc_vector.data(), f_dc->buffer.get(), n_f_dc_bytes);

    const std::size_t n_f_rest_bytes = f_rest->buffer.size_bytes();
    std::vector<float> f_rest_vector(f_rest->count * n_f_rest);
    std::memcpy(f_rest_vector.data(), f_rest->buffer.get(), n_f_rest_bytes);

    const std::size_t n_opacity_bytes = opacity->buffer.size_bytes();
    std::vector<float> opacity_vector(opacity->count * 1);
    std::memcpy(opacity_vector.data(), opacity->buffer.get(), n_opacity_bytes);

    const std::size_t n_scales_bytes = scales->buffer.size_bytes();
    std::vector<float> scales_vector(scales->count * 3);
    std::memcpy(scales_vector.data(), scales->buffer.get(), n_scales_bytes);

    const std::size_t n_rot_bytes = rot->buffer.size_bytes();
    std::vector<float> rot_vector(rot->count * 4);
    std::memcpy(rot_vector.data(), rot->buffer.get(), n_rot_bytes);

    // std::vector to torch::Tensor
    this->xyz_ = torch::from_blob(
        xyz_vector.data(), {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    this->features_dc_ = torch::from_blob(
        f_dc_vector.data(), {num_points, 3, 1},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).transpose(1, 2).contiguous();

    this->features_rest_ = torch::from_blob(
        f_rest_vector.data(), {num_points, 3, n_f_rest / 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_).transpose(1, 2).contiguous();

    this->opacity_ = torch::from_blob(
        opacity_vector.data(), {num_points, 1},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    this->scaling_ = torch::from_blob(
        scales_vector.data(), {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    this->rotation_ = torch::from_blob(
        rot_vector.data(), {num_points, 4},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    this->active_sh_degree_ = this->max_sh_degree_;
}

void GaussianModelLine::savePly(std::filesystem::path result_path)
{
    // Prepare data to write
    torch::Tensor xyz = this->xyz_.detach().cpu();
    torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor f_dc = this->features_dc_.detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor f_rest = this->features_rest_.detach().transpose(1, 2).flatten(1).contiguous().cpu();
    torch::Tensor opacities = this->opacity_.detach().cpu();
    torch::Tensor scale = this->scaling_.detach().cpu();
    torch::Tensor rotation = this->rotation_.detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(result_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + result_path.string());

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // normals
    result_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"},
        tinyply::Type::FLOAT32, normals.size(0),
        reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_dc
    std::size_t n_f_dc = this->features_dc_.size(1) * this->features_dc_.size(2);
    std::vector<std::string> property_names_f_dc(n_f_dc);
    for (int i = 0; i < n_f_dc; ++i)
        property_names_f_dc[i] = "f_dc_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_dc,
        tinyply::Type::FLOAT32, this->features_dc_.size(0),
        reinterpret_cast<uint8_t*>(f_dc.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // f_rest
    std::size_t n_f_rest = this->features_rest_.size(1) * this->features_rest_.size(2);
    std::vector<std::string> property_names_f_rest(n_f_rest);
    for (int i = 0; i < n_f_rest; ++i)
        property_names_f_rest[i] = "f_rest_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_f_rest,
        tinyply::Type::FLOAT32, this->features_rest_.size(0),
        reinterpret_cast<uint8_t*>(f_rest.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // opacities
    result_file.add_properties_to_element(
        "vertex", {"opacity"},
        tinyply::Type::FLOAT32, opacities.size(0),
        reinterpret_cast<uint8_t*>(opacities.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // scale
    std::size_t n_scale = scale.size(1);
    std::vector<std::string> property_names_scale(n_scale);
    for (int i = 0; i < n_scale; ++i)
        property_names_scale[i] = "scale_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_scale,
        tinyply::Type::FLOAT32, scale.size(0),
        reinterpret_cast<uint8_t*>(scale.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // rotation
    std::size_t n_rotation = rotation.size(1);
    std::vector<std::string> property_names_rotation(n_rotation);
    for (int i = 0; i < n_rotation; ++i)
        property_names_rotation[i] = "rot_" + std::to_string(i);

    result_file.add_properties_to_element(
        "vertex", property_names_rotation,
        tinyply::Type::FLOAT32, rotation.size(0),
        reinterpret_cast<uint8_t*>(rotation.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

void GaussianModelLine::saveSparsePointsPly(std::filesystem::path result_path)
{
    // Prepare data to write
    torch::Tensor xyz = this->sparse_points_xyz_.detach().cpu();
    torch::Tensor normals = torch::zeros_like(xyz);
    torch::Tensor color = (this->sparse_points_color_ * 255.0f).toType(torch::kUInt8).detach().cpu();

    std::filebuf fb_binary;
    fb_binary.open(result_path, std::ios::out | std::ios::binary);
    std::ostream outstream_binary(&fb_binary);
    if (outstream_binary.fail()) throw std::runtime_error("failed to open " + result_path.string());

    tinyply::PlyFile result_file;

    // xyz
    result_file.add_properties_to_element(
        "vertex", {"x", "y", "z"},
        tinyply::Type::FLOAT32, xyz.size(0),
        reinterpret_cast<uint8_t*>(xyz.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // normals
    result_file.add_properties_to_element(
        "vertex", {"nx", "ny", "nz"},
        tinyply::Type::FLOAT32, normals.size(0),
        reinterpret_cast<uint8_t*>(normals.data_ptr<float>()),
        tinyply::Type::INVALID, 0);

    // color
    result_file.add_properties_to_element(
        "vertex", {"red", "green", "blue"},
        tinyply::Type::UINT8, color.size(0),
        reinterpret_cast<uint8_t*>(color.data_ptr<uint8_t>()),
        tinyply::Type::INVALID, 0);

    // Write the file
    result_file.write(outstream_binary, true);

    fb_binary.close();
}

float GaussianModelLine::percentDense()
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    return percent_dense_;
}

void GaussianModelLine::setPercentDense(const float percent_dense)
{
    std::unique_lock<std::mutex> lock(mutex_settings_);
    percent_dense_ = percent_dense;
}

/**
 * @brief get_expon_lr_func
 * @details Modified from Plenoxels
 *  Continuous learning rate decay function. Adapted from JaxNeRF
 *  The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
 *  is log-linearly interpolated elsewhere (equivalent to exponential decay).
 *  If lr_delay_steps>0 then the learning rate will be scaled by some smooth
 *  function of lr_delay_mult, such that the initial learning rate is
 *  lr_init*lr_delay_mult at the beginning of optimization but will be eased back
 *  to the normal learning rate when steps>lr_delay_steps.
 *  :param conf: config subtree 'lr' or similar
 *  :param max_steps: int, the number of steps during optimization.
 *  :return HoF which takes step as input
 * @param iteration 
 * @return float 
 */
float GaussianModelLine::exponLrFunc(int step)
{
    if (step < 0 || (lr_init_ == 0.0f && lr_final_ == 0.0f))
        return 0.0f;

    float delay_rate;
    if (lr_delay_steps_ > 0)
        delay_rate = lr_delay_mult_ + (1.0f - lr_delay_mult_) * std::sin(M_PI_2f32 * std::clamp(static_cast<float>(step) / lr_delay_steps_, 0.0f, 1.0f));
    else
        delay_rate = 1.0f;
    float t = std::clamp(static_cast<float>(step) / max_steps_, 0.0f, 1.0f);
    float log_lerp = std::exp(std::log(lr_init_) * (1 - t) + std::log(lr_final_) * t);
    return delay_rate * log_lerp;
}
