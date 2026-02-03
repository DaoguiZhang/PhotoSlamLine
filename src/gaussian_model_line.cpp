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
    float k_t,      // along-line: 0.5~1.0
    float k_n)      // normal:     0.3~0.6
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
    
    // 1. 初始化 GPU 张量
    torch::Tensor fused_point_cloud = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor color = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    is_line_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    line_dir_w_ = torch::zeros(
        {num_points, 3}, 
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_)); 

    // 2. 填充基础数据 (XYZ, Color, IsLine) - CPU 辅助填充
    auto is_line_cpu = is_line_.cpu(); // 创建 CPU 副本
    auto line_acc = is_line_cpu.accessor<bool, 1>(); 
    auto pcd_it = pcd.begin();
    
    // 注意：这里你是直接操作 GPU tensor 的 index，虽然慢但能跑
    // 为了性能建议也像 is_line 那样用 CPU accessor，但目前保持原样不出错
    for (int point_idx = 0; point_idx < num_points; ++point_idx) {
        auto& point = (*pcd_it).second;
        fused_point_cloud.index({point_idx, 0}) = point.xyz_(0);
        fused_point_cloud.index({point_idx, 1}) = point.xyz_(1);
        fused_point_cloud.index({point_idx, 2}) = point.xyz_(2);
        color.index({point_idx, 0}) = point.color_(0);
        color.index({point_idx, 1}) = point.color_(1);
        color.index({point_idx, 2}) = point.color_(2);
        line_acc[point_idx] = (point.source_ == PointSourceType::LINE_SAMPLED); 
        ++pcd_it;
    }

    // 3. 计算 SH 特征
    torch::Tensor fused_color = sh_utils::RGB2SH(color);
    auto temp = this->max_sh_degree_ + 1;
    torch::Tensor features = torch::zeros(
        {fused_color.size(0), 3, temp * temp},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;
    features.index({torch::indexing::Slice(), torch::indexing::Slice(3, features.size(1)), torch::indexing::Slice(1, features.size(2))}) = 0.0f;

    // 4. 初始化默认 Rotation (Identity) 和 Scale (Isotropic)
    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1; // w=1
    
    torch::Tensor scales = torch::zeros({num_points, 3}, torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    {
        // 使用 KNN 计算普通点的初始大小
        torch::Tensor point_cloud_copy = fused_point_cloud.clone();
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(point_cloud_copy), 0.0000001);
        torch::Tensor scales_dist = torch::log(torch::sqrt(dist2));          
        scales_dist = scales_dist.unsqueeze(1).repeat({1, 3});               
        scales.copy_(scales_dist);
    }

    // 5. 针对线特征进行特殊初始化 (CPU 循环)
    // 关键：创建 CPU 副本进行操作
    torch::Tensor scales_cpu = scales.to(torch::kCPU).contiguous();
    torch::Tensor rots_cpu   = rots.to(torch::kCPU).contiguous();
    torch::Tensor line_dir_cpu = line_dir_w_.cpu().contiguous(); // 之前是全0

    auto dir_acc = line_dir_cpu.accessor<float, 2>();
    auto scales_acc = scales_cpu.accessor<float, 2>();
    auto rots_acc   = rots_cpu.accessor<float, 2>();
    
    pcd_it = pcd.begin();
    for (int i = 0; i < num_points; ++i) {
        const Point3D& p = (*pcd_it).second;
        
        if (p.source_ == PointSourceType::MAP_POINT)
        {
            // 普通点：保持默认方向和 distCUDA2 计算的 scale/rotation
            dir_acc[i][0] = 1.f; dir_acc[i][1] = 0.f; dir_acc[i][2] = 0.f;
            // 不要 continue，因为 pcd_it 需要递增 (虽然在循环头里没递增，你是放在下面)
            // 在你的原始代码中 pcd_it 是在循环体外递增的吗？
            // 修正：你的原始代码 pcd_it 是 map iterator，这里应该每次循环都要由 iterator 拿到
        }
        else 
        {
            // 线特征点：覆盖 Scale 和 Rotation
            Eigen::Vector3f d = p.line_dir_.normalized();
            dir_acc[i][0] = d.x(); dir_acc[i][1] = d.y(); dir_acc[i][2] = d.z();

            // --- Scale (各向异性) ---
            const float step  = (p.sample_step_ > 0.f) ? p.sample_step_ : 0.05f; 
            Eigen::Vector3f log_s = initLineSampleLogScale(step, p.ref_depth_z_, p.ref_focal_);
            scales_acc[i][0] = log_s[0];
            scales_acc[i][1] = log_s[1];
            scales_acc[i][2] = log_s[2];

            // --- Rotation (对齐方向) ---
            Eigen::Vector4f q = initQuatAlignXToDir(p.line_dir_);
            rots_acc[i][0] = q[0];
            rots_acc[i][1] = q[1];
            rots_acc[i][2] = q[2];
            rots_acc[i][3] = q[3];
        }
        ++pcd_it; // 确保遍历 map
    }

    // =========================================================================
    // 【关键修复】 将修改后的 CPU 数据写回 GPU！！！
    // 如果没有这几行，你的线特征初始化就全是默认值（圆球+未对齐），导致优化失败
    // =========================================================================
    scales = scales_cpu.to(device_type_).contiguous();
    rots   = rots_cpu.to(device_type_).contiguous();
    line_dir_w_ = line_dir_cpu.to(device_type_).contiguous(); // 更新成员变量
    this->is_line_ = is_line_cpu.to(device_type_).contiguous(); // 更新成员变量

    // 6. 初始化 Opacity 和其他参数
    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {fused_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    this->exist_since_iter_ = torch::zeros(
        {fused_point_cloud.size(0)},
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    // 7. 赋值给可优化参数 (Requires Grad)
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
                               
    // 这里现在使用的是已经包含了线特征初始化的正确 Tensor
    this->scaling_ = scales.requires_grad_();
    this->rotation_ = rots.requires_grad_();
    this->opacity_ = opacities.requires_grad_();

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    this->point_ids_ = torch::arange(0, num_points, 
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));

    // Debug Buffers
    this->debug_hit_count_ =
        torch::zeros({this->getXYZ().size(0)},
                 torch::TensorOptions().dtype(torch::kInt32).device(device_type_));
    this->debug_alpha_accum_ =
        torch::zeros({this->getXYZ().size(0)},
                 torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

    this->max_radii2D_ = torch::zeros({this->getXYZ().size(0)}, torch::TensorOptions().device(device_type_));
    
    // is_line_ 已经在上面更新过了，这里不需要再赋值
    // ================= [DEBUG PROBE: 检查初始化质量] =================
    if (this->is_line_.any().item<bool>()) {
        std::cerr << "\n[DEBUG createFromPcd] Checking Line Initialization:" << std::endl;
        
        // 提取线高斯的 Scaling (Exp后才是真实大小) 和 Rotation
        auto line_mask_idx = torch::nonzero(this->is_line_).squeeze();
        auto s = torch::exp(this->scaling_.index({line_mask_idx})); 
        auto r = this->rotation_.index({line_mask_idx});
        
        int num_check = std::min((int)s.size(0), 3);
        for (int i = 0; i < num_check; ++i) {
            float sx = s[i][0].item<float>(); // 沿线方向
            float sy = s[i][1].item<float>(); // 法向
            float sz = s[i][2].item<float>(); // 法向
            
            float qw = r[i][0].item<float>();
            float qx = r[i][1].item<float>();
            
            std::cerr << "  Line " << i << ": Scale=[" 
                      << sx << ", " << sy << ", " << sz << "] "
                      << "Ratio=" << sx / std::max(sy, 1e-6f) 
                      << " | Rot(w,x)=[" << qw << ", " << qx << "...]" << std::endl;
        }
        std::cerr << "  > If Ratio > 1.0 and Rot != [1,0,0,0], initialization is GOOD.\n" << std::endl;
    } else {
        std::cerr << "\n[DEBUG createFromPcd] No Line Points found in this batch.\n" << std::endl;
    }
    // =================================================================
}

#if 0
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

    // std::cerr << "[Gaussian Model]Number of points at initialization : " << fused_point_cloud.size(0) << std::endl;

    torch::Tensor rots = torch::zeros({fused_point_cloud.size(0), 4}, torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1;
    
    // ---- init scales: 分两类 ----
    // 先给普通点按原始策略算一个 scales_dist（log-space）
    //torch::Tensor scales = torch::zeros({num_points, 3},
    //    torch::TensorOptions().dtype(torch::kFloat));
    // 【修改后】必须指定 device(device_type_)
    torch::Tensor scales = torch::zeros({num_points, 3}, 
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));

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

    this->point_ids_ = torch::arange(0, num_points, 
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));

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
#endif

void GaussianModelLine::increasePcd(
    const std::vector<Point3D>& new_points,
    const int iteration)
{
    const int N = static_cast<int>(new_points.size());
    if (N == 0) return;

    // 1. 准备 CPU 缓冲
    torch::Tensor xyz_cpu = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor col_cpu = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor dir_cpu = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor is_line_cpu = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));
    
    // 准备 Scale 和 Rotation 的 CPU 缓冲 (N,3) 和 (N,4)
    torch::Tensor scale_cpu = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor rot_cpu   = torch::zeros({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    // 默认 Rotation 为 Identity (w=1)
    rot_cpu.index_put_({torch::indexing::Slice(), 0}, 1.0f);

    {
        auto xyz_acc = xyz_cpu.accessor<float, 2>();
        auto col_acc = col_cpu.accessor<float, 2>();
        auto dir_acc = dir_cpu.accessor<float, 2>();
        auto line_acc = is_line_cpu.accessor<bool, 1>();
        auto s_acc = scale_cpu.accessor<float, 2>();
        auto r_acc = rot_cpu.accessor<float, 2>();

        for (int i = 0; i < N; ++i) {
            const auto& p = new_points[i];
            
            xyz_acc[i][0] = p.xyz_.x(); xyz_acc[i][1] = p.xyz_.y(); xyz_acc[i][2] = p.xyz_.z();
            col_acc[i][0] = p.color_.x(); col_acc[i][1] = p.color_.y(); col_acc[i][2] = p.color_.z();
            
            bool is_line_pt = (p.source_ == PointSourceType::LINE_SAMPLED);
            line_acc[i] = is_line_pt;

            if (is_line_pt) {
                // Line: Set Direction, Anisotropic Scale, Aligned Rotation
                Eigen::Vector3f d = p.line_dir_.normalized();
                dir_acc[i][0] = d.x(); dir_acc[i][1] = d.y(); dir_acc[i][2] = d.z();

                const float step = (p.sample_step_ > 0.f) ? p.sample_step_ : 0.05f;
                Eigen::Vector3f log_s = initLineSampleLogScale(step, p.ref_depth_z_, p.ref_focal_);
                s_acc[i][0] = log_s[0]; s_acc[i][1] = log_s[1]; s_acc[i][2] = log_s[2];

                Eigen::Vector4f q = initQuatAlignXToDir(p.line_dir_);
                r_acc[i][0] = q[0]; r_acc[i][1] = q[1]; r_acc[i][2] = q[2]; r_acc[i][3] = q[3];
            } else {
                // Point: Default Direction, Placeholder Scale (will be calc by distCUDA2), Identity Rotation
                dir_acc[i][0] = 1.0f;
            }
        }
    }

    // 2. 移动到 GPU
    auto new_xyz = xyz_cpu.to(device_type_).contiguous();
    auto new_color = col_cpu.to(device_type_).contiguous();
    auto new_line_dir_w = dir_cpu.to(device_type_).contiguous();
    auto new_is_line = is_line_cpu.to(device_type_).contiguous();
    
    auto new_rotation = rot_cpu.to(device_type_).contiguous();
    auto new_scaling = scale_cpu.to(device_type_).contiguous();

    // 3. 为普通点计算 Scale (覆盖掉上面的 0)
    {
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(new_xyz.clone()), 1e-7);
        torch::Tensor dist_scales = torch::log(torch::sqrt(dist2)).unsqueeze(1).repeat({1, 3});
        
        torch::Tensor is_point_mask = ~new_is_line;
        if (is_point_mask.any().item<bool>()) {
            new_scaling.index_put_({is_point_mask}, dist_scales.index({is_point_mask}));
        }
    }

    // 4. Features & Opacity & Aux
    torch::Tensor fused_color = sh_utils::RGB2SH(new_color);
    const int sh_dim = (max_sh_degree_ + 1) * (max_sh_degree_ + 1);
    torch::Tensor features = torch::zeros({N, 3, sh_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    features.index_put_({torch::indexing::Slice(), torch::indexing::Slice(), 0}, fused_color);

    auto new_features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
    auto new_features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, sh_dim)}).transpose(1, 2).contiguous();

    auto new_opacity = general_utils::inverse_sigmoid(0.1f * torch::ones({N, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)));
    auto new_exist_since_iter = torch::full({N}, iteration, torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    // 5. Call Postfix (Safe Version)
    densificationPostfixWithLineAwareness(
        new_xyz, new_features_dc, new_features_rest, new_opacity, 
        new_scaling, new_rotation, new_exist_since_iter, 
        new_is_line, new_line_dir_w
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
}

#if 0
void GaussianModelLine::increasePcd(
    const std::vector<Point3D>& new_points,
    const int iteration)
{
    const int N = static_cast<int>(new_points.size());
    if (N == 0) return;

    // =================================================================================
    // 1) 第一步：在 CPU 上一次性打包所有数据 (XYZ, Color, Direction, IsLine)
    // =================================================================================
    torch::Tensor xyz_cpu = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor col_cpu = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor dir_cpu = torch::zeros({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    torch::Tensor is_line_cpu = torch::zeros({N}, torch::TensorOptions().dtype(torch::kBool).device(torch::kCPU));

    {
        auto xyz_acc = xyz_cpu.accessor<float, 2>();
        auto col_acc = col_cpu.accessor<float, 2>();
        auto dir_acc = dir_cpu.accessor<float, 2>();
        auto line_acc = is_line_cpu.accessor<bool, 1>();

        for (int i = 0; i < N; ++i) {
            const auto& p = new_points[i];
            
            // 基础属性
            xyz_acc[i][0] = p.xyz_.x();
            xyz_acc[i][1] = p.xyz_.y();
            xyz_acc[i][2] = p.xyz_.z();
            col_acc[i][0] = p.color_.x();
            col_acc[i][1] = p.color_.y();
            col_acc[i][2] = p.color_.z();
            
            // 线特征属性
            if (p.source_ == PointSourceType::LINE_SAMPLED) {
                Eigen::Vector3f d = p.line_dir_.normalized();
                dir_acc[i][0] = d.x();
                dir_acc[i][1] = d.y();
                dir_acc[i][2] = d.z();
                line_acc[i] = true;     //设置好是否为line point
            } else {
                // 普通点给个默认方向(1,0,0)，防止 NaN，但 is_line=false
                dir_acc[i][0] = 1.0f; 
                dir_acc[i][1] = 0.0f; 
                dir_acc[i][2] = 0.0f;
                line_acc[i] = false;
            }
        }
    }

    // =================================================================================
    // 2) 统一移动到 GPU
    // =================================================================================
    torch::Tensor new_xyz   = xyz_cpu.to(device_type_, /*non_blocking=*/false).contiguous();
    torch::Tensor new_color = col_cpu.to(device_type_, /*non_blocking=*/false).contiguous();
    // 【关键】直接使用上面生成的 tensor，不要重新定义！
    torch::Tensor new_line_dir_w = dir_cpu.to(device_type_, /*non_blocking=*/false).contiguous();
    torch::Tensor new_is_line    = is_line_cpu.to(device_type_, /*non_blocking=*/false).contiguous();

    // =================================================================================
    // 3) 初始化 SH 特征 (GPU)
    // =================================================================================
    torch::Tensor fused_color = sh_utils::RGB2SH(new_color);
    const int sh_dim = (max_sh_degree_ + 1) * (max_sh_degree_ + 1);

    torch::Tensor features = torch::zeros({N, 3, sh_dim}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    features.index({torch::indexing::Slice(), torch::indexing::Slice(0, 3), 0}) = fused_color;

    torch::Tensor new_features_dc = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(0, 1)}).transpose(1, 2).contiguous();
    torch::Tensor new_features_rest = features.index({torch::indexing::Slice(), torch::indexing::Slice(), torch::indexing::Slice(1, sh_dim)}).transpose(1, 2).contiguous();

    // =================================================================================
    // 4) 初始化 Scale 和 Rotation (先在 GPU 做通用的，再去 CPU 修正线特征)
    // =================================================================================
    // 默认 Rotation: identity
    torch::Tensor new_rotation = torch::zeros({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    new_rotation.index({torch::indexing::Slice(), 0}) = 1.f;

    // 默认 Scale: dist-based isotropic
    torch::Tensor new_scaling;
    {
        torch::Tensor dist2 = torch::clamp_min(distCUDA2(new_xyz.clone()), 1e-7);
        torch::Tensor s = torch::log(torch::sqrt(dist2)).unsqueeze(1).repeat({1, 3}); 
        new_scaling = s.contiguous();
    }

    // -----------------------------------------------------------
    // CPU 修正循环：针对 Line Sampled 点，计算特殊的 Scale 和 Rotation
    // -----------------------------------------------------------
    torch::Tensor scaling_cpu  = new_scaling.to(torch::kCPU).contiguous();
    torch::Tensor rotation_cpu = new_rotation.to(torch::kCPU).contiguous();
    
    // 【优化】不需要再把 is_line 拷回 CPU 了，因为我们有原始的 new_points 数组
    // 我们可以直接通过 p.source_ 判断
    {
        auto s_acc = scaling_cpu.accessor<float, 2>();
        auto r_acc = rotation_cpu.accessor<float, 2>();

        for (int i = 0; i < N; ++i) {
            const auto& p = new_points[i];
            
            // 仅修正线特征点
            if (p.source_ != PointSourceType::LINE_SAMPLED) continue;

            // 1. 各向异性 Scale
            const float step = (p.sample_step_ > 0.f) ? p.sample_step_ : 0.05f;
            Eigen::Vector3f log_s = initLineSampleLogScale(step, p.ref_depth_z_, p.ref_focal_);
            s_acc[i][0] = log_s[0];
            s_acc[i][1] = log_s[1];
            s_acc[i][2] = log_s[2];

            // 2. 对齐 Rotation
            Eigen::Vector4f q = initQuatAlignXToDir(p.line_dir_);
            r_acc[i][0] = q[0];
            r_acc[i][1] = q[1];
            r_acc[i][2] = q[2];
            r_acc[i][3] = q[3];
            
            // 注意：不需要再设置 is_line_acc[i] = true，因为第一步已经设过了
        }
    }

    // 拷回 GPU
    new_scaling  = scaling_cpu.to(device_type_).contiguous();
    new_rotation = rotation_cpu.to(device_type_).contiguous();

    // =================================================================================
    // 5) 初始化 Opacity
    // =================================================================================
    torch::Tensor new_opacity = general_utils::inverse_sigmoid(0.1f * torch::ones({N, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)));
    torch::Tensor new_exist_since_iter = torch::full({N}, iteration, torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    // =================================================================================
    // 6) 调用带 Line Awareness 的 Postfix
    // =================================================================================
    densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_since_iter,
        new_is_line,       // 这里传入的是第一步正确生成的 tensor
        new_line_dir_w     // 这里传入的是第一步正确生成的 tensor
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
}
#endif

void GaussianModelLine::increasePcd(std::vector<float> points, std::vector<float> colors, const int iteration)
{
    // auto time1 = std::chrono::steady_clock::now();
    assert(points.size() == colors.size());
    assert(points.size() % 3 == 0);
    auto num_new_points = static_cast<int>(points.size() / 3);
    if (num_new_points == 0)
        return;

    // 1. 基础数据转换 (保持不变)
    torch::Tensor new_point_cloud = torch::from_blob(
        points.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);
    
    torch::Tensor new_colors = torch::from_blob(
        colors.data(), {num_new_points, 3},
        torch::TensorOptions().dtype(torch::kFloat32)).to(device_type_);

    // 2. 更新稀疏点云缓存 (保持不变)
    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
    }

    // 3. 计算 SH 特征 (保持不变)
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

    // 4. 初始化几何属性 (Scale, Rot, Opacity)
    torch::Tensor dist2 = torch::clamp_min(
        distCUDA2(new_point_cloud.clone()), 0.0000001);
    torch::Tensor scales = torch::log(torch::sqrt(dist2));
    auto scales_ndimension = scales.ndimension();
    scales = scales.unsqueeze(scales_ndimension).repeat({1, 3});
    
    torch::Tensor rots = torch::zeros(
        {new_point_cloud.size(0), 4},
         torch::TensorOptions().device(device_type_));
    rots.index({torch::indexing::Slice(), 0}) = 1; // Identity quaternion
    
    torch::Tensor opacities = general_utils::inverse_sigmoid(
        0.1f * torch::ones(
                   {new_point_cloud.size(0), 1},
                   torch::TensorOptions().dtype(torch::kFloat).device(device_type_)));

    torch::Tensor new_exist_since_iter = torch::full(
        {new_point_cloud.size(0)},
        iteration,
        torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    // =========================================================
    // 【新增】为普通点创建默认的 Line 属性，以对齐维度
    // =========================================================
    
    // 1. is_line 全为 false
    torch::Tensor new_is_line = torch::zeros(
        {num_new_points}, 
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    
    // 2. line_dir 给个默认值 (例如 X 轴 [1, 0, 0])，防止数值计算错误
    // 虽然 is_line=false 时这个值理论上不会被用到，但保持非零是个好习惯
    torch::Tensor new_line_dir_w = torch::zeros(
        {num_new_points, 3}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    new_line_dir_w.index({torch::indexing::Slice(), 0}) = 1.0f; 

    // 准备参数
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

    // ================= [DEBUG PROBE: 检查 increasePcd 数据流] =================
    std::cerr << "\n[DEBUG increasePcd(Vector)] Preparing to call Postfix. N_new=" << num_new_points << std::endl;
    std::cerr << "  > new_xyz shape:      " << new_xyz.sizes() << std::endl;
    std::cerr << "  > new_rotation shape: " << new_rotation.sizes() << " (Expect [N, 4])" << std::endl;
    std::cerr << "  > new_scaling shape:  " << new_scaling.sizes() << " (Expect [N, 3])" << std::endl;
    std::cerr << "  > new_features_dc:    " << new_features_dc.sizes() << std::endl;
    
    // 强制检查：如果 Rotation 维度不对，打印红色警告
    if (new_rotation.dim() != 2 || new_rotation.size(1) != 4) {
        std::cerr << "\033[1;31m[CRITICAL ALERT] new_rotation dimension is WRONG inside increasePcd!\033[0m" << std::endl;
        // 尝试自动修复 (虽然之前的 torch::zeros 应该是对的，但以防万一)
        if (new_rotation.numel() == num_new_points * 4) {
             std::cerr << "[Auto-Fix] Reshaping rotation to [N, 4]..." << std::endl;
             new_rotation = new_rotation.view({num_new_points, 4}).contiguous();
        }
    }

    // =========================================================
    // 【替换】调用结构感知的 Postfix
    // =========================================================
    densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter,
        new_is_line,       // 传入
        new_line_dir_w     // 传入
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
}


void GaussianModelLine::increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors, const int iteration)
{
// auto time1 = std::chrono::steady_clock::now();
    auto num_new_points = new_point_cloud.size(0);
    if (num_new_points == 0)
        return;

    // 1. 更新稀疏点云缓存 (保持不变)
    if (sparse_points_xyz_.size(0) == 0) {
        sparse_points_xyz_ = new_point_cloud;
        sparse_points_color_ = new_colors;
    }
    else {
        sparse_points_xyz_ = torch::cat({sparse_points_xyz_, new_point_cloud}, /*dim=*/0);
        sparse_points_color_ = torch::cat({sparse_points_color_, new_colors}, /*dim=*/0);
    }

    // 2. 计算 SH 特征 (保持不变)
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

    // 3. 计算 Scale, Rotation, Opacity (保持不变)
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
// std::cerr << "increasePcd(tensor) preparation time: " << time << " ms" <<std::endl;

    // =================================================================================
    // 【新增】创建默认的 line 属性张量
    // 即使这些是普通点，也必须扩展这些张量以保持维度对齐 (N)
    // =================================================================================
    
    // 1. is_line 全为 false (bool)
    torch::Tensor new_is_line = torch::zeros(
        {num_new_points}, 
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    
    // 2. line_dir_w 全为默认值 [1, 0, 0] (float32)
    torch::Tensor new_line_dir_w = torch::zeros(
        {num_new_points, 3}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device_type_));
    
    // 赋予一个非零向量，避免归一化时除零
    new_line_dir_w.index({torch::indexing::Slice(), 0}) = 1.0f; 

    // =================================================================================
    // 【替换】使用带 Line Awareness 的 Postfix
    // =================================================================================
    densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter,
        new_is_line,       // Add
        new_line_dir_w     // Add
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
// auto time3 = std::chrono::steady_clock::now();
// time = std::chrono::duration_cast<std::chrono::milliseconds>(time3-time2).count();
// std::cerr << "increasePcd(tensor) postfix time: " << time << " ms" <<std::endl;
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

#if 0
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
#endif

void GaussianModelLine::densificationPostfixWithLineAwareness(
    torch::Tensor& new_xyz,
    torch::Tensor& new_features_dc,
    torch::Tensor& new_features_rest,
    torch::Tensor& new_opacities,
    torch::Tensor& new_scaling,
    torch::Tensor& new_rotation,
    torch::Tensor& new_exist_since_iter,
    torch::Tensor& new_is_line,
    torch::Tensor& new_line_dir_w)
{
    // ================= [DEBUG PROBE: 强制打印到错误流] =================
    // 使用 std::cerr 而不是 std::cerr，防止崩溃时日志丢失
    std::cerr << "\n[DEBUG FLOW] Entering Postfix. N=" << new_xyz.size(0) << std::endl;
    std::cerr << "  > XYZ Shape: " << new_xyz.sizes() << std::endl;
    std::cerr << "  > Rotation Shape: " << new_rotation.sizes() << " (Expect: [N, 4])" << std::endl;
    std::cerr << "  > Scaling Shape:  " << new_scaling.sizes()  << " (Expect: [N, 3])" << std::endl;
    // =========================================================
    // [CRITICAL FIX] 维度强制校验与修复
    // 解决 "Expected size 4 but got size N" 问题
    // =========================================================
    int64_t N = new_xyz.size(0); // 以 XYZ 的点数为基准

    // 1. 修复 Rotation: 必须是 [N, 4]
    // 报错 Expected size 4 but got 115 就是在这里被拦截修复
    if (new_rotation.dim() == 1 && new_rotation.numel() == N * 4) {
        new_rotation = new_rotation.view({N, 4});
    }
    else if (new_rotation.size(0) == 4 && new_rotation.size(1) == N) {
        new_rotation = new_rotation.t().contiguous(); 
    }
    // 最后的防线
    if (new_rotation.size(1) != 4) {
        std::cerr << "[GaussianModel] CRITICAL ERROR: Rotation shape " << new_rotation.sizes() << " is wrong! Trying to reshape..." << std::endl;
        new_rotation = new_rotation.view({N, 4}).contiguous();
    }

    // 2. 修复 Scaling: 必须是 [N, 3]
    if (new_scaling.dim() == 1 && new_scaling.numel() == N * 3) {
        new_scaling = new_scaling.view({N, 3});
    }
    else if (new_scaling.size(0) == 3 && new_scaling.size(1) == N) {
        new_scaling = new_scaling.t().contiguous();
    }
    new_scaling = new_scaling.contiguous();

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
        if (group.params().size() != 1) continue; 

        auto& extension_tensor = tensors_dict[group_idx];
        auto& param = group.params()[0];
        auto key = param.unsafeGetTensorImpl();

        if (state.find(key) != state.end()) {
            // ---- extend Adam state ----
            auto& stored_state = static_cast<torch::optim::AdamParamState&>(*state[key]);
            auto new_state = std::make_unique<torch::optim::AdamParamState>();
            
            new_state->step(stored_state.step());
            
            // 拼接 Adam 状态 (Momentum)
            new_state->exp_avg(
                torch::cat({stored_state.exp_avg(), torch::zeros_like(extension_tensor)}, 0));
            new_state->exp_avg_sq(
                torch::cat({stored_state.exp_avg_sq(), torch::zeros_like(extension_tensor)}, 0));

            state.erase(key);

            // 【关键】拼接参数并开启梯度
            // 注意：这里我们使用 extension_tensor，它已经被上面的修复逻辑校正了维度
            param = torch::cat({param, extension_tensor}, 0).requires_grad_();
            
            key = param.unsafeGetTensorImpl();
            state[key] = std::move(new_state);

            // 更新 param_group 里的引用
            group.params()[0] = param;
            optimizable_tensors[group_idx] = param;
        }
        else {
            // no state yet
            param = torch::cat({param, extension_tensor}, 0).requires_grad_();
            group.params()[0] = param;
            optimizable_tensors[group_idx] = param;
        }
    }

    // =====================================================
    // B. 写回类成员变量 (必须做，否则下一次迭代找不到参数)
    // =====================================================
    this->xyz_           = optimizable_tensors[0];
    this->features_dc_   = optimizable_tensors[1];
    this->features_rest_ = optimizable_tensors[2];
    this->opacity_       = optimizable_tensors[3];
    this->scaling_       = optimizable_tensors[4];
    // 【验证点 1】：赋值前检查 optimizer 返回的 rotation 形状
    auto rot_from_opt = optimizable_tensors[5];
    if (rot_from_opt.dim() != 2 || rot_from_opt.size(1) != 4) {
        std::cerr << "\n[VERIFY FAIL] Optimizer produced BAD Rotation!" << std::endl;
        std::cerr << "  > Shape: " << rot_from_opt.sizes() << std::endl;
        // 尝试最后一次抢救，防止下一帧崩掉
        // optimizable_tensors[5] = rot_from_opt.reshape({-1, 4}).contiguous();
    }
    this->rotation_      = optimizable_tensors[5];
    // 【验证点 2】：赋值后检查成员变量状态
    if (this->rotation_.dim() == 1) {
        std::cerr << "\033[1;31m[FATAL CONFIRMED] this->rotation_ IS NOW 1D!\033[0m" << std::endl;
        std::cerr << "  > Size: " << this->rotation_.sizes() << std::endl;
        std::cerr << "  > This will crash the NEXT iteration." << std::endl;
    } else {
        // std::cerr << "[VERIFY OK] Postfix finished. Rotation shape: " << this->rotation_.sizes() << std::endl;
    }


    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    // 更新辅助向量 (如果你的 optimizer step 依赖它)
    this->Tensor_vec_xyz_ = {this->xyz_}; 
    
    // ---- 完善 point_ids_ 的维护 ----
    int64_t max_id = 0;
    if (this->point_ids_.size(0) > 0) {
        max_id = this->point_ids_.max().item<int64_t>();
    }
    torch::Tensor new_ids = torch::arange(max_id + 1, max_id + 1 + N, 
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    this->point_ids_ = torch::cat({this->point_ids_, new_ids}, 0);

    // =====================================================
    // C. Structure-only tensors (NO optimizer state)
    // =====================================================
    // 确保这些 tensor 也在正确的设备上
    this->is_line_ = torch::cat({this->is_line_, new_is_line.to(device_type_)}, 0);
    this->line_dir_w_ = torch::cat({this->line_dir_w_, new_line_dir_w.to(device_type_)}, 0);
    this->exist_since_iter_ = torch::cat({this->exist_since_iter_, new_exist_since_iter.to(device_type_)}, 0);

    // =====================================================
    // D. Reset auxiliary buffers (size must match new total)
    // =====================================================
    int64_t TotalN = this->xyz_.size(0);
    this->xyz_gradient_accum_ = torch::zeros({TotalN, 1}, torch::TensorOptions().device(device_type_));
    this->denom_ = torch::zeros({TotalN, 1}, torch::TensorOptions().device(device_type_));
    this->max_radii2D_ = torch::zeros({TotalN}, torch::TensorOptions().device(device_type_));
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

#if 0
// 辅助函数：构建旋转矩阵 (直接复用你提供的逻辑或引用头文件)
// 如果项目中已有 general_utils::build_rotation，则直接使用，无需重复定义
// inline torch::Tensor build_rotation(torch::Tensor &r) { ... } 

void GaussianModelLine::densifyAndSplitWithLineAwareness(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent,
    int N)
{
    using namespace torch::indexing;

    int n_init_points = this->getXYZ().size(0);
    
    // =========================================================================
    // 1. 筛选 (Selection): 找出梯度大且 Scaling 大的点
    // =========================================================================
    auto padded_grad = torch::zeros({n_init_points}, torch::TensorOptions().device(device_type_));
    padded_grad.slice(/*dim=*/0L, /*start=*/0, /*end=*/grads.size(0)).copy_(grads.squeeze());
    
    auto selected_mask = torch::where(padded_grad >= grad_threshold, true, false);
    selected_mask = torch::logical_and(
        selected_mask,
        std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) > percentDense() * scene_extent
    );

    int64_t M = selected_mask.sum().item<int64_t>();
    if (M == 0) return;

    // =========================================================================
    // 2. 数据准备 (Data Preparation)
    // =========================================================================
    // 提取被选中的数据
    auto xyz_sel = this->getXYZ().index({selected_mask});
    auto scale_sel = this->getScalingActivation().index({selected_mask});
    auto rot_sel = this->rotation_.index({selected_mask});
    
    // [Safety] 强制确保 Rotation 维度正确 [M, 4]
    if (rot_sel.dim() != 2 || rot_sel.size(1) != 4) {
        rot_sel = rot_sel.reshape({-1, 4}).contiguous();
    }
    
    // 提取线特征属性
    auto is_line_sel = this->is_line_.index({selected_mask});
    auto line_dir_sel = this->line_dir_w_.index({selected_mask});

    // 计算旋转矩阵 R [M, 3, 3]
    auto rots = general_utils::build_rotation(rot_sel);

    // =========================================================================
    // 3. 分类处理 (Branching): Point vs Line
    // =========================================================================
    // 通过 Scaling 形状辅助判断 (Line 通常长宽比很大)
    auto sorted_scale = std::get<0>(scale_sel.sort(1, true)); // 降序
    auto geom_is_line = sorted_scale.index({Slice(), 0}) / (sorted_scale.index({Slice(), 1}) + 1e-6) > 4.0f;
    
    // 最终分类掩码 (相对于 selected_mask 的子集)
    auto mask_line_subset = torch::logical_and(is_line_sel, geom_is_line);
    auto mask_point_subset = ~mask_line_subset;

    // 容器
    std::vector<torch::Tensor> list_xyz, list_scale, list_rot;
    std::vector<torch::Tensor> list_is_line, list_line_dir;
    // 辅助索引 (用于从原始 sel 数据中提取 Feature 等)
    std::vector<torch::Tensor> list_indices_in_sel; 

    auto opts = xyz_sel.options();

    // -------------------------------------------------------------------------
    // 分支 A: 普通点 (Points) - 原始逻辑 (3D 采样, 全局缩小)
    // -------------------------------------------------------------------------
    if (mask_point_subset.any().item<bool>()) {
        auto xyz_p = xyz_sel.index({mask_point_subset});       // [Mp, 3]
        auto scale_p = scale_sel.index({mask_point_subset});   // [Mp, 3]
        auto rots_p = rots.index({mask_point_subset});         // [Mp, 3, 3]
        auto rot_raw_p = rot_sel.index({mask_point_subset});   // [Mp, 4]

        int64_t Mp = xyz_p.size(0);

        // 3D 随机采样: 均值为0，标准差为 scale_p
        // [Mp, N, 3] -> [Mp*N, 3]
        auto stds = scale_p.repeat({N, 1}); 
        auto means = torch::zeros({stds.size(0), 3}, opts);
        auto samples = at::normal(means, stds); 

        // 旋转并平移: P_new = P_old + R * sample
        auto rots_repeated = rots_p.repeat({N, 1, 1}); // [Mp*N, 3, 3]
        auto new_xyz_p = torch::bmm(rots_repeated, samples.unsqueeze(-1)).squeeze(-1) + xyz_p.repeat({N, 1});
        
        // Scaling: 所有轴都缩小
        auto new_scale_p = torch::log(scale_p.repeat({N, 1}) / (0.8 * N));
        
        // Rotation: 简单复制
        auto new_rot_p = rot_raw_p.repeat({N, 1}); // [Mp*N, 4]

        // 存入列表
        list_xyz.push_back(new_xyz_p);
        list_scale.push_back(new_scale_p);
        list_rot.push_back(new_rot_p);
        
        // 结构属性
        list_is_line.push_back(torch::zeros({Mp * N}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(torch::zeros({Mp * N, 3}, opts)); // Dummy direction

        // 记录索引，用于后续提取 Features/Opacity
        list_indices_in_sel.push_back(torch::nonzero(mask_point_subset).squeeze(1).repeat({N}));
    }

    // -------------------------------------------------------------------------
    // 分支 B: 线特征 (Lines) - 新逻辑 (1D 采样, 仅缩小长轴)
    // -------------------------------------------------------------------------
    if (mask_line_subset.any().item<bool>()) {
        auto xyz_l = xyz_sel.index({mask_line_subset});       // [Ml, 3]
        auto scale_l = scale_sel.index({mask_line_subset});   // [Ml, 3]
        auto rots_l = rots.index({mask_line_subset});         // [Ml, 3, 3]
        auto rot_raw_l = rot_sel.index({mask_line_subset});   // [Ml, 4]
        auto dir_l = line_dir_sel.index({mask_line_subset});  // [Ml, 3]

        int64_t Ml = xyz_l.size(0);
        const int axis = 0; // 假设局部 X 轴是线方向 (Long axis)

        // 1D 随机采样: 只在 X 轴 (长轴) 上采样
        // 构造采样标准差: X轴用 scale_x, Y/Z轴设为 0
        auto scale_repeated = scale_l.repeat({N, 1}); // [Ml*N, 3]
        auto stds_1d = torch::zeros_like(scale_repeated);
        stds_1d.index_put_({Slice(), axis}, scale_repeated.index({Slice(), axis})); // 只填充 X 列

        auto means = torch::zeros({stds_1d.size(0), 3}, opts);
        auto samples = at::normal(means, stds_1d); // 采样结果: [noise_x, 0, 0]

        // 旋转并平移
        auto rots_repeated = rots_l.repeat({N, 1, 1});
        auto new_xyz_l = torch::bmm(rots_repeated, samples.unsqueeze(-1)).squeeze(-1) + xyz_l.repeat({N, 1});

        // Scaling: 只缩小 X 轴 (长度变短)，Y/Z (粗细) 保持不变
        // new_s_x = s_x / (0.8 * N)
        // new_s_y = s_y
        auto new_scale_act = scale_l.repeat({N, 1});
        new_scale_act.index_put_({Slice(), axis}, new_scale_act.index({Slice(), axis}) / (0.8 * N));
        auto new_scale_l = torch::log(new_scale_act);

        // Rotation: 简单复制 (方向不变)
        auto new_rot_l = rot_raw_l.repeat({N, 1});

        // Line Dir: 复制
        auto new_dir_l = dir_l.repeat({N, 1});
        new_dir_l = new_dir_l / (new_dir_l.norm(2, 1, true) + 1e-6);

        // 存入列表
        list_xyz.push_back(new_xyz_l);
        list_scale.push_back(new_scale_l);
        list_rot.push_back(new_rot_l);
        
        list_is_line.push_back(torch::ones({Ml * N}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(new_dir_l);

        list_indices_in_sel.push_back(torch::nonzero(mask_line_subset).squeeze(1).repeat({N}));
    }

    if (list_xyz.empty()) return;

    // =========================================================================
    // 4. 合并数据 (Concatenate)
    // =========================================================================
    auto new_xyz = torch::cat(list_xyz, 0);
    auto new_scaling = torch::cat(list_scale, 0);
    auto new_rotation = torch::cat(list_rot, 0);
    auto new_is_line = torch::cat(list_is_line, 0);
    auto new_line_dir_w = torch::cat(list_line_dir, 0);
    
    // 使用索引提取对应的 Features 和 Opacity (避免重复写逻辑)
    auto combined_indices = torch::cat(list_indices_in_sel, 0); // [Total_New_N]
    
    // 从 selected_mask 的原始数据中提取，并复制 N 份
    // 注意：features 等原始数据需要先用 selected_mask 筛选出 M 个，再用 combined_indices 提取
    auto feat_dc_sel = this->features_dc_.index({selected_mask});
    auto feat_rst_sel = this->features_rest_.index({selected_mask});
    auto opac_sel = this->opacity_.index({selected_mask});
    auto iter_sel = this->exist_since_iter_.index({selected_mask});

    auto new_features_dc = feat_dc_sel.index({combined_indices});
    auto new_features_rest = feat_rst_sel.index({combined_indices});
    auto new_opacity = opac_sel.index({combined_indices});
    auto new_exist_since_iter = iter_sel.index({combined_indices});

    // ================= [DEBUG PROBE 2: Split 检查] =================
    std::cerr << "[DEBUG FLOW] Inside densifyAndSplit. Prepared tensors:" << std::endl;
    std::cerr << "  > new_rotation list size: " << list_rot.size() << std::endl;
    if (!list_rot.empty()) {
        std::cerr << "  > first tensor in rot list: " << list_rot[0].sizes() << std::endl;
    }
    std::cerr << "  > Final new_rotation shape: " << new_rotation.sizes() << std::endl;
    // ============================================================

    // =========================================================================
    // 5. 调用 Postfix (加入优化器)
    // =========================================================================
    this->densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_since_iter,
        new_is_line,
        new_line_dir_w
    );

    // =========================================================================
    // 6. 删除旧点 (Prune)
    // =========================================================================
    auto prune_filter = torch::cat({
        selected_mask, // 原始被选中的点被删除 (因为已经分裂成了新的)
        torch::zeros({new_xyz.size(0)}, torch::TensorOptions().device(device_type_).dtype(torch::kBool))
    });
    this->prunePointsWithLineAwareness(prune_filter);
}
#endif

#if 0
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

    // ================= [DEBUG PROBE 2: Split 检查] =================
    std::cerr << "[DEBUG FLOW] Inside densifyAndSplit. Prepared tensors:" << std::endl;
    std::cerr << "  > new_rotation list size: " << new_rot_list.size() << std::endl;
    if (!new_rot_list.empty()) {
        std::cerr << "  > first tensor in rot list: " << new_rot_list[0].sizes() << std::endl;
    }
    std::cerr << "  > Final new_rotation shape: " << new_rotation.sizes() << std::endl;
    // ============================================================

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



// 辅助函数：构建旋转矩阵 (直接复用你提供的逻辑或引用头文件)
// 如果项目中已有 general_utils::build_rotation，则直接使用，无需重复定义
// inline torch::Tensor build_rotation(torch::Tensor &r) { ... } 

void GaussianModelLine::densifyAndSplitWithLineAwareness(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent,
    int N)
{
    using namespace torch::indexing;

    int n_init_points = this->getXYZ().size(0);
    
    // =========================================================================
    // 1. 筛选 (Selection): 找出梯度大且 Scaling 大的点
    // =========================================================================
    auto padded_grad = torch::zeros({n_init_points}, torch::TensorOptions().device(device_type_));
    padded_grad.slice(/*dim=*/0L, /*start=*/0, /*end=*/grads.size(0)).copy_(grads.squeeze());
    
    auto selected_mask = torch::where(padded_grad >= grad_threshold, true, false);
    selected_mask = torch::logical_and(
        selected_mask,
        std::get<0>(torch::max(this->getScalingActivation(), /*dim=*/1)) > percentDense() * scene_extent
    );

    int64_t M = selected_mask.sum().item<int64_t>();
    if (M == 0) return;

    // =========================================================================
    // 2. 数据准备 (Data Preparation)
    // =========================================================================
    // 提取被选中的数据
    auto xyz_sel = this->getXYZ().index({selected_mask});
    auto scale_sel = this->getScalingActivation().index({selected_mask});
    auto rot_sel = this->rotation_.index({selected_mask});
    
    // [Safety] 强制确保 Rotation 维度正确 [M, 4]
    if (rot_sel.dim() != 2 || rot_sel.size(1) != 4) {
        rot_sel = rot_sel.reshape({-1, 4}).contiguous();
    }
    
    // 提取线特征属性
    auto is_line_sel = this->is_line_.index({selected_mask});
    auto line_dir_sel = this->line_dir_w_.index({selected_mask});

    // 计算旋转矩阵 R [M, 3, 3]
    auto rots = general_utils::build_rotation(rot_sel);

    // =========================================================================
    // 3. 分类处理 (Branching): Point vs Line
    // =========================================================================
    // 通过 Scaling 形状辅助判断 (Line 通常长宽比很大)
    auto sorted_scale = std::get<0>(scale_sel.sort(1, true)); // 降序
    auto geom_is_line = sorted_scale.index({Slice(), 0}) / (sorted_scale.index({Slice(), 1}) + 1e-6) > 4.0f;
    
    // 最终分类掩码 (相对于 selected_mask 的子集)
    auto mask_line_subset = torch::logical_and(is_line_sel, geom_is_line);
    auto mask_point_subset = ~mask_line_subset;

    // 容器
    std::vector<torch::Tensor> list_xyz, list_scale, list_rot;
    std::vector<torch::Tensor> list_is_line, list_line_dir;
    // 辅助索引 (用于从原始 sel 数据中提取 Feature 等)
    std::vector<torch::Tensor> list_indices_in_sel; 

    auto opts = xyz_sel.options();

    // -------------------------------------------------------------------------
    // 分支 A: 普通点 (Points) - 原始逻辑 (3D 采样, 全局缩小)
    // -------------------------------------------------------------------------
    if (mask_point_subset.any().item<bool>()) {
        auto xyz_p = xyz_sel.index({mask_point_subset});       // [Mp, 3]
        auto scale_p = scale_sel.index({mask_point_subset});   // [Mp, 3]
        auto rots_p = rots.index({mask_point_subset});         // [Mp, 3, 3]
        auto rot_raw_p = rot_sel.index({mask_point_subset});   // [Mp, 4]

        int64_t Mp = xyz_p.size(0);

        // 3D 随机采样: 均值为0，标准差为 scale_p
        // [Mp, N, 3] -> [Mp*N, 3]
        auto stds = scale_p.repeat({N, 1}); 
        auto means = torch::zeros({stds.size(0), 3}, opts);
        auto samples = at::normal(means, stds); 

        // 旋转并平移: P_new = P_old + R * sample
        auto rots_repeated = rots_p.repeat({N, 1, 1}); // [Mp*N, 3, 3]
        auto new_xyz_p = torch::bmm(rots_repeated, samples.unsqueeze(-1)).squeeze(-1) + xyz_p.repeat({N, 1});
        
        // Scaling: 所有轴都缩小
        auto new_scale_p = torch::log(scale_p.repeat({N, 1}) / (0.8 * N));
        
        // Rotation: 简单复制
        auto new_rot_p = rot_raw_p.repeat({N, 1}); // [Mp*N, 4]

        // 存入列表
        list_xyz.push_back(new_xyz_p);
        list_scale.push_back(new_scale_p);
        list_rot.push_back(new_rot_p);
        
        // 结构属性
        list_is_line.push_back(torch::zeros({Mp * N}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(torch::zeros({Mp * N, 3}, opts)); // Dummy direction

        // 记录索引，用于后续提取 Features/Opacity
        list_indices_in_sel.push_back(torch::nonzero(mask_point_subset).squeeze(1).repeat({N}));
    }

    // -------------------------------------------------------------------------
    // 分支 B: 线特征 (Lines) - 新逻辑 (1D 采样, 仅缩小长轴)
    // -------------------------------------------------------------------------
    if (mask_line_subset.any().item<bool>()) {
        auto xyz_l = xyz_sel.index({mask_line_subset});       // [Ml, 3]
        auto scale_l = scale_sel.index({mask_line_subset});   // [Ml, 3]
        auto rots_l = rots.index({mask_line_subset});         // [Ml, 3, 3]
        auto rot_raw_l = rot_sel.index({mask_line_subset});   // [Ml, 4]
        auto dir_l = line_dir_sel.index({mask_line_subset});  // [Ml, 3]

        int64_t Ml = xyz_l.size(0);
        const int axis = 0; // 假设局部 X 轴是线方向 (Long axis)

        // 1D 随机采样: 只在 X 轴 (长轴) 上采样
        // 构造采样标准差: X轴用 scale_x, Y/Z轴设为 0
        auto scale_repeated = scale_l.repeat({N, 1}); // [Ml*N, 3]
        auto stds_1d = torch::zeros_like(scale_repeated);
        stds_1d.index_put_({Slice(), axis}, scale_repeated.index({Slice(), axis})); // 只填充 X 列

        auto means = torch::zeros({stds_1d.size(0), 3}, opts);
        auto samples = at::normal(means, stds_1d); // 采样结果: [noise_x, 0, 0]

        // 旋转并平移
        auto rots_repeated = rots_l.repeat({N, 1, 1});
        auto new_xyz_l = torch::bmm(rots_repeated, samples.unsqueeze(-1)).squeeze(-1) + xyz_l.repeat({N, 1});

        // Scaling: 只缩小 X 轴 (长度变短)，Y/Z (粗细) 保持不变
        // new_s_x = s_x / (0.8 * N)
        // new_s_y = s_y
        auto new_scale_act = scale_l.repeat({N, 1});
        new_scale_act.index_put_({Slice(), axis}, new_scale_act.index({Slice(), axis}) / (0.8 * N));
        auto new_scale_l = torch::log(new_scale_act);

        // Rotation: 简单复制 (方向不变)
        auto new_rot_l = rot_raw_l.repeat({N, 1});

        // Line Dir: 复制
        auto new_dir_l = dir_l.repeat({N, 1});
        new_dir_l = new_dir_l / (new_dir_l.norm(2, 1, true) + 1e-6);

        // 存入列表
        list_xyz.push_back(new_xyz_l);
        list_scale.push_back(new_scale_l);
        list_rot.push_back(new_rot_l);
        
        list_is_line.push_back(torch::ones({Ml * N}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(new_dir_l);

        list_indices_in_sel.push_back(torch::nonzero(mask_line_subset).squeeze(1).repeat({N}));
    }

    if (list_xyz.empty()) return;

    // =========================================================================
    // 4. 合并数据 (Concatenate)
    // =========================================================================
    auto new_xyz = torch::cat(list_xyz, 0);
    auto new_scaling = torch::cat(list_scale, 0);
    auto new_rotation = torch::cat(list_rot, 0);
    auto new_is_line = torch::cat(list_is_line, 0);
    auto new_line_dir_w = torch::cat(list_line_dir, 0);
    
    // 使用索引提取对应的 Features 和 Opacity (避免重复写逻辑)
    auto combined_indices = torch::cat(list_indices_in_sel, 0); // [Total_New_N]
    
    // 从 selected_mask 的原始数据中提取，并复制 N 份
    // 注意：features 等原始数据需要先用 selected_mask 筛选出 M 个，再用 combined_indices 提取
    auto feat_dc_sel = this->features_dc_.index({selected_mask});
    auto feat_rst_sel = this->features_rest_.index({selected_mask});
    auto opac_sel = this->opacity_.index({selected_mask});
    auto iter_sel = this->exist_since_iter_.index({selected_mask});

    auto new_features_dc = feat_dc_sel.index({combined_indices});
    auto new_features_rest = feat_rst_sel.index({combined_indices});
    auto new_opacity = opac_sel.index({combined_indices});
    auto new_exist_since_iter = iter_sel.index({combined_indices});

    // =========================================================================
    // 5. 调用 Postfix (加入优化器)
    // =========================================================================
    this->densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_since_iter,
        new_is_line,
        new_line_dir_w
    );

    // =========================================================================
    // 6. 删除旧点 (Prune)
    // =========================================================================
    auto prune_filter = torch::cat({
        selected_mask, // 原始被选中的点被删除 (因为已经分裂成了新的)
        torch::zeros({new_xyz.size(0)}, torch::TensorOptions().device(device_type_).dtype(torch::kBool))
    });
    this->prunePointsWithLineAwareness(prune_filter);
}

#endif

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

void GaussianModelLine::densifyAndSplitWithLineAwareness(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent,
    int N)
{
    using namespace torch::indexing;

    const int64_t n_init = this->getXYZ().size(0);
    if (n_init == 0) return;

    // =========================================================================
    // [DEBUG 1] 检查函数入口时的 Rotation 状态
    // =========================================================================
    std::cerr << "\n[DEBUG SPLIT] Step 1: Entry" << std::endl;
    std::cerr << "  > Global Rotation Shape: " << this->rotation_.sizes() << std::endl;
    std::cerr << "  > Global Scaling Shape:  " << this->getScalingActivation().sizes() << std::endl;

    // 如果这里 Rotation 的第1维是 3，说明在上一帧（Clone步骤）就被 Scaling 覆盖了
    if (this->rotation_.dim() == 2 && this->rotation_.size(1) == 3) {
        std::cerr << "\033[1;31m[FATAL] Rotation is 3D at entry! It was corrupted in previous step.\033[0m" << std::endl;
    }

    // =========================================================================
    // 1. Selection
    // =========================================================================
    auto padded_grad = torch::zeros({n_init}, torch::TensorOptions().device(device_type_).dtype(grads.dtype()));
    padded_grad.slice(0, 0, grads.size(0)).copy_(grads.squeeze());

    auto selected_mask = padded_grad >= grad_threshold;
    selected_mask = torch::logical_and(
        selected_mask,
        std::get<0>(torch::max(this->getScalingActivation(), 1)) > percentDense() * scene_extent
    );

    const int64_t M = selected_mask.sum().item<int64_t>();
    std::cerr << "[DEBUG SPLIT] Step 2: Selection. Selected M = " << M << std::endl;
    if (M == 0) return;

    // =========================================================================
    // 2. Data Preparation
    // =========================================================================
    auto xyz_sel = this->getXYZ().index({selected_mask});
    auto scale_sel = this->getScalingActivation().index({selected_mask});
    
    // [Safety] 强制确保 Rotation 维度正确 [M, 4]
    auto rot_sel = this->rotation_.index({selected_mask});

    // =========================================================================
    // [DEBUG 2] 检查提取出来的 rot_sel 形状
    // =========================================================================
    std::cerr << "[DEBUG SPLIT] Step 3: Indexing" << std::endl;
    std::cerr << "  > rot_sel Shape: " << rot_sel.sizes() << " (Numel: " << rot_sel.numel() << ")" << std::endl;

    if (rot_sel.dim() != 2 || rot_sel.size(1) != 4) {
        std::cerr << "[Warning] Fixing rot_sel shape in Split: " << rot_sel.sizes() << std::endl;
        rot_sel = rot_sel.reshape({-1, 4}).contiguous();
    }

    auto is_line_sel = this->is_line_.index({selected_mask});
    auto line_dir_sel = this->line_dir_w_.index({selected_mask});

    // 计算旋转矩阵 R [M, 3, 3]
    //auto rots = general_utils::build_rotation(rot_sel);
    auto rot_sel_clone = rot_sel.clone();
    auto rots = general_utils::build_rotation(rot_sel_clone);

    // =========================================================================
    // 3. Classification (Point vs Line)
    // =========================================================================
    auto sorted_scale = std::get<0>(scale_sel.sort(1, true));
    auto geom_is_line = sorted_scale.index({Slice(), 0}) / (sorted_scale.index({Slice(), 1}) + 1e-6) > 4.0f;
    
    auto mask_line_subset = torch::logical_and(is_line_sel, geom_is_line);
    auto mask_point_subset = ~mask_line_subset;

    // Containers
    std::vector<torch::Tensor> list_xyz, list_scale, list_rot;
    std::vector<torch::Tensor> list_is_line, list_line_dir;
    std::vector<torch::Tensor> list_indices_in_sel; 

    auto opts = xyz_sel.options();

    // -------------------------------------------------------------------------
    // Branch A: Point Split (3D Sampling)
    // -------------------------------------------------------------------------
    if (mask_point_subset.any().item<bool>()) {
        auto xyz_p = xyz_sel.index({mask_point_subset});
        auto scale_p = scale_sel.index({mask_point_subset});
        auto rots_p = rots.index({mask_point_subset});
        auto rot_p = rot_sel.index({mask_point_subset}); // [Mp, 4]

        // [DEBUG]
        std::cerr << "  > [Point Branch] rot_p before reshape: " << rot_p.sizes() << std::endl;

        // 再次确保维度安全
        if (rot_p.dim() == 1) rot_p = rot_p.reshape({-1, 4});

        int64_t Mp = xyz_p.size(0);

        // 3D 随机采样 [Mp*N, 3]
        auto stds = scale_p.repeat({N, 1}); 
        auto means = torch::zeros({stds.size(0), 3}, opts);
        auto samples = at::normal(means, stds); 

        // P_new = P_old + R * sample
        auto new_xyz_p = torch::bmm(rots_p.repeat({N, 1, 1}), samples.unsqueeze(-1)).squeeze(-1) + xyz_p.repeat({N, 1});
        auto new_scale_p = torch::log(scale_p.repeat({N, 1}) / (0.8 * N));
        
        // [关键修复]: Rotation 必须重复 N 次，和 XYZ 的数量对齐！
        // 之前的 Bug 是因为这里漏了 .repeat({N, 1})
        auto new_rot_p = rot_p.repeat({N, 1}); 

        list_xyz.push_back(new_xyz_p);
        list_scale.push_back(new_scale_p);
        list_rot.push_back(new_rot_p);
        
        list_is_line.push_back(torch::zeros({Mp * N}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(torch::zeros({Mp * N, 3}, opts));
        list_indices_in_sel.push_back(torch::nonzero(mask_point_subset).squeeze(1).repeat({N}));
    }

    // -------------------------------------------------------------------------
    // Branch B: Line Split (1D Sampling along Axis)
    // -------------------------------------------------------------------------
    if (mask_line_subset.any().item<bool>()) {
        auto xyz_l = xyz_sel.index({mask_line_subset});
        auto scale_l = scale_sel.index({mask_line_subset});
        auto rots_l = rots.index({mask_line_subset});
        auto rot_l = rot_sel.index({mask_line_subset}); // [Ml, 4]
        auto dir_l = line_dir_sel.index({mask_line_subset});

        if (rot_l.dim() == 1) rot_l = rot_l.reshape({-1, 4});

        int64_t Ml = xyz_l.size(0);
        const int axis = 0; // 局部 X 轴是线方向

        // 1D 随机采样
        auto scale_repeated = scale_l.repeat({N, 1}); 
        auto stds_1d = torch::zeros_like(scale_repeated);
        stds_1d.index_put_({Slice(), axis}, scale_repeated.index({Slice(), axis})); 

        auto means = torch::zeros({stds_1d.size(0), 3}, opts);
        auto samples = at::normal(means, stds_1d); 

        auto new_xyz_l = torch::bmm(rots_l.repeat({N, 1, 1}), samples.unsqueeze(-1)).squeeze(-1) + xyz_l.repeat({N, 1});
        
        // Scale 仅缩小长轴
        auto new_scale_act = scale_l.repeat({N, 1});
        new_scale_act.index_put_({Slice(), axis}, new_scale_act.index({Slice(), axis}) / (0.8 * N));
        auto new_scale_l = torch::log(new_scale_act);

        // [关键修复]: Rotation 必须重复 N 次！
        auto new_rot_l = rot_l.repeat({N, 1});

        auto new_dir_l = dir_l.repeat({N, 1});
        new_dir_l = new_dir_l / (new_dir_l.norm(2, 1, true) + 1e-6);

        list_xyz.push_back(new_xyz_l);
        list_scale.push_back(new_scale_l);
        list_rot.push_back(new_rot_l);
        
        list_is_line.push_back(torch::ones({Ml * N}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(new_dir_l);
        list_indices_in_sel.push_back(torch::nonzero(mask_line_subset).squeeze(1).repeat({N}));
    }

    if (list_xyz.empty()) return;

    // =========================================================================
    // 4. Concatenation
    // =========================================================================
    auto new_xyz = torch::cat(list_xyz, 0);
    auto new_scaling = torch::cat(list_scale, 0);
    auto new_rotation = torch::cat(list_rot, 0); // 应该是 [TotalNew, 4]

    // =========================================================================
    // [DEBUG 3] 检查合并后的数据维度
    // =========================================================================
    std::cerr << "[DEBUG SPLIT] Step 4: Concatenation" << std::endl;
    std::cerr << "  > new_xyz Shape:      " << new_xyz.sizes() << std::endl;
    std::cerr << "  > new_scaling Shape:  " << new_scaling.sizes() << std::endl;
    std::cerr << "  > new_rotation Shape: " << new_rotation.sizes() << std::endl;
    
    // ================= [DEBUG PROBE: 致命错误拦截] =================
    // 检查 XYZ 行数和 Rotation 行数是否一致
    if (new_xyz.size(0) != new_rotation.size(0)) {
        std::cerr << "\033[1;31m[FATAL ERROR in Split] Mismatch detected!\033[0m" << std::endl;
        std::cerr << "  > XYZ Rows: " << new_xyz.size(0) << std::endl;
        std::cerr << "  > Rot Rows: " << new_rotation.size(0) << std::endl;
        std::cerr << "  > N (Split Factor): " << N << std::endl;
        // 强制返回，防止错误的维度导致 Postfix 崩溃
        return; 
    }

    // 检查 Rotation 维度是否正确
    if (new_rotation.dim() != 2 || new_rotation.size(1) != 4) {
        std::cerr << "[Auto-Fix] Split generated bad rotation: " << new_rotation.sizes() << ". Fixing..." << std::endl;
        new_rotation = new_rotation.reshape({-1, 4}).contiguous();
    }
    
    // 你的 Debug Log
    std::cerr << "[DEBUG FLOW] Inside densifyAndSplit. Prepared tensors:" << std::endl;
    std::cerr << "  > new_rotation list size: " << list_rot.size() << std::endl;
    if (!list_rot.empty()) {
        std::cerr << "  > first tensor in rot list: " << list_rot[0].sizes() << std::endl;
    }
    std::cerr << "  > Final new_rotation shape: " << new_rotation.sizes() << std::endl;
    // ============================================================

    auto combined_indices = torch::cat(list_indices_in_sel, 0); 
    
    auto new_features_dc = this->features_dc_.index({selected_mask}).index({combined_indices});
    auto new_features_rest = this->features_rest_.index({selected_mask}).index({combined_indices});
    auto new_opacity = this->opacity_.index({selected_mask}).index({combined_indices});
    auto new_exist_iter = this->exist_since_iter_.index({selected_mask}).index({combined_indices});
    auto new_is_line = torch::cat(list_is_line, 0);
    auto new_line_dir_w = torch::cat(list_line_dir, 0);

    // =========================================================================
    // 5. Postfix
    // =========================================================================
    this->densificationPostfixWithLineAwareness(
        new_xyz, new_features_dc, new_features_rest, new_opacity,
        new_scaling, new_rotation, new_exist_iter,
        new_is_line, new_line_dir_w
    );

    // =========================================================================
    // 6. Prune
    // =========================================================================
    auto prune_filter = torch::cat({
        selected_mask, 
        torch::zeros({new_xyz.size(0)}, torch::TensorOptions().device(device_type_).dtype(torch::kBool))
    });
    this->prunePointsWithLineAwareness(prune_filter);
}

void GaussianModelLine::densifyAndCloneWithLineAwareness(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent)
{
    using namespace torch::indexing;

    // =========================================================================
    // 1. Selection
    // =========================================================================
    auto grad_norm = torch::frobenius_norm(grads, /*dim=*/-1);
    auto selected_mask = grad_norm >= grad_threshold;
    selected_mask = torch::logical_and(
        selected_mask,
        std::get<0>(torch::max(this->getScalingActivation(), 1)) <= percentDense() * scene_extent
    );

    const int64_t M = selected_mask.sum().item<int64_t>();
    if (M == 0) return;

    // =========================================================================
    // 2. Data Preparation
    // =========================================================================
    auto xyz_sel = this->getXYZ().index({selected_mask});
    auto scale_sel = this->getScalingActivation().index({selected_mask});
    
    // [Safety] 强制确保 Rotation 维度正确 [M, 4]
    auto rot_sel = this->rotation_.index({selected_mask});
    if (rot_sel.dim() != 2 || rot_sel.size(1) != 4) {
        rot_sel = rot_sel.reshape({-1, 4}).contiguous();
    }

    auto is_line_sel = this->is_line_.index({selected_mask});
    auto line_dir_sel = this->line_dir_w_.index({selected_mask});
    auto rot_sel_clone = rot_sel.clone();
    auto rots = general_utils::build_rotation(rot_sel_clone);

    // =========================================================================
    // 3. Classification
    // =========================================================================
    auto sorted_scale = std::get<0>(scale_sel.sort(1, true));
    auto geom_is_line = sorted_scale.index({Slice(), 0}) / (sorted_scale.index({Slice(), 1}) + 1e-6) > 4.0f;
    auto mask_line_subset = torch::logical_and(is_line_sel, geom_is_line);
    auto mask_point_subset = ~mask_line_subset;

    // Containers
    std::vector<torch::Tensor> list_xyz, list_scale, list_rot;
    std::vector<torch::Tensor> list_is_line, list_line_dir;
    std::vector<torch::Tensor> list_indices_in_sel; 
    auto opts = xyz_sel.options();

    // -------------------------------------------------------------------------
    // Branch A: Point Clone
    // -------------------------------------------------------------------------
    if (mask_point_subset.any().item<bool>()) {
        auto xyz_p = xyz_sel.index({mask_point_subset});
        auto scale_p = scale_sel.index({mask_point_subset});
        auto rot_p = rot_sel.index({mask_point_subset});
        if (rot_p.dim() == 1) rot_p = rot_p.reshape({-1, 4});

        list_xyz.push_back(xyz_p); 
        list_scale.push_back(torch::log(scale_p)); 
        list_rot.push_back(rot_p); // Clone 不需要 repeat，直接由 copy

        int64_t Mp = xyz_p.size(0);
        list_is_line.push_back(torch::zeros({Mp}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(torch::zeros({Mp, 3}, opts));
        list_indices_in_sel.push_back(torch::nonzero(mask_point_subset).squeeze(1));
    }

    // -------------------------------------------------------------------------
    // Branch B: Line Clone
    // -------------------------------------------------------------------------
    if (mask_line_subset.any().item<bool>()) {
        auto xyz_l = xyz_sel.index({mask_line_subset});
        auto scale_l = scale_sel.index({mask_line_subset});
        auto rot_l = rot_sel.index({mask_line_subset});
        if (rot_l.dim() == 1) rot_l = rot_l.reshape({-1, 4});
        
        auto rots_l = rots.index({mask_line_subset}); 
        auto dir_l = line_dir_sel.index({mask_line_subset});

        int64_t Ml = xyz_l.size(0);
        const int axis = 0; 
        auto sigma_par = scale_l.index({Slice(), axis}).unsqueeze(1) * 0.1f; 
        auto eps = at::normal(torch::zeros({Ml, 1}, opts), sigma_par);
        auto local_offset = torch::zeros({Ml, 3}, opts);
        local_offset.index_put_({Slice(), axis}, eps.squeeze(1));

        auto new_xyz_l = torch::bmm(rots_l, local_offset.unsqueeze(-1)).squeeze(-1) + xyz_l;

        list_xyz.push_back(new_xyz_l);
        list_scale.push_back(torch::log(scale_l)); 
        list_rot.push_back(rot_l); // Clone 保持 Rotation 不变

        list_is_line.push_back(torch::ones({Ml}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(dir_l);
        list_indices_in_sel.push_back(torch::nonzero(mask_line_subset).squeeze(1));
    }

    if (list_xyz.empty()) return;

    // =========================================================================
    // 4. Concatenation
    // =========================================================================
    auto new_xyz = torch::cat(list_xyz, 0);
    auto new_scaling = torch::cat(list_scale, 0);
    auto new_rotation = torch::cat(list_rot, 0);
    
    // [CRITICAL DEBUG]
    if (new_xyz.size(0) != new_rotation.size(0)) {
        std::cerr << "\033[1;31m[FATAL ERROR in Clone] Mismatch detected!\033[0m" << std::endl;
        std::cerr << "  > XYZ Rows: " << new_xyz.size(0) << std::endl;
        std::cerr << "  > Rot Rows: " << new_rotation.size(0) << std::endl;
        return; 
    }

    if (new_rotation.dim() != 2 || new_rotation.size(1) != 4) {
        new_rotation = new_rotation.reshape({-1, 4}).contiguous();
    }

    auto combined_indices = torch::cat(list_indices_in_sel, 0);
    auto new_features_dc = this->features_dc_.index({selected_mask}).index({combined_indices});
    auto new_features_rest = this->features_rest_.index({selected_mask}).index({combined_indices});
    auto new_opacity = this->opacity_.index({selected_mask}).index({combined_indices});
    auto new_exist_iter = this->exist_since_iter_.index({selected_mask}).index({combined_indices});
    auto new_is_line = torch::cat(list_is_line, 0);
    auto new_line_dir_w = torch::cat(list_line_dir, 0);

    // ================= [DEBUG PROBE 3] =================
    std::cerr << "[DEBUG FLOW] Inside densifyAndClone. Prepared tensors:" << std::endl;
    std::cerr << "  > Final new_rotation shape: " << new_rotation.sizes() << std::endl;
    // ============================================================

    // =========================================================================
    // 5. Postfix
    // =========================================================================
    this->densificationPostfixWithLineAwareness(
        new_xyz, new_features_dc, new_features_rest, new_opacity,
        new_scaling, new_rotation, new_exist_iter,
        new_is_line, new_line_dir_w
    );
}

#if 0

void GaussianModelLine::densifyAndCloneWithLineAwareness(
    torch::Tensor& grads,
    float grad_threshold,
    float scene_extent)
{
    using namespace torch::indexing;

    // =========================================================================
    // 1. 筛选 (Selection): 找出梯度大且 Scaling 小的点 (需要加密)
    // =========================================================================
    auto grad_norm = torch::frobenius_norm(grads, /*dim=*/-1);
    auto selected_mask = grad_norm >= grad_threshold;

    // 与 Split 不同，Clone 针对的是小尺度的点
    selected_mask = torch::logical_and(
        selected_mask,
        std::get<0>(torch::max(this->getScalingActivation(), 1)) <= percentDense() * scene_extent
    );

    const int64_t M = selected_mask.sum().item<int64_t>();
    if (M == 0) return;

    // =========================================================================
    // 2. 数据准备 (Data Preparation)
    // =========================================================================
    auto xyz_sel = this->getXYZ().index({selected_mask});
    auto scale_sel = this->getScalingActivation().index({selected_mask});
    
    // [Safety] 强制确保 Rotation 维度正确 [M, 4]
    auto rot_sel = this->rotation_.index({selected_mask});
    if (rot_sel.dim() != 2 || rot_sel.size(1) != 4) {
        rot_sel = rot_sel.reshape({-1, 4}).contiguous();
    }

    auto is_line_sel = this->is_line_.index({selected_mask});
    auto line_dir_sel = this->line_dir_w_.index({selected_mask});

    // 计算旋转矩阵 R [M, 3, 3]
    auto rots = general_utils::build_rotation(rot_sel);

    // =========================================================================
    // 3. 分类处理 (Branching): Point vs Line
    // =========================================================================
    // 同样使用长宽比辅助判断
    auto sorted_scale = std::get<0>(scale_sel.sort(1, true));
    auto geom_is_line = sorted_scale.index({Slice(), 0}) / (sorted_scale.index({Slice(), 1}) + 1e-6) > 4.0f;

    auto mask_line_subset = torch::logical_and(is_line_sel, geom_is_line);
    auto mask_point_subset = ~mask_line_subset;

    // 容器
    std::vector<torch::Tensor> list_xyz, list_scale, list_rot;
    std::vector<torch::Tensor> list_is_line, list_line_dir;
    // 辅助索引 (用于提取 Features/Opacity)
    std::vector<torch::Tensor> list_indices_in_sel; 

    auto opts = xyz_sel.options();

    // -------------------------------------------------------------------------
    // 分支 A: 普通点 Clone - 原地复制
    // -------------------------------------------------------------------------
    if (mask_point_subset.any().item<bool>()) {
        auto xyz_p = xyz_sel.index({mask_point_subset});
        auto scale_p = scale_sel.index({mask_point_subset});
        auto rot_p = rot_sel.index({mask_point_subset});

        // 简单直接复制，不改变位置（或者加极小的噪声防止数值重叠）
        // Scale 和 Rotation 保持不变
        list_xyz.push_back(xyz_p); 
        list_scale.push_back(torch::log(scale_p)); // 注意：存的是 log scale
        list_rot.push_back(rot_p);

        // 结构属性
        int64_t Mp = xyz_p.size(0);
        list_is_line.push_back(torch::zeros({Mp}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(torch::zeros({Mp, 3}, opts));

        list_indices_in_sel.push_back(torch::nonzero(mask_point_subset).squeeze(1));
    }

    // -------------------------------------------------------------------------
    // 分支 B: 线特征 Clone - 沿线微扰
    // -------------------------------------------------------------------------
    if (mask_line_subset.any().item<bool>()) {
        auto xyz_l = xyz_sel.index({mask_line_subset});
        auto scale_l = scale_sel.index({mask_line_subset});
        auto rot_l = rot_sel.index({mask_line_subset});
        auto rots_l = rots.index({mask_line_subset}); // [Ml, 3, 3]
        auto dir_l = line_dir_sel.index({mask_line_subset});

        int64_t Ml = xyz_l.size(0);
        const int axis = 0; // 局部 X 轴是线方向

        // [策略] Clone 时，我们希望增加线的密度。
        // 可以让新点沿着线的方向稍微偏移一点点（比如 10% 的长度），填补空隙。
        // 偏移量：在 [-0.1*len, 0.1*len] 之间随机
        auto sigma_par = scale_l.index({Slice(), axis}).unsqueeze(1) * 0.1f; 
        auto eps = at::normal(torch::zeros({Ml, 1}, opts), sigma_par);
        
        // 构造局部偏移 [noise_x, 0, 0]
        auto local_offset = torch::zeros({Ml, 3}, opts);
        local_offset.index_put_({Slice(), axis}, eps.squeeze(1));

        // 变换到世界坐标: P_new = P_old + R * local_offset
        auto new_xyz_l = torch::bmm(rots_l, local_offset.unsqueeze(-1)).squeeze(-1) + xyz_l;

        list_xyz.push_back(new_xyz_l);
        list_scale.push_back(torch::log(scale_l)); // 保持 Scale 不变
        list_rot.push_back(rot_l);                 // 保持 Rotation 不变 (方向一致)

        list_is_line.push_back(torch::ones({Ml}, opts.dtype(torch::kBool)));
        list_line_dir.push_back(dir_l);

        list_indices_in_sel.push_back(torch::nonzero(mask_line_subset).squeeze(1));
    }

    if (list_xyz.empty()) return;

    // =========================================================================
    // 4. 合并数据 (Concatenate)
    // =========================================================================
    auto new_xyz = torch::cat(list_xyz, 0);
    auto new_scaling = torch::cat(list_scale, 0);
    auto new_rotation = torch::cat(list_rot, 0);
    auto new_is_line = torch::cat(list_is_line, 0);
    auto new_line_dir_w = torch::cat(list_line_dir, 0);

    // 再次强制维度检查 (兜底)
    if (new_rotation.dim() != 2 || new_rotation.size(1) != 4) {
        new_rotation = new_rotation.reshape({-1, 4}).contiguous();
    }

    // 提取属性 (Features, Opacity)
    auto combined_indices = torch::cat(list_indices_in_sel, 0);
    
    auto feat_dc_sel = this->features_dc_.index({selected_mask});
    auto feat_rst_sel = this->features_rest_.index({selected_mask});
    auto opac_sel = this->opacity_.index({selected_mask});
    auto iter_sel = this->exist_since_iter_.index({selected_mask});

    auto new_features_dc = feat_dc_sel.index({combined_indices});
    auto new_features_rest = feat_rst_sel.index({combined_indices});
    auto new_opacity = opac_sel.index({combined_indices});
    auto new_exist_iter = iter_sel.index({combined_indices});

    // ================= [DEBUG PROBE 3: Clone 检查] =================
    std::cerr << "[DEBUG FLOW] Inside densifyAndClone. Prepared tensors:" << std::endl;
    std::cerr << "  > Final new_rotation shape: " << new_rotation.sizes() << std::endl;
    // ============================================================

    // =========================================================================
    // 5. Postfix (加入新点)
    // =========================================================================
    this->densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacity,
        new_scaling,
        new_rotation,
        new_exist_iter,
        new_is_line,
        new_line_dir_w
    );
    
    // 注意：Clone 操作**不删除**旧点，所以没有 prunePoints 步骤。
    // 我们是在增加密度，不是拆分。
    
    
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
    
    // [CRITICAL FIX] 强制确保 Rotation 是 [M, 4]
    auto rot_sel = this->rotation_.index({selected_mask});                
    if (rot_sel.dim() != 2 || rot_sel.size(1) != 4) {
        std::cerr << "[Warning] Fixing rot_sel shape in Clone Input: " << rot_sel.sizes() << std::endl;
        rot_sel = rot_sel.reshape({-1, 4}).contiguous();
    }

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

    // Final line mask
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
        
        // [Safety Check] 确保 rot_p 是 [Mp, 4]
        auto rot_p = rot_sel.index({is_point});
        if (rot_p.dim() != 2) rot_p = rot_p.reshape({-1, 4});
        rot_list.push_back(rot_p);

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
    // =========================================================
    if (is_line.any().item<bool>())
    {
        const int axis = 0; // line direction = local x-axis

        auto xyz_l   = xyz_sel.index({is_line});
        auto scale_l = scale_act_sel.index({is_line});
        
        // [Safety Check] 确保 rot_l 是 [Ml, 4]
        auto rot_l   = rot_sel.index({is_line});
        if (rot_l.dim() != 2) rot_l = rot_l.reshape({-1, 4});
        
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
    auto new_rotation     = torch::cat(rot_list, 0); // Should be [N, 4]
    
    // [CRITICAL FIX] 再次强制检查 new_rotation 维度
    // 防止 torch::cat 之后形状依然不对 (例如 list 里都是 1D tensor)
    if (new_rotation.dim() != 2 || new_rotation.size(1) != 4) {
         std::cerr << "[Auto-Fix] Clone generated bad rotation: " << new_rotation.sizes() << ". Fixing..." << std::endl;
         new_rotation = new_rotation.reshape({-1, 4}).contiguous();
    }

    auto new_features_dc  = torch::cat(feat_dc_list, 0);
    auto new_features_rst = torch::cat(feat_rest_list, 0);
    auto new_opacity      = torch::cat(opacity_list, 0);
    auto new_exist_iter   = torch::cat(exist_iter_list, 0);

    auto new_is_line      = torch::cat(new_is_line_list, 0);
    auto new_line_dir_w   = torch::cat(new_line_dir_w_list, 0);

    // ================= [DEBUG PROBE 3: Clone 检查] =================
    std::cerr << "[DEBUG FLOW] Inside densifyAndClone. Prepared tensors:" << std::endl;
    std::cerr << "  > Final new_rotation shape: " << new_rotation.sizes() << std::endl;
    // ============================================================

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

#endif

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
    // 1. Clone
    std::cerr << "[DEBUG] Before Clone: Rot Shape = " << this->rotation_.sizes() << std::endl;
    this->densifyAndCloneWithLineAwareness(grads, max_grad, extent);
    std::cerr << "[DEBUG] After Clone: Rot Shape = " << this->rotation_.sizes() << std::endl;

    const int split_N = 2; // recommended
    std::cerr << "[DEBUG] Before Split: Rot Shape = " << this->rotation_.sizes() << std::endl;
    this->densifyAndSplitWithLineAwareness(grads, max_grad, extent, split_N);
    std::cerr << "[DEBUG] After Split: Rot Shape = " << this->rotation_.sizes() << std::endl;

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

    std::cerr << "\t[ply_header] Type: " << (ply_file.is_binary_file() ? "binary" : "ascii") << std::endl;
    for (const auto & c : ply_file.get_comments())
        std::cerr << "\t[ply_header] Comment: " << c << std::endl;
    for (const auto & c : ply_file.get_info())
        std::cerr << "\t[ply_header] Info: " << c << std::endl;

    for (const auto &e : ply_file.get_elements()) {
        std::cerr << "\t[ply_header] element: " << e.name << " (" << e.size << ")" << std::endl;
        for (const auto &p : e.properties) {
            std::cerr << "\t[ply_header] \tproperty: " << p.name << " (type=" << tinyply::PropertyTable[p.propertyType].str << ")";
            if (p.isList)
                std::cerr << " (list_type=" << tinyply::PropertyTable[p.listType].str << ")";
            std::cerr << std::endl;
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

    if (xyz)     std::cerr << "\tRead " << xyz->count     << " total xyz "     << std::endl;
    if (f_dc)    std::cerr << "\tRead " << f_dc->count    << " total f_dc "    << std::endl;
    if (f_rest)  std::cerr << "\tRead " << f_rest->count  << " total f_rest "  << std::endl;
    if (opacity) std::cerr << "\tRead " << opacity->count << " total opacity " << std::endl;
    if (scales)  std::cerr << "\tRead " << scales->count  << " total scales "  << std::endl;
    if (rot)     std::cerr << "\tRead " << rot->count     << " total rot "     << std::endl;

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
