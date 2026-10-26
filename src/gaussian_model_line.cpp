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
#include <torch/torch.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include "include/rasterize_points.h"
#include "LineMode.h"

namespace {
// LocalMapping point supplementation is NOT ordinary 3DGS densification.
// Record it separately with the explicit call site and before/after counts so
// a total-Gaussian jump cannot be mistaken for an early densify boundary.
struct PcdSupplementGuard
{
    GaussianModelLine* m_;
    const char* kind_;
    int iteration_;
    long long before_;
    long long line_before_;
    PcdSupplementGuard(GaussianModelLine* m, const char* kind, int iteration)
        : m_(m), kind_(kind), iteration_(iteration), before_(0),
          line_before_(0)
    {
        if (m_ && m_->xyz_.defined())
            before_ = m_->xyz_.size(0);
        if (m_ && m_->line_idx_.defined())
            line_before_ = m_->line_idx_.numel();
    }
    ~PcdSupplementGuard()
    {
        const char* d = std::getenv("PHOTO_SLAM_ADAM_TOPO_DIAG_DIR");
        if (!d || !d[0] || !m_ || !m_->xyz_.defined()) return;
        std::filesystem::path path =
            std::filesystem::path(d) / "localmapping_supplement.csv";
        std::filesystem::create_directories(path.parent_path());
        const bool fresh = !std::filesystem::exists(path) ||
                           std::filesystem::file_size(path) == 0;
        std::ofstream out(path, std::ios::app);
        if (fresh)
            out << "iteration,kind,rows_before,rows_after,delta,"
                   "line_rows_before,line_rows_after\n";
        const long long after = m_->xyz_.size(0);
        const long long line_after =
            m_->line_idx_.defined() ? m_->line_idx_.numel() : -1;
        out << iteration_ << ',' << kind_ << ',' << before_ << ',' << after
            << ',' << (after - before_) << ',' << line_before_ << ','
            << line_after << '\n';
    }
};
}

// ---------------------------------------------------------------------------
// perf-stage3: online Gaussian creation-source audit.
// A host-only (no device sync) guard that turns a creation call into one audit
// event. See GaussianModelLine::auditEvent for the accounting basis.
// ---------------------------------------------------------------------------
struct CreationAuditGuard
{
    GaussianModelLine* m_;
    const char* entry_;
    long long before_;
    bool on_;
    CreationAuditGuard(GaussianModelLine* m, const char* entry)
        : m_(m), entry_(entry), before_(0), on_(false)
    {
        m_->auditInitIfNeeded();
        on_ = m_->auditEnabled();
        if (on_) before_ = m_->auditTotal();
    }
    ~CreationAuditGuard()
    {
        if (on_) m_->auditEvent(entry_, m_->auditTotal() - before_, 0);
    }
};

// Line-Gaussian retention ablation knobs (env-configurable, default = baseline).
//   PHOTO_SLAM_LINE_INIT_OPACITY : initial opacity of line-sampled Gaussians (B)
//   PHOTO_SLAM_LINE_MIN_OPACITY  : opacity-prune threshold for line Gaussians (D)
//                                  (-1 = use the point threshold)
//   PHOTO_SLAM_LINE_GRACE_ITER   : iterations after birth during which line
//                                  Gaussians are exempt from opacity pruning (C)
//   PHOTO_SLAM_LINE_LAZY_FACTOR  : lazy(gradient)-prune threshold factor for
//                                  line Gaussians (line grad is ~100x smaller
//                                  than point grad; 0 = never lazy-prune lines)
//   PHOTO_SLAM_LINE_SIZE_EXTENT   : world-size prune factor for lines
//                                  (line world scale > factor*extent pruned;
//                                  lines are elongated by design, so this is
//                                  larger than the 0.1 point factor)
namespace {
float lineInitialOpacity() {
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_INIT_OPACITY");
        return e ? std::atof(e) : 0.1f;
    }();
    return v;
}
float lineMinOpacity() {
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_MIN_OPACITY");
        // Line Gaussians are thin and naturally settle at much lower opacity
        // than point Gaussians (measured opMean ~0.05 vs point threshold 0.1).
        // A dedicated, lower opacity-prune threshold keeps geometrically valid
        // lines alive without touching the point threshold.
        return e ? std::atof(e) : 0.01f;
    }();
    return v;
}
int lineGraceIter() {
    static const int v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_GRACE_ITER");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}
float lineLazyFactor() {
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_LAZY_FACTOR");
        // Gradient-based "lazy" pruning is a point-oriented criterion that does
        // not apply to line Gaussians: their thin anisotropic footprint gives
        // ~100-1000x smaller per-pixel gradient than points by construction, so
        // any non-zero gradient threshold eventually prunes every line over the
        // long training (verified: factor 0.001 -> candidates=0 at loop; 0 ->
        // candidates>0). Lines remain governed by opacity (lineMinOpacity) and
        // size (lineSizeExtentFactor) pruning, so this is not an unconditional
        // exemption. Keep the knob for experiments.
        return e ? std::atof(e) : 0.0f;
    }();
    return v;
}
float lineSizeExtentFactor() {
    static const float v = []() {
        const char* e = std::getenv("PHOTO_SLAM_LINE_SIZE_EXTENT");
        return e ? std::atof(e) : 0.5f;
    }();
    return v;
}
bool linePruneDiag() {
    static const bool v = []() {
        const char* e = std::getenv("PHOTO_SLAM_DEBUG_LINE_PRUNE");
        return e && std::atoi(e) != 0;
    }();
    return v;
}
// Densification strategy for the frozen-geometry mapper ablation.
//   1 (default) = directional: line Gaussians are cloned/split along their own
//                 axis by the line-aware routines.
//   0           = ordinary: every Gaussian (including line-seeded ones) uses the
//                 upstream 3DGS clone/split. Initialisation and pruning are
//                 untouched, so this isolates the densification strategy.
// NOTE: deliberately NOT cached in a static, because the matching
//       mapper ablation replays several conditions inside one process and
//       flips this switch between them.
bool useDirectionalLineDensify() {
    const char* e = std::getenv("PHOTO_SLAM_ABL_DIRECTIONAL_DENSIFY");
    return e == nullptr ? true : (std::atoi(e) != 0);
}

// densify 内部分解（opt-in，PHOTO_SLAM_DENS_PROF=1）：prune / densify / emptyCache。
struct DensifyProfiler {
    bool enabled = false;
    bool inited = false;
    long calls = 0;
    double prune_ms = 0, densify_ms = 0, emptycache_ms = 0;
    void init() {
        if (inited) return;
        inited = true;
        const char* e = std::getenv("PHOTO_SLAM_DENS_PROF");
        enabled = e && std::atoi(e) != 0;
    }
    void print() const {
        if (!enabled || calls == 0) return;
        std::cerr << "[DensProfiler] calls=" << calls
                  << " prune=" << prune_ms / calls << " ms/call"
                  << " densify=" << densify_ms / calls << " ms/call"
                  << " emptyCache=" << emptycache_ms / calls << " ms/call\n";
    }
};
DensifyProfiler g_dens_prof;
} // namespace

void printDensifyProfiler() {
    g_dens_prof.print();
}


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
    // 不再调用 line_mask.any().item<bool>()（CPU-GPU 同步）；无线时布尔索引
    // 得到空张量、sum 为 0，语义不变。

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

torch::Tensor GaussianModelLine::computeVectorizedLineLossRef(float lambda_coherence)
{
    using namespace torch::indexing;

    auto line_mask = this->is_line_; 
    // 基础安全检查。注意：不再调用 line_mask.any().item<bool>()，
    // 否则每次迭代都会强制一次 CPU-GPU 同步。无线高斯时下面的布尔索引
    // 会得到空张量，其 sum() 恒为 0，语义不变。
    if (!this->xyz_init_.defined() || this->xyz_init_.size(0) != this->xyz_.size(0)) {
        return torch::zeros({}, torch::TensorOptions().device(device_type_));
    }
    if (this->line_dir_w_.size(0) != this->xyz_.size(0)) {
        return torch::zeros({}, torch::TensorOptions().device(device_type_));
    }

    // 1. 提取所有线点的当前位置 (P)、初始参考位置 (A) 和单位方向向量 (u)
    auto p_m = this->xyz_.index({line_mask});           // [Nl, 3]
    auto p_ref = this->xyz_init_.index({line_mask});    // [Nl, 3] (作为直线上的一点参考锚点)
    auto d_m = this->line_dir_w_.index({line_mask});    // [Nl, 3] (单位方向向量)

    // 2. 计算向量 AP = P - A
    auto ap = p_m - p_ref; 

    // 3. 计算 AP 在方向 d_m 上的投影向量
    // 公式: proj = (AP · d) * d
    // 使用批量点积 sum(dim=1)
    auto dist_parallel = torch::sum(ap * d_m, /*dim=*/1, /*keepdim=*/true); // [Nl, 1]
    auto projection = dist_parallel * d_m; // [Nl, 3]

    // 4. 计算垂直分量 (即点到直线的垂直误差向量)
    // 公式: e_perp = AP - proj
    auto error_perp = ap - projection; 

    // Huber Loss 参数 delta（通常设为 0.01m 左右）
    float delta = 0.01f; 

    auto dist_perp = torch::norm(error_perp, 2, 1, true);

    // 计算 Huber 损失
    // 当 dist <= delta: 0.5 * dist^2
    // 当 dist >  delta: delta * (dist - 0.5 * delta)
    auto is_small_error = dist_perp <= delta;
    auto loss_small = 0.5 * dist_perp.pow(2);
    auto loss_large = delta * (dist_perp - 0.5 * delta);

    auto huber_loss = torch::where(is_small_error, loss_small, loss_large);

    auto line_loss = lambda_coherence * huber_loss.sum();
    if (ORB_SLAM3::IsLineLossDiag()) {
        // Diagnostic only (PHOTO_SLAM_DEBUG_LINE_LOSS=1).
        std::cerr << "[LINE-LOSS] L_ori=" << line_loss.item().toFloat()
                  << " n_line_gaussians=" << p_m.size(0)
                  << " lambda_coherence=" << lambda_coherence << std::endl;
    }
    return line_loss;

    // 5. 返回垂直距离的平方和损失
    // 只有垂直离开直线的位移会产生 Loss，沿线滑动 Loss 为 0
    //return lambda_coherence * error_perp.pow(2).sum();
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

torch::Tensor GaussianModelLine::getScaling()
{
    // 3DGS 中 scaling_ 存储的是 log 空间的值，激活需用 exp
    return torch::exp(this->scaling_);
}

torch::Tensor GaussianModelLine::getRotationActivation()
{
    // Robust normalization: clamp sum-of-squares BEFORE sqrt so the sqrt
    // backward never sees 0 (sqrt(0) -> Inf -> NaN for a degenerate quaternion).
    auto norm_sq = torch::sum(this->rotation_ * this->rotation_, 1, true).clamp_min(1e-16f);
    return this->rotation_ / torch::sqrt(norm_sq);
}

void GaussianModelLine::normalizeRotationQuaternions(bool in_place)
{
    if (this->rotation_.size(0) == 0) return;
    torch::NoGradGuard no_grad;
    auto norm_sq = torch::sum(this->rotation_ * this->rotation_, 1, true).clamp_min(1e-16f);
    auto normalized = this->rotation_ / torch::sqrt(norm_sq);
    if (in_place) {
        // Append phase: keep parameter identity and Adam moments untouched.
        // Mathematically identical normalization, no tensor replacement.
        this->rotation_.copy_(normalized);
        return;
    }
    this->rotation_ = replaceTensorToOptimizer(normalized, 5);
    this->Tensor_vec_rotation_ = {this->rotation_};
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
    std::vector<long long> initial_ref_kf_ids(static_cast<std::size_t>(num_points), -1LL);
    
    // 1. 初始化 GPU 张量
    torch::Tensor fused_point_cloud = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    torch::Tensor color = torch::zeros(
        {num_points, 3},
        torch::TensorOptions().dtype(torch::kFloat).device(device_type_));
    invalidateLineCache();   // cache version bump (assignment below)
    is_line_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    invalidateLineCache();   // cache version bump (assignment below)
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
        initial_ref_kf_ids[static_cast<std::size_t>(point_idx)] = point.ref_kf_id_;
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
    invalidateLineCache();   // cache version bump (assignment below)
    line_dir_w_ = line_dir_cpu.to(device_type_).contiguous(); // 更新成员变量
    invalidateLineCache();   // cache version bump (assignment below)
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
    // 创建一个不带梯度的初始位置副本
    invalidateLineCache();   // cache version bump (assignment below)
    this->xyz_init_ = this->xyz_.detach().clone();
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

    // Deformation anchor per Gaussian (see Point3D::ref_kf_id_).
    this->ref_kf_id_ = torch::from_blob(
        reinterpret_cast<std::int64_t*>(initial_ref_kf_ids.data()), {num_points},
        torch::TensorOptions().dtype(torch::kLong)).clone().to(device_type_);
    this->coord_event_id_ = torch::zeros(
        {num_points}, torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    this->last_applied_gba_event_ = 0;
    this->coord_event_seq_ = 0;
    long long n_missing_ref = 0;
    for (long long v : initial_ref_kf_ids) if (v < 0) ++n_missing_ref;
    if (n_missing_ref > 0)
        std::cerr << "[RefKf] initial map: " << n_missing_ref << "/" << num_points
                  << " Gaussians have no reference keyframe (reported, not guessed)\n";

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
    invalidateLineCache();   // cache version bump (assignment below)
    is_line_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kBool).device(device_type_));
    invalidateLineCache();   // cache version bump (assignment below)
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
    invalidateLineCache();   // cache version bump (assignment below)
    this->is_line_ = is_line_cpu.to(device_type_);  //cp back to gpu
}
#endif

void GaussianModelLine::increasePcd(
    const std::vector<Point3D>& new_points,
    const int iteration)
{
    const int N = static_cast<int>(new_points.size());
    if (N == 0) return;
    PcdSupplementGuard pcd_guard(this, "localmapping_increasePcd", iteration);
    CreationAuditGuard audit_guard(this, "add");   // perf-stage3: host-only
    creationLogInitIfNeeded();                      // mapper-budget-v1
    if (clog_enabled_) {
        // Line seeds carry a world direction; log it so the replay can rebuild
        // the same anisotropic initialisation without relaxing any threshold.
        auto lx = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32));
        auto lc = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32));
        auto ld = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32));
        auto ax = lx.accessor<float, 2>();
        auto ac = lc.accessor<float, 2>();
        auto ad = ld.accessor<float, 2>();
        for (int i = 0; i < N; ++i) {
            for (int k = 0; k < 3; ++k) {
                ax[i][k] = static_cast<float>(new_points[i].xyz_(k));
                ac[i][k] = static_cast<float>(new_points[i].color_(k));
                ad[i][k] = static_cast<float>(new_points[i].line_dir_(k));
            }
        }
        creationLogAppend(audit_reason_, lx, lc, ld);
    }

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

    auto new_opacity = general_utils::inverse_sigmoid(
        lineInitialOpacity() * torch::ones({N, 1}, torch::TensorOptions().dtype(torch::kFloat32).device(device_type_)));
    auto new_exist_since_iter = torch::full({N}, iteration, torch::TensorOptions().dtype(torch::kInt32).device(device_type_));

    auto new_xyz_init = new_xyz.clone().detach();

    // Per-row deformation anchor (Point3D carries it from the producer).
    std::vector<long long> ref_ids(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i)
        ref_ids[static_cast<std::size_t>(i)] = new_points[i].ref_kf_id_;
    torch::Tensor new_ref_kf_ids = torch::from_blob(
        reinterpret_cast<std::int64_t*>(ref_ids.data()), {N},
        torch::TensorOptions().dtype(torch::kLong)).clone().to(device_type_);

    // 5. Call Postfix (Safe Version)
    densificationPostfixWithLineAwareness(
        new_xyz, new_features_dc, new_features_rest, new_opacity, 
        new_scaling, new_rotation, new_exist_since_iter, 
        new_is_line, new_line_dir_w, new_xyz_init, new_ref_kf_ids
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

void GaussianModelLine::increasePcd(std::vector<float> points, std::vector<float> colors,
                                   const int iteration,
                                   const torch::Tensor& ref_kf_ids)
{
    // auto time1 = std::chrono::steady_clock::now();
    assert(points.size() == colors.size());
    assert(points.size() % 3 == 0);
    auto num_new_points = static_cast<int>(points.size() / 3);
    if (num_new_points == 0)
        return;
    PcdSupplementGuard pcd_guard(this, "localmapping_increasePcd", iteration);
    CreationAuditGuard audit_guard(this, "add");   // perf-stage3: host-only
    creationLogInitIfNeeded();                      // mapper-budget-v1
    if (clog_enabled_) {
        auto lx = torch::from_blob(points.data(), {num_new_points, 3},
                     torch::TensorOptions().dtype(torch::kFloat32)).clone();
        auto lc = torch::from_blob(colors.data(), {num_new_points, 3},
                     torch::TensorOptions().dtype(torch::kFloat32)).clone();
        creationLogAppend(audit_reason_, lx, lc, torch::Tensor());
    }

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
    //std::cerr << "\n[DEBUG increasePcd(Vector)] Preparing to call Postfix. N_new=" << num_new_points << std::endl;
    //std::cerr << "  > new_xyz shape:      " << new_xyz.sizes() << std::endl;
    //std::cerr << "  > new_rotation shape: " << new_rotation.sizes() << " (Expect [N, 4])" << std::endl;
    //std::cerr << "  > new_scaling shape:  " << new_scaling.sizes() << " (Expect [N, 3])" << std::endl;
    //std::cerr << "  > new_features_dc:    " << new_features_dc.sizes() << std::endl;
    
    // 强制检查：如果 Rotation 维度不对，打印红色警告
    if (new_rotation.dim() != 2 || new_rotation.size(1) != 4) {
        std::cerr << "\033[1;31m[CRITICAL ALERT] new_rotation dimension is WRONG inside increasePcd!\033[0m" << std::endl;
        // 尝试自动修复 (虽然之前的 torch::zeros 应该是对的，但以防万一)
        if (new_rotation.numel() == num_new_points * 4) {
             std::cerr << "[Auto-Fix] Reshaping rotation to [N, 4]..." << std::endl;
             new_rotation = new_rotation.view({num_new_points, 4}).contiguous();
        }
    }

    // 在 increasePcd 内部
    auto new_xyz_init = new_xyz.clone().detach();
    torch::Tensor new_ref_kf_ids;
    if (ref_kf_ids.defined())
    {
        new_ref_kf_ids = ref_kf_ids.to(device_type_).to(torch::kLong).flatten();
        TORCH_CHECK(new_ref_kf_ids.size(0) == num_new_points,
                    "increasePcd: ref_kf_ids has ", new_ref_kf_ids.size(0),
                    " rows, expected ", num_new_points);
    }
    else
    {
        new_ref_kf_ids = torch::full(
            {num_new_points}, -1LL,
            torch::TensorOptions().dtype(torch::kLong).device(device_type_));
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
        new_line_dir_w,    // 传入
        new_xyz_init,      // 传入
        new_ref_kf_ids
    );

    c10::cuda::CUDACachingAllocator::emptyCache();
}


void GaussianModelLine::increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors,
                                   const int iteration, const torch::Tensor& ref_kf_ids)
{
// auto time1 = std::chrono::steady_clock::now();
    auto num_new_points = new_point_cloud.size(0);
    if (num_new_points == 0)
        return;
    PcdSupplementGuard pcd_guard(this, "localmapping_increasePcd", iteration);
    CreationAuditGuard audit_guard(this, "add");   // perf-stage3: host-only
    creationLogInitIfNeeded();                      // mapper-budget-v1
    if (clog_enabled_)
        creationLogAppend(audit_reason_, new_point_cloud, new_colors, torch::Tensor());

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

    // 在 increasePcd 内部
    auto new_xyz_init = new_xyz.clone().detach();

    // =================================================================================
    // 【替换】使用带 Line Awareness 的 Postfix
    // =================================================================================
    torch::Tensor new_ref_kf_ids;
    if (ref_kf_ids.defined())
    {
        new_ref_kf_ids = ref_kf_ids.to(device_type_).to(torch::kLong).flatten();
        TORCH_CHECK(new_ref_kf_ids.size(0) == num_new_points,
                    "increasePcd: ref_kf_ids has ", new_ref_kf_ids.size(0),
                    " rows, expected ", num_new_points);
    }
    else
    {
        new_ref_kf_ids = torch::full(
            {num_new_points}, -1LL,
            torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    }
    densificationPostfixWithLineAwareness(
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_exist_since_iter,
        new_is_line,       // Add
        new_line_dir_w,    // Add
        new_xyz_init,      // Add
        new_ref_kf_ids
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
    // geom-sync-v1: T_tensor is M^T (EigenMatrix2TorchTensor + transpose), so
    // slice-based indexing must transpose back first. The previous code used
    // R^T and rotated line_dir_w_ by the inverse rotation.
    torch::Tensor M = T_tensor.transpose(0, 1).contiguous();
    torch::Tensor R = M.slice(0, 0, 3).slice(1, 0, 3);              // true R
    torch::Tensor t = M.slice(0, 0, 3).slice(1, 3, 4).transpose(0, 1); // [1,3]
    invalidateLineCache();   // cache version bump (assignment below)
    this->line_dir_w_ = torch::matmul(this->line_dir_w_, R.transpose(0, 1));
    // The line anchor is a world point on the line: anchor' = s R anchor + t.
    if (this->xyz_init_.defined() && this->xyz_init_.size(0) == this->xyz_.size(0)) {
        this->xyz_init_ = torch::matmul(this->xyz_init_, R.transpose(0, 1)) * s
                          + t.expand({this->xyz_init_.size(0), 3});
        invalidateLineCache();
    }

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

// ---------------------------------------------------------------------------
// geom-sync-v3: reference-keyframe anchored GBA correction.
//
// Every Gaussian carries ref_kf_id_ (the keyframe whose pose was current when
// the row was created) and coord_event_id_ (the last GBA event id whose
// coordinates the row already contains). An event supplies, per keyframe,
//     G = Tcw_after^-1 * Tcw_before
// which maps OLD world coordinates to NEW ones. Only rows anchored at that
// keyframe and created before this event are moved, so a replayed/queued event
// cannot move points that were already born in corrected coordinates.
//
// This routine is a PURE CORRECTION: it never creates or removes a row.
// ---------------------------------------------------------------------------
void GaussianModelLine::ensureGbaMetadata()
{
    const int64_t N = xyz_.defined() ? xyz_.size(0) : 0;
    const auto lo = torch::TensorOptions().dtype(torch::kLong).device(device_type_);
    if (!ref_kf_id_.defined() || ref_kf_id_.numel() != N)
    {
        ref_kf_id_ = torch::full({N}, -1LL, lo);
        std::cerr << "[RefKf] ref_kf_id_ missing/size-mismatched; "
                  << "reported as unknown for all " << N << " rows\n";
    }
    else
    {
        ref_kf_id_ = ref_kf_id_.to(device_type_).to(torch::kLong).contiguous();
    }
    if (!coord_event_id_.defined() || coord_event_id_.numel() != N)
        coord_event_id_ = torch::zeros({N}, lo);
    else
        coord_event_id_ = coord_event_id_.to(device_type_).to(torch::kLong).contiguous();
}

unsigned long GaussianModelLine::allocateCoordEventId()
{
    ensureGbaMetadata();
    if (coord_event_seq_ < last_applied_gba_event_)
        coord_event_seq_ = last_applied_gba_event_;
    return ++coord_event_seq_;
}

namespace { void fgmHashTensor(const torch::Tensor& t, std::uint64_t& h); }

std::string GaussianModelLine::adamFingerprint()
{
    std::uint64_t h = 1469598103934665603ULL;
    if (!optimizer_)
        return std::string("no_optimizer");
    auto& groups = optimizer_->param_groups();
    auto& state = optimizer_->state();
    for (std::size_t i = 0; i < groups.size() && i < 6; ++i)
    {
        if (groups[i].params().empty()) continue;
        auto& p = groups[i].params()[0];
        auto it = state.find(p.unsafeGetTensorImpl());
        if (it == state.end())
        {
            h ^= 0x9e3779b97f4a7c15ULL * (i + 1);
            continue;
        }
        auto& st = static_cast<torch::optim::AdamParamState&>(*it->second);
        h ^= static_cast<std::uint64_t>(st.step());
        h *= 1099511628211ULL;
        fgmHashTensor(st.exp_avg(), h);
        fgmHashTensor(st.exp_avg_sq(), h);
        h ^= 0xc2b2ae3d27d4eb4fULL * (i + 1);
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%016llx",
                  static_cast<unsigned long long>(h));
    return std::string(buf);
}

GaussianModelLine::GbaCorrectionStats GaussianModelLine::applyGbaCorrections(
    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
    unsigned long event_id, GbaCorrectionMode mode, float world_scale)
{
    GbaCorrectionStats st;
    if (!xyz_.defined() || xyz_.size(0) == 0)
        return st;

    const float ws =
        (std::isfinite(world_scale) && world_scale > 1e-6f) ? world_scale : 1.0f;
    const bool do_scale = std::abs(ws - 1.0f) > 1e-6f;

    torch::NoGradGuard no_grad;
    ensureGbaMetadata();
    const int64_t N = xyz_.size(0);
    const auto bo = torch::TensorOptions().dtype(torch::kBool).device(device_type_);
    const auto lo = torch::TensorOptions().dtype(torch::kLong).device(device_type_);
    const auto fo = torch::TensorOptions().dtype(torch::kFloat32).device(device_type_);

    if (!xyz_init_.defined() || xyz_init_.size(0) != N)
        xyz_init_ = xyz_.detach().clone();
    const bool have_line = is_line_.defined() && is_line_.size(0) == N &&
                           line_dir_w_.defined() && line_dir_w_.size(0) == N;

    torch::Tensor covered = torch::zeros({N}, bo);

    for (const auto& kv : world_deltas)
    {
        const long long kfid = static_cast<long long>(kv.first);
        const Sophus::SE3f& G = kv.second;

        torch::Tensor mask = ref_kf_id_.eq(kfid);
        covered = torch::logical_or(covered, mask);

        torch::Tensor stale = torch::logical_and(
            mask, coord_event_id_.ge(static_cast<long long>(event_id)));
        st.stale_rows += static_cast<long long>(stale.sum().item<std::int64_t>());

        torch::Tensor active = torch::logical_and(
            mask, coord_event_id_.lt(static_cast<long long>(event_id)));
        const long long n = static_cast<long long>(active.sum().item<std::int64_t>());
        if (n == 0)
            continue;

        Eigen::Matrix3f Rm = G.rotationMatrix();
        Eigen::Vector3f tm = G.translation();
        Eigen::Quaternionf qG(Rm);
        const Eigen::Vector4f qc = qG.coeffs(); // x, y, z, w
        // Eigen is column-major: from_blob would expose R^T to row-major torch
        // indexing, so transpose once to obtain the true R.
        torch::Tensor R = torch::from_blob(
            Rm.data(), {3, 3}, torch::TensorOptions().dtype(torch::kFloat32))
            .clone().transpose(0, 1).contiguous().to(device_type_);
        torch::Tensor t = torch::from_blob(
            tm.data(), {1, 3}, torch::TensorOptions().dtype(torch::kFloat32))
            .clone().to(device_type_);
        torch::Tensor sel = torch::nonzero(active).flatten();

        // Centres and world anchors: p' = ws * (R p) + t.
        auto rigid_move = [&](torch::Tensor& dst) {
            torch::Tensor moved =
                torch::matmul(dst, R.transpose(0, 1)) * ws + t;
            dst.index_copy_(0, sel, moved.index_select(0, sel));
        };
        {
            // Measure the actual displacement before mutating xyz_ so that
            // "processed" and "non-zero displacement" are separate numbers.
            torch::Tensor before = xyz_.index_select(0, sel);
            torch::Tensor after =
                torch::matmul(before, R.transpose(0, 1)) * ws + t;
            torch::Tensor disp = (after - before).norm(2, 1);
            st.nonzero_moved_rows += static_cast<long long>(
                disp.gt(1e-6f).sum().item<std::int64_t>());
            st.translation_sum += disp.sum().item<double>();
            st.max_translation = std::max(
                st.max_translation, disp.max().item<float>());
        }
        rigid_move(xyz_);
        if (mode == GbaCorrectionMode::MeansOnly)
        {
            // Diagnostic decomposition: centres only. Anchors, quaternions,
            // line directions, Adam moments and the coordinate version are
            // deliberately left untouched. Not used by production.
            st.moved_rows += n;
            ++st.groups_applied;
            continue;
        }
        rigid_move(xyz_init_);

        // Gaussian orientation: q' = q_G * q (row-vector/layout w,x,y,z).
        {
            torch::Tensor q = rotation_;
            torch::Tensor w2 = q.index({torch::indexing::Slice(), 0});
            torch::Tensor x2 = q.index({torch::indexing::Slice(), 1});
            torch::Tensor y2 = q.index({torch::indexing::Slice(), 2});
            torch::Tensor z2 = q.index({torch::indexing::Slice(), 3});
            const float w1 = qc[3], x1 = qc[0], y1 = qc[1], z1 = qc[2];
            torch::Tensor nw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
            torch::Tensor nx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
            torch::Tensor ny = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
            torch::Tensor nz = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
            torch::Tensor qn = torch::stack({nw, nx, ny, nz}, 1);
            qn = qn / torch::sqrt(torch::sum(qn * qn, 1, true).clamp_min(1e-16f));
            rotation_.index_copy_(0, sel, qn.index_select(0, sel));
        }

        // World line direction: d' = R d (rotation only, no translation and
        // no uniform scale: a direction is scale-invariant).
        if (have_line)
        {
            torch::Tensor line_active = torch::logical_and(active, is_line_);
            const long long nl =
                static_cast<long long>(line_active.sum().item<std::int64_t>());
            if (nl > 0)
            {
                torch::Tensor lsel = torch::nonzero(line_active).flatten();
                torch::Tensor rotated =
                    torch::matmul(line_dir_w_, R.transpose(0, 1));
                line_dir_w_.index_copy_(
                    0, lsel, rotated.index_select(0, lsel));
                st.line_rows_moved += nl;
            }
        }

        // Covariance under a uniform scale: scaling_ stores log-standard
        // deviations, so s * Sigma -> scaling_ + log(s) on the selected rows.
        // The quaternion is scale-invariant and was already rotated above.
        if (do_scale && scaling_.defined() && scaling_.size(0) == N)
        {
            torch::Tensor scaled = scaling_.index_select(0, sel) +
                                  std::log(ws);
            if (optimizer_)
            {
                torch::Tensor next = scaling_.clone();
                next.index_copy_(0, sel, scaled);
                scaling_ = replaceTensorToOptimizer(next, 4);
                Tensor_vec_scaling_ = {scaling_};
            }
            else
            {
                scaling_.index_copy_(0, sel, scaled);
            }
        }

        // Adam policy: the xyz first moment is a world-space vector and is
        // transformed with the geometry. The second moment is an axis-wise
        // positive quantity; it is left unchanged (documented approximation).
        // Other parameter groups are unaffected.
        if (optimizer_)
        {
            auto& xyz_param = optimizer_->param_groups()[0].params()[0];
            auto key = xyz_param.unsafeGetTensorImpl();
            auto it = optimizer_->state().find(key);
            if (it != optimizer_->state().end())
            {
                auto& adam =
                    static_cast<torch::optim::AdamParamState&>(*it->second);
                if (adam.exp_avg().defined() && adam.exp_avg().size(0) == N)
                {
                    torch::Tensor ea = adam.exp_avg();
                    torch::Tensor rot =
                        torch::matmul(ea, R.transpose(0, 1)) * ws;
                    ea.index_copy_(0, sel, rot.index_select(0, sel));
                    adam.exp_avg(ea);
                }
            }
        }

        coord_event_id_.index_fill_(
            0, sel, static_cast<long long>(event_id));
        st.moved_rows += n;
        ++st.groups_applied;
    }

    // In-place content changes: bump the line cache explicitly.
    invalidateLineCache();

    st.unassigned_rows =
        static_cast<long long>(ref_kf_id_.lt(0).sum().item<std::int64_t>());
    torch::Tensor missing = torch::logical_and(
        torch::logical_not(covered),
        torch::logical_and(ref_kf_id_.ge(0),
                           coord_event_id_.lt(static_cast<long long>(event_id))));
    st.missing_event_rows =
        static_cast<long long>(missing.sum().item<std::int64_t>());

    if (event_id > last_applied_gba_event_)
        last_applied_gba_event_ = event_id;
    return st;
}

// ---------------------------------------------------------------------------
// geom-sync-v1 一: CPU reference test for the rigid-correction contract.
//
// Purpose: prove, numerically, that the transform entry recovers the TRUE R and
// t from the tensor layout the mapper actually passes, and that the row-vector
// multiplication rotates a direction by R (not R^T). It also measures the OLD
// expression so the defect is a number, not an argument.
//
// Pure CPU; independent of the CUDA kernel and of any real loop closure.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// geom-sync-v3: model-level contract for applyGbaCorrections().
//
// Builds a 4-row model on CPU (2 rows anchored at kf 10, 1 at kf 11, 1 with no
// anchor) and checks: rigid centre/anchor/rotation map, line direction, scale
// invariance, covariance transformation, unselected rows unchanged, cache
// refresh, duplicate/stale rejection and sequential composition.
// ---------------------------------------------------------------------------
static bool runGbaApplyContractTest()
{
    using namespace torch::indexing;
    auto fo = torch::TensorOptions().dtype(torch::kFloat32);

    GaussianModelLine m(3);
    m.device_type_ = torch::kCPU;

    const float c45 = std::cos(3.14159265358979f / 4.f);
    const float s45 = std::sin(3.14159265358979f / 4.f);
    m.xyz_ = torch::tensor({
        {0.5f, 0.0f, 3.0f},
        {1.0f, 2.0f, 3.0f},
        {0.0f, 1.0f, 2.0f},
        {2.0f, 2.0f, 2.0f}}, fo).requires_grad_();
    m.xyz_init_ = m.xyz_.detach().clone();
    m.xyz_init_.index_put_({1, 0}, 0.5f);   // anchor at p1 - 0.5 * dir
    m.is_line_ = torch::tensor(
        {false, true, false, false},
        torch::TensorOptions().dtype(torch::kBool));
    m.line_dir_w_ = torch::tensor({
        {1.f, 0.f, 0.f}, {1.f, 0.f, 0.f},
        {1.f, 0.f, 0.f}, {1.f, 0.f, 0.f}}, fo);
    m.scaling_ = torch::log(torch::tensor({
        {0.10f, 0.20f, 0.30f},
        {0.40f, 0.50f, 0.60f},
        {0.20f, 0.20f, 0.20f},
        {0.30f, 0.30f, 0.30f}}, fo)).requires_grad_();
    m.rotation_ = torch::tensor({
        {1.f, 0.f, 0.f, 0.f},
        {c45, 0.f, 0.f, s45},   // 90 deg about Z
        {1.f, 0.f, 0.f, 0.f},
        {1.f, 0.f, 0.f, 0.f}}, fo).requires_grad_();
    m.features_dc_ = torch::zeros({4, 1, 3}, fo).requires_grad_();
    m.features_rest_ = torch::zeros({4, 0, 3}, fo).requires_grad_();
    m.opacity_ = torch::zeros({4, 1}, fo).requires_grad_();
    m.exist_since_iter_ = torch::zeros(
        {4}, torch::TensorOptions().dtype(torch::kInt32));
    m.point_ids_ = torch::arange(
        0, 4, torch::TensorOptions().dtype(torch::kLong));
    m.ref_kf_id_ = torch::tensor(
        {10, 10, 11, -1}, torch::TensorOptions().dtype(torch::kLong));
    m.coord_event_id_ = torch::zeros(
        {4}, torch::TensorOptions().dtype(torch::kLong));
    m.last_applied_gba_event_ = 0;
    m.rebuildParamVectors();
    GaussianOptimizationParamsLine args;
    m.trainingSetup(args);

    // Pre-correction covariance of row 1 (line) and its rotation.
    Eigen::Vector3f s1(0.40f, 0.50f, 0.60f);
    Eigen::Matrix3f S1 = s1.array().square().matrix().asDiagonal();
    Eigen::Quaternionf q1(c45, 0.f, 0.f, s45);   // w,x,y,z
    Eigen::Matrix3f Rg1 = q1.toRotationMatrix();
    Eigen::Matrix3f Sigma1 = Rg1 * S1 * Rg1.transpose();

    const float c = std::cos(3.14159265358979f / 2.f);
    const float sn = std::sin(3.14159265358979f / 2.f);
    Eigen::Matrix3f Rz;
    Rz << c, -sn, 0.f, sn, c, 0.f, 0.f, 0.f, 1.f;
    Eigen::Vector3f tG(0.2f, 0.4f, -0.1f);
    const Sophus::SE3f G1(Rz, tG);

    std::vector<std::pair<unsigned long, Sophus::SE3f>> d1{{10u, G1}};
    auto st1 = m.applyGbaCorrections(d1, 1);

    auto get = [&](const torch::Tensor& t, int i, int j) {
        return t.index({i, j}).item<float>();
    };
    auto vec = [&](const torch::Tensor& t, int i) {
        return Eigen::Vector3f(get(t, i, 0), get(t, i, 1), get(t, i, 2));
    };

    // Row 0 centre: p' = R p + t.
    Eigen::Vector3f p0(0.5f, 0.0f, 3.0f);
    Eigen::Vector3f p0e = Rz * p0 + tG;
    bool centre_ok = (vec(m.xyz_, 0) - p0e).norm() < 1e-5f;
    printf("[GbaContract][diag] row0=(%.4f,%.4f,%.4f) expected=(%.4f,%.4f,%.4f) "
           "G_t=(%.4f,%.4f,%.4f) G_R00=%.4f G_R01=%.4f\n",
           vec(m.xyz_, 0).x(), vec(m.xyz_, 0).y(), vec(m.xyz_, 0).z(),
           p0e.x(), p0e.y(), p0e.z(), G1.translation().x(),
           G1.translation().y(), G1.translation().z(),
           G1.rotationMatrix()(0, 0), G1.rotationMatrix()(0, 1));

    // Row 1 line direction and anchor.
    Eigen::Vector3f d1e = Rz * Eigen::Vector3f(1.f, 0.f, 0.f);
    bool dir_ok = (vec(m.line_dir_w_, 1) - d1e).norm() < 1e-5f;
    Eigen::Vector3f a1(0.5f, 2.0f, 3.0f);   // xyz_init_[1]
    Eigen::Vector3f a1e = Rz * a1 + tG;
    bool anchor_ok = (vec(m.xyz_init_, 1) - a1e).norm() < 1e-5f;
    // Line residual: (centre - anchor) stays parallel to the new direction.
    Eigen::Vector3f residual = vec(m.xyz_, 1) - vec(m.xyz_init_, 1);
    bool residual_ok = residual.cross(d1e).norm() < 1e-4f;

    // Unselected / missing / unassigned rows unchanged.
    bool rows_unchanged =
        (vec(m.xyz_, 2) - Eigen::Vector3f(0.f, 1.f, 2.f)).norm() < 1e-6f &&
        (vec(m.xyz_, 3) - Eigen::Vector3f(2.f, 2.f, 2.f)).norm() < 1e-6f;

    // Scale untouched, covariance = R_G Sigma R_G^T for the line row.
    Eigen::Vector3f s1a(get(m.scaling_, 1, 0), get(m.scaling_, 1, 1),
                        get(m.scaling_, 1, 2));
    // scaling_ stores log-scale.
    bool scale_ok = (s1a - Eigen::Vector3f(s1.array().log())).norm() < 1e-6f;
    Eigen::Quaternionf q1a(get(m.rotation_, 1, 0), get(m.rotation_, 1, 1),
                           get(m.rotation_, 1, 2), get(m.rotation_, 1, 3));
    q1a.normalize();
    Eigen::Vector3f s1a_act = s1a.array().exp();
    Eigen::Matrix3f Sigma1a =
        q1a.toRotationMatrix() * s1a_act.array().square().matrix().asDiagonal() *
        q1a.toRotationMatrix().transpose();
    Eigen::Matrix3f Sigma1e = Rz * Sigma1 * Rz.transpose();
    bool cov_ok = (Sigma1a - Sigma1e).norm() < 1e-5f;

    // Line cache must expose the corrected anchor/direction.
    m.refreshLineCache();
    bool cache_ok = true;
    {
        auto line_anchor = m.line_anchor_cached_.index({0});
        auto line_dir = m.line_dir_cached_.index({0});
        cache_ok = (vec(line_anchor.unsqueeze(0), 0) - a1e).norm() < 1e-5f &&
                   (vec(line_dir.unsqueeze(0), 0) - d1e).norm() < 1e-5f;
    }

    const bool counts_ok =
        st1.moved_rows == 2 && st1.groups_applied == 1 &&
        st1.unassigned_rows == 1 && st1.missing_event_rows == 1 &&
        st1.stale_rows == 0;

    // Replaying the same event must not move anything.
    auto st_dup = m.applyGbaCorrections(d1, 1);
    bool dup_ok = st_dup.moved_rows == 0 && st_dup.stale_rows == 2;

    // A second event composes: p'' = G2 * G1 * p.
    const Sophus::SE3f G2(Eigen::Matrix3f::Identity(),
                          Eigen::Vector3f(0.1f, 0.f, 0.f));
    std::vector<std::pair<unsigned long, Sophus::SE3f>> d2{{10u, G2}};
    auto st2 = m.applyGbaCorrections(d2, 2);
    Eigen::Vector3f p0ee = G2 * p0e;
    bool order_ok = st2.moved_rows == 2 &&
                    (vec(m.xyz_, 0) - p0ee).norm() < 1e-5f &&
                    m.lastAppliedGbaEvent() == 2;

    // Event 3: uniform Sim(3) scale on the kf-11 row. It must scale both the
    // centre and the world anchor, add log(s) to the log-scale, and leave the
    // other keyframe's rows untouched.
    const Sophus::SE3f G3(Eigen::Matrix3f::Identity(),
                          Eigen::Vector3f(0.f, 0.f, 0.5f));
    std::vector<std::pair<unsigned long, Sophus::SE3f>> d3{{11u, G3}};
    const float world_scale = 2.0f;
    auto st3 = m.applyGbaCorrections(
        d3, 3, GaussianModelLine::GbaCorrectionMode::Full, world_scale);
    Eigen::Vector3f p2(0.f, 1.f, 2.f);
    Eigen::Vector3f p2e = world_scale * p2 + G3.translation();
    Eigen::Vector3f a2e = world_scale * p2 + G3.translation();
    Eigen::Vector3f s2a(get(m.scaling_, 2, 0), get(m.scaling_, 2, 1),
                        get(m.scaling_, 2, 2));
    bool scale_apply_ok =
        st3.moved_rows == 1 &&
        (vec(m.xyz_, 2) - p2e).norm() < 1e-5f &&
        (vec(m.xyz_init_, 2) - a2e).norm() < 1e-5f &&
        (s2a - (Eigen::Vector3f(0.2f, 0.2f, 0.2f).array().log() +
                std::log(world_scale)).matrix()).norm() < 1e-5f &&
        (vec(m.xyz_, 0) - p0ee).norm() < 1e-6f;

    // New rows created after event 3 must inherit event version 3. Replaying
    // event 1 must not move them; a genuinely newer event 4 must.
    torch::Tensor nx = torch::tensor({{4.f, 5.f, 6.f}}, fo);
    torch::Tensor nfdc = torch::zeros({1, 1, 3}, fo);
    torch::Tensor nfrest = torch::zeros({1, 0, 3}, fo);
    torch::Tensor nopa = torch::zeros({1, 1}, fo);
    torch::Tensor nsca = torch::log(torch::tensor({{0.1f, 0.1f, 0.1f}}, fo));
    torch::Tensor nrot = torch::tensor({{1.f, 0.f, 0.f, 0.f}}, fo);
    torch::Tensor nexi = torch::zeros(
        {1}, torch::TensorOptions().dtype(torch::kInt32));
    torch::Tensor nline = torch::zeros(
        {1}, torch::TensorOptions().dtype(torch::kBool));
    torch::Tensor ndir = torch::tensor({{1.f, 0.f, 0.f}}, fo);
    torch::Tensor ninit = nx.clone();
    torch::Tensor nref = torch::tensor(
        {10}, torch::TensorOptions().dtype(torch::kLong));
    const int64_t new_idx = m.xyz_.size(0);
    m.densificationPostfixWithLineAwareness(
        nx, nfdc, nfrest, nopa, nsca, nrot, nexi, nline, ndir, ninit, nref);
    const long long new_stamp =
        m.coord_event_id_.index({new_idx}).item<std::int64_t>();
    const bool stamp_ok = new_stamp == 3 &&
                          m.lastAppliedGbaEvent() == 3;
    const Eigen::Vector3f np_before = vec(m.xyz_, new_idx);
    auto st_replay = m.applyGbaCorrections(d1, 1);
    const bool new_replay_ok =
        st_replay.moved_rows == 0 &&
        (vec(m.xyz_, new_idx) - np_before).norm() < 1e-6f;
    auto st_new = m.applyGbaCorrections(d1, 4);
    Eigen::Vector3f np_after = Rz * np_before + tG;
    const bool new_event4_ok =
        st_new.moved_rows >= 1 &&
        (vec(m.xyz_, new_idx) - np_after).norm() < 1e-5f;

    const bool pass = centre_ok && dir_ok && anchor_ok && residual_ok &&
                      rows_unchanged && scale_ok && cov_ok && cache_ok &&
                      counts_ok && dup_ok && order_ok && scale_apply_ok &&
                      stamp_ok && new_replay_ok && new_event4_ok;
    printf("[GbaContract] centre=%d dir=%d anchor=%d residual=%d rows=%d "
           "scale=%d covariance=%d cache=%d counts=%d duplicate=%d order=%d "
           "scale_apply=%d stamp=%d new_replay=%d new_event=%d : %s\n",
           centre_ok, dir_ok, anchor_ok, residual_ok, rows_unchanged, scale_ok,
           cov_ok, cache_ok, counts_ok, dup_ok, order_ok, scale_apply_ok,
           stamp_ok, new_replay_ok, new_event4_ok, pass ? "PASS" : "FAIL");
    fflush(stdout);
    return pass;
}

bool runRigidCorrectionContractTest()
{
    auto cpu = torch::TensorOptions().dtype(torch::kFloat);
    float worst_rec = 0.f, worst_dir = 0.f, worst_anchor = 0.f, old_dir = 0.f;

    auto check = [&](const Eigen::Matrix4f& M, const Eigen::Vector3f& d,
                     const Eigen::Vector3f& a, float s, const char* label) {
        // Emulate exactly what the mapper does: EigenMatrix2TorchTensor(M)
        // yields T[i][j] = M(i,j); then .transpose(0,1).
        Eigen::Matrix4f MT = M.transpose();
        torch::Tensor T = torch::from_blob(MT.data(), {4, 4}, cpu).clone();
        torch::Tensor diff_pose = T.transpose(0, 1);

        // New entry.
        torch::Tensor Mc = diff_pose.transpose(0, 1).contiguous();
        torch::Tensor R = Mc.slice(0, 0, 3).slice(1, 0, 3);
        torch::Tensor t = Mc.slice(0, 0, 3).slice(1, 3, 4).transpose(0, 1);

        float rec = 0.f;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rec = std::max(rec, std::abs(R.index({i, j}).item<float>() - M(i, j)));
        for (int j = 0; j < 3; ++j)
            rec = std::max(rec, std::abs(t.index({0, j}).item<float>() - M(j, 3)));
        worst_rec = std::max(worst_rec, rec);

        torch::Tensor dv = torch::from_blob((void*)d.data(), {1, 3}, cpu).clone();
        torch::Tensor av = torch::from_blob((void*)a.data(), {1, 3}, cpu).clone();

        Eigen::Vector3f dref = M.block<3, 3>(0, 0) * d;
        torch::Tensor rot = torch::matmul(dv, R.transpose(0, 1));
        float ed = 0.f;
        for (int j = 0; j < 3; ++j)
            ed = std::max(ed, std::abs(rot.index({0, j}).item<float>() - dref[j]));
        worst_dir = std::max(worst_dir, ed);

        // OLD expression (slice straight off diff_pose) for contrast.
        torch::Tensor Rold = diff_pose.slice(0, 0, 3).slice(1, 0, 3);
        torch::Tensor rotold = torch::matmul(dv, Rold.transpose(0, 1));
        float eo = 0.f;
        for (int j = 0; j < 3; ++j)
            eo = std::max(eo, std::abs(rotold.index({0, j}).item<float>() - dref[j]));
        old_dir = std::max(old_dir, eo);

        Eigen::Vector3f aref = s * (M.block<3, 3>(0, 0) * a) + M.block<3, 1>(0, 3);
        torch::Tensor moved = torch::matmul(av, R.transpose(0, 1)) * s + t;
        float ea = 0.f;
        for (int j = 0; j < 3; ++j)
            ea = std::max(ea, std::abs(moved.index({0, j}).item<float>() - aref[j]));
        worst_anchor = std::max(worst_anchor, ea);

        printf("[SyncContract] %-14s recover=%.2e direction=%.2e anchor=%.2e "
               "old_expression=%.2e\n", label, rec, ed, ea, eo);
    };

    auto mk = [](const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
        Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
        M.block<3, 3>(0, 0) = R;
        M.block<3, 1>(0, 3) = t;
        return M;
    };

    check(mk(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()),
          {1.f, 0.f, 0.f}, {0.3f, -0.2f, 1.7f}, 1.f, "identity");
    check(mk(Eigen::Matrix3f::Identity(), {0.5f, -0.25f, 1.f}),
          {0.f, 1.f, 0.f}, {1.f, 2.f, 3.f}, 1.f, "translation");
    const float c = std::cos(3.14159265358979f / 2.f), sn = std::sin(3.14159265358979f / 2.f);
    Eigen::Matrix3f Rz;
    Rz << c, -sn, 0.f, sn, c, 0.f, 0.f, 0.f, 1.f;
    check(mk(Rz, {0.2f, 0.4f, -0.1f}), {1.f, 0.f, 0.f}, {0.5f, 0.1f, -0.7f}, 1.f, "Rz90+trans");
    check(mk(Rz, {0.2f, 0.4f, -0.1f}), {1.f, 0.f, 0.f}, {0.5f, 0.1f, -0.7f}, 2.f, "Rz90+scale2");

    // ------------------------------------------------------------------
    // geom-sync-v2 里程碑 1: measure the GPU centre path instead of deriving it.
    // Calls the real API the mapper calls, with the tensor the mapper builds
    // (EigenMatrix2TorchTensor(M, CUDA).transpose(0,1)), and compares against the
    // CPU reference. Tolerance fixed BEFORE running: 1e-4 absolute (fp32 kernel
    // on a 4x4 with O(1) coordinates).
    // ------------------------------------------------------------------
    float worst_gpu = 0.f;
    bool gpu_ran = false;
    if (torch::cuda::is_available()) {
        gpu_ran = true;
        const float tol_gpu = 1e-4f;
        std::vector<Eigen::Vector3f> pts_cpu = {
            {0.4f, -0.3f, 1.9f}, {-1.1f, 0.7f, 0.5f},
            {2.0f, 1.3f, -0.8f}, {0.05f, -2.2f, 3.1f}};
        auto run_case = [&](const Eigen::Matrix4f& M, const char* label) {
            torch::Tensor pts = torch::zeros({(long)pts_cpu.size(), 3},
                                             torch::TensorOptions().dtype(torch::kFloat));
            for (size_t i = 0; i < pts_cpu.size(); ++i)
                for (int j = 0; j < 3; ++j)
                    pts.index_put_({(long)i, j}, pts_cpu[i][j]);
            torch::Tensor pts_gpu = pts.to(torch::kCUDA);
            torch::Tensor mt = tensor_utils::EigenMatrix2TorchTensor(M, torch::kCUDA)
                                   .transpose(0, 1);
            transformPoints(pts_gpu, mt);
            cudaDeviceSynchronize();
            torch::Tensor back = pts_gpu.to(torch::kCPU);
            float e = 0.f;
            for (size_t i = 0; i < pts_cpu.size(); ++i) {
                Eigen::Vector3f ref = M.block<3, 3>(0, 0) * pts_cpu[i] + M.block<3, 1>(0, 3);
                for (int j = 0; j < 3; ++j)
                    e = std::max(e, std::abs(back.index({(long)i, j}).item<float>() - ref[j]));
            }
            worst_gpu = std::max(worst_gpu, e);
            printf("[SyncContract] GPU centre %-14s max_abs_vs_cpu=%.2e %s\n",
                   label, e, e < tol_gpu ? "OK" : "MISMATCH");
        };
        run_case(mk(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()), "identity");
        run_case(mk(Eigen::Matrix3f::Identity(), {0.5f, -0.25f, 1.f}), "translation");
        run_case(mk(Rz, {0.2f, 0.4f, -0.1f}), "Rz90+trans");
    } else {
        printf("[SyncContract] GPU centre SKIPPED (no CUDA device)\n");
    }

    const bool gba_apply_pass = runGbaApplyContractTest();
    const bool pass = (worst_rec < 1e-5f) && (worst_dir < 1e-5f) &&
                      (worst_anchor < 1e-5f) && (!gpu_ran || worst_gpu < 1e-4f) &&
                      gba_apply_pass;
    printf("[SyncContract] worst recover=%.2e direction=%.2e anchor=%.2e "
           "gpu_centre=%.2e | old expression error=%.2e | gba_apply=%s | %s\n",
           worst_rec, worst_dir, worst_anchor, worst_gpu, old_dir,
           gba_apply_pass ? "PASS" : "FAIL",
           pass ? "PASS (new entry)" : "FAIL (new entry)");
    fflush(stdout);
    return pass;
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

    // Snapshot the pre-transform mask so we can isolate the points transformed
    // by THIS call (avoid double-rotating line directions across KF iterations).
    torch::Tensor pre_mask = point_not_transformed_flags.clone();

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

    // Points transformed by this call = those that were NOT transformed before
    // and are now marked as transformed.
    torch::Tensor newly_transformed = torch::logical_and(
        pre_mask, torch::logical_not(point_not_transformed_flags));

    if (newly_transformed.any().item<bool>()) {
        // Sync line direction (world) for line-sampled gaussians only. The
        // anisotropic gaussian rotation (rotation_) is already updated by the
        // kernel; line_dir_w_ must follow the same rigid rotation.
        // ------------------------------------------------------------------
        // geom-sync-v1: explicit rigid-correction entry for world-space line
        // metadata.
        //
        // Contract (derived from the actual code, not assumed):
        //   * `diff_pose` arrives as EigenMatrix2TorchTensor(M).transpose(0,1).
        //     EigenMatrix2TorchTensor returns T[i][j] = M(i,j), so what we get
        //     here is logically M^T.
        //   * The CUDA kernel consumes `diff_pose.contiguous()`, which for that
        //     transposed view is M in COLUMN-major order -- exactly what
        //     transformPoint4x3 indexes ([0],[4],[8],[12]). That is why the
        //     CENTRES are correct and are left to the kernel.
        //   * Index-based slicing, however, sees M^T, so
        //     diff_pose.slice(0,0,3).slice(1,0,3) is R^T -- NOT R. The previous
        //     code used that slice for the line direction, i.e. it applied the
        //     INVERSE rotation to line_dir_w_.
        // ------------------------------------------------------------------
        torch::Tensor M = diff_pose.transpose(0, 1).contiguous().to(torch::kFloat);
        torch::Tensor R = M.slice(0, 0, 3).slice(1, 0, 3);   // true R, [3,3]
        // M is a homogeneous matrix: the translation is COLUMN 3 (rows 0..2),
        // not row 3. Reading row 3 (as the first draft did) yields (0,0,0) and
        // was caught by the contract self-test before it shipped.
        torch::Tensor t = M.slice(0, 0, 3).slice(1, 3, 4).transpose(0, 1); // [1,3]

        torch::Tensor line_mask = torch::logical_and(newly_transformed, this->is_line_);
        if (line_mask.any().item<bool>() && this->line_dir_w_.size(0) == this->xyz_.size(0)) {
            torch::Tensor dirs = this->line_dir_w_;
            // Row-vector convention: (R d)^T = d^T R^T. Directions take the
            // rotation only -- no translation, no scale.
            torch::Tensor rotated = torch::matmul(dirs, R.transpose(0, 1));
            torch::Tensor m3 = line_mask.unsqueeze(1).expand({-1, 3});
            invalidateLineCache();   // cache version bump (assignment below)
            this->line_dir_w_ = torch::where(m3, rotated, dirs);
        }

        // World-space line ANCHOR. xyz_init_ is consumed as a point on the line
        // by the structural losses (see the anchor read above), so it must take
        // exactly the same rigid map as the centre:
        //     anchor' = s * (R * anchor) + t
        // (the kernel applies p' = M (s p), i.e. the scale is applied before
        // the rotation).
        if (this->xyz_init_.defined() && this->xyz_init_.size(0) == this->xyz_.size(0)) {
            torch::Tensor anchors = this->xyz_init_;
            torch::Tensor moved = torch::matmul(anchors, R.transpose(0, 1)) * scale
                                  + t.expand({anchors.size(0), 3});
            torch::Tensor m3 = newly_transformed.unsqueeze(1).expand({-1, 3});
            this->xyz_init_ = torch::where(m3, moved, anchors);
            invalidateLineCache();
        }

        // Sync world scale / covariance magnitude for a scale-changing loop.
        // (SE(3) loops have scale == 1 and this is a no-op.)
        if (std::abs(scale - 1.0f) > 1e-6f) {
            torch::Tensor scaled = this->scaling_ * scale;
            torch::Tensor m = newly_transformed.unsqueeze(1).expand({-1, this->scaling_.size(1)});
            torch::Tensor new_scaling = torch::where(m, scaled, this->scaling_);
            this->scaling_ = replaceTensorToOptimizer(new_scaling, 4);
            this->Tensor_vec_scaling_ = {this->scaling_};
        }
    }

    // Diagnostic: why are line Gaussians (not) transformed by loop correction?
    static const bool diag = []() {
        const char* e = std::getenv("PHOTO_SLAM_DEBUG_LINE_LOOP");
        return e && std::atoi(e) != 0;
    }();
    if (diag) {
        torch::Tensor belong = torch::logical_and(
            this->is_line_,
            torch::abs(this->exist_since_iter_ - kf_creation_iter) < stable_num_iter_existence);
        torch::Tensor vis = markVisible(points, kf_world_view_transform, kf_full_proj_transform);
        torch::Tensor line_vis = torch::logical_and(this->is_line_, vis);
        std::cerr << "[LoopDiag] kf_iter=" << kf_creation_iter
                  << " lineTotal=" << this->is_line_.sum().item<int64_t>()
                  << " lineBelong=" << belong.sum().item<int64_t>()
                  << " lineVisible=" << line_vis.sum().item<int64_t>()
                  << " lineBelongVis=" << torch::logical_and(belong, line_vis).sum().item<int64_t>()
                  << " lineTransformedThisKf=" << torch::logical_and(newly_transformed, this->is_line_).sum().item<int64_t>()
                  << std::endl;
    }

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
    audit_iter_ = step;   // perf-stage3 audit: host-only iteration stamp
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
    const auto old_impl = key;
    const long long old_rows = param.size(0);
    const bool old_grad = param.grad().defined();
    long long old_step = -1;
    auto old_state_it = state.find(key);
    if (old_state_it != state.end())
        old_step = static_cast<long long>(
            static_cast<torch::optim::AdamParamState&>(*old_state_it->second)
                .step());
    auto new_state = std::make_unique<torch::optim::AdamParamState>();
    auto state_it = state.find(key);
    if (state_it != state.end())
    {
        auto& stored_state =
            static_cast<torch::optim::AdamParamState&>(*state_it->second);
        new_state->step(stored_state.step());
    }
    else
    {
        // A correction can run before this parameter group has ever produced
        // an Adam state (e.g. a scale-changing LOOP before the first step).
        new_state->step(0);
    }
    const long long new_step = static_cast<long long>(new_state->step());
    new_state->exp_avg(torch::zeros_like(tensor));
    new_state->exp_avg_sq(torch::zeros_like(tensor));
    // new_state->max_exp_avg_sq(stored_state.max_exp_avg_sq().clone()); // needed only when options.amsgrad(true), which is false by default

    state.erase(key);
    param = tensor.requires_grad_();
    //key = c10::guts::to_string(param.unsafeGetTensorImpl());
    key = param.unsafeGetTensorImpl();
    state[key] = std::move(new_state);

    if (const char* d = std::getenv("PHOTO_SLAM_ADAM_TOPO_DIAG_DIR"))
    {
        if (d[0])
        {
            std::filesystem::path p = std::filesystem::path(d) / "replacements.csv";
            std::filesystem::create_directories(p.parent_path());
            const bool fresh = !std::filesystem::exists(p) ||
                               std::filesystem::file_size(p) == 0;
            std::ofstream out(p, std::ios::app);
            if (fresh)
                out << "tensor_idx,old_rows,new_rows,old_impl,new_impl,"
                       "old_grad,new_grad,step_before,step_after\n";
            out << tensor_idx << ',' << old_rows << ',' << tensor.size(0) << ','
                << reinterpret_cast<std::uintptr_t>(old_impl) << ','
                << reinterpret_cast<std::uintptr_t>(
                       param.unsafeGetTensorImpl())
                << ',' << (old_grad ? 1 : 0) << ','
                << (param.grad().defined() ? 1 : 0) << ',' << old_step << ','
                << new_step << '\n';
        }
    }

    auto optimizable_tensors = param;
    return optimizable_tensors;
}

void GaussianModelLine::prunePoints(torch::Tensor& mask)
{
    ensureGbaMetadata();   // size must match the pre-prune xyz_
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

    this->ref_kf_id_ = this->ref_kf_id_.index({valid_points_mask});
    this->coord_event_id_ = this->coord_event_id_.index({valid_points_mask});
}

void GaussianModelLine::prunePointsWithLineAwareness(torch::Tensor& mask)
{
    using namespace torch::indexing;

    ensureGbaMetadata();   // size must match the pre-prune xyz_
    const long long prune_rows_before = this->xyz_.size(0);
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

    // --- 🌟 关键修改：同步裁剪影子张量 ---
    if (this->xyz_init_.defined()) {
    invalidateLineCache();   // cache version bump (assignment below)
        this->xyz_init_ = this->xyz_init_.index({valid_points_mask});
    }

    // =====================================================
    // C. Structure-only & auxiliary tensors (NO optimizer)
    // =====================================================
    invalidateLineCache();   // cache version bump (assignment below)
    this->is_line_ =
        this->is_line_.index({valid_points_mask});

    invalidateLineCache();   // cache version bump (assignment below)
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

    this->ref_kf_id_ =
        this->ref_kf_id_.index({valid_points_mask});
    this->coord_event_id_ =
        this->coord_event_id_.index({valid_points_mask});

    if (const char* d = std::getenv("PHOTO_SLAM_ADAM_TOPO_DIAG_DIR"))
    {
        if (d[0])
        {
            std::filesystem::path p = std::filesystem::path(d) / "prunes.csv";
            std::filesystem::create_directories(p.parent_path());
            const bool fresh = !std::filesystem::exists(p) ||
                               std::filesystem::file_size(p) == 0;
            std::ofstream out(p, std::ios::app);
            if (fresh)
                out << "rows_before,rows_after,changed\n";
            out << prune_rows_before << ',' << this->xyz_.size(0) << ','
                << (prune_rows_before != this->xyz_.size(0) ? 1 : 0) << '\n';
        }
    }
}


void GaussianModelLine::pruneExceedinglyAnisotropic(float threshold)
{
    torch::NoGradGuard no_grad;
    torch::Tensor s = getScaling();
    
    // 计算每个高斯球的长宽比
    auto max_s = std::get<0>(torch::max(s, 1));
    auto min_s = std::get<0>(torch::min(s, 1));
    torch::Tensor ratio = max_s / (min_s + 1e-6);
    
    // 找出那些长得像“针”一样的点
    torch::Tensor cull_mask = ratio > threshold;
    
    if (cull_mask.any().item<bool>()) {
        std::cerr << "[Gaussian Model] Pruning " << cull_mask.sum().item<int>() 
                  << " spiky Gaussians (Ratio > " << threshold << ")" << std::endl;
        // 调用你现有的剪裁函数
        this->prunePointsWithLineAwareness(cull_mask);
    }
}

void GaussianModelLine::pruneTinyGaussians(float min_scale) {
    torch::NoGradGuard no_grad;
    // 获取 Scaling [N, 3]，计算其最大轴
    auto s = torch::exp(this->scaling_); 
    auto max_s = std::get<0>(torch::max(s, 1));
    
    // 如果三个轴都非常小，说明是这种无意义的刺眼碎点
    auto tiny_mask = max_s < min_scale; 
    
    if (tiny_mask.any().item<bool>()) {
        this->prunePointsWithLineAwareness(tiny_mask);
    }
}

// void GaussianModelLine::pruneIsolatedGaussians(float radius, int min_neighbors) {
//     torch::NoGradGuard no_grad;
//     torch::Tensor points = this->xyz_.detach();
//     if (points.size(0) == 0) return;
//     auto coords = torch::floor(points / radius).to(torch::kInt64);
//     auto hashes = coords.select(1, 0) * 1000000L + 
//                  coords.select(1, 1) * 1000L + 
//                  coords.select(1, 2);
//     // 🌟 修正：直接在 hashes 张量上调用 unique
//     auto unique_res = hashes._unique2(true, true, true);
//     // std::get 获取 std::tuple 的成员
//     torch::Tensor unique_vals = std::get<0>(unique_res);
//     torch::Tensor inverse_indices = std::get<1>(unique_res);
//     torch::Tensor counts = std::get<2>(unique_res);
//     // 映射回原始点云密度
//     auto point_voxel_counts = counts.index({inverse_indices});
    
//     // 生成掩码并清理
//     torch::Tensor isolated_mask = point_voxel_counts < min_neighbors;
    
//     // 建议只针对非线特征点清理，保护百叶窗结构
//     auto final_prune_mask = torch::logical_and(isolated_mask, ~this->is_line_);

//     if (final_prune_mask.any().item<bool>()) {
//         this->prunePointsWithLineAwareness(final_prune_mask);
//     }
// }


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
    invalidateLineCache();   // cache version bump (assignment below)
    this->is_line_ =
        torch::cat({this->is_line_, new_is_line.to(this->is_line_.device())}, 0);

    invalidateLineCache();   // cache version bump (assignment below)
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
    torch::Tensor& new_line_dir_w,
    torch::Tensor& new_xyz_init, // 🌟 新增参数：新点的参考位置
    const torch::Tensor& new_ref_kf_ids
)
{
    // ================= [DEBUG PROBE: 强制打印到错误流] =================
    // 使用 std::cerr 而不是 std::cerr，防止崩溃时日志丢失
    //std::cerr << "\n[DEBUG FLOW] Entering Postfix. N=" << new_xyz.size(0) << std::endl;
    //std::cerr << "  > XYZ Shape: " << new_xyz.sizes() << std::endl;
    //std::cerr << "  > Rotation Shape: " << new_rotation.sizes() << " (Expect: [N, 4])" << std::endl;
    //std::cerr << "  > Scaling Shape:  " << new_scaling.sizes()  << " (Expect: [N, 3])" << std::endl;
    // =========================================================
    // [CRITICAL FIX] 维度强制校验与修复
    // 解决 "Expected size 4 but got size N" 问题
    // =========================================================
    int64_t N = new_xyz.size(0); // 以 XYZ 的点数为基准
    // Must run BEFORE xyz_ is extended: at this point xyz_.size(0) is the old
    // row count that ref_kf_id_ / coord_event_id_ must match.
    ensureGbaMetadata();

    // 🌟 核心防御代码：确保所有影子属性维度对齐
    //TORCH_CHECK(new_is_line.size(0) == n_new, "Mismatch in new_is_line");
    //TORCH_CHECK(new_xyz_init.size(0) == n_nwe, "Mismatch in new_xyz_init");

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

    // --- 🌟 关键修改：同步拼接影子张量 ---
    invalidateLineCache();   // cache version bump (assignment below)
    this->xyz_init_ = torch::cat({this->xyz_init_, new_xyz_init.to(device_type_)}, 0).detach();

    // =====================================================
    // C. Structure-only tensors (NO optimizer state)
    // =====================================================
    // 确保这些 tensor 也在正确的设备上
    invalidateLineCache();   // cache version bump (assignment below)
    this->is_line_ = torch::cat({this->is_line_, new_is_line.to(device_type_)}, 0);
    invalidateLineCache();   // cache version bump (assignment below)
    this->line_dir_w_ = torch::cat({this->line_dir_w_, new_line_dir_w.to(device_type_)}, 0);
    this->exist_since_iter_ = torch::cat({this->exist_since_iter_, new_exist_since_iter.to(device_type_)}, 0);

    // ---- deformation metadata (geom-sync-v3) ----------------------------
    const auto lo = torch::TensorOptions().dtype(torch::kLong).device(device_type_);
    torch::Tensor add_ref;
    if (new_ref_kf_ids.defined())
    {
        add_ref = new_ref_kf_ids.to(device_type_).to(torch::kLong).flatten();
        TORCH_CHECK(add_ref.size(0) == N,
                    "new_ref_kf_ids has ", add_ref.size(0),
                    " rows, expected ", N);
    }
    else
    {
        add_ref = torch::full({N}, -1LL, lo);
    }
    this->ref_kf_id_ = torch::cat({this->ref_kf_id_, add_ref}, 0);
    this->coord_event_id_ = torch::cat(
        {this->coord_event_id_,
         torch::full({N}, static_cast<long long>(last_applied_gba_event_), lo)},
        0);

    // 🌟 拼接后再次确认全局同步
    TORCH_CHECK(this->xyz_.size(0) == this->xyz_init_.size(0), "Global xyz and xyz_init out of sync!");
    TORCH_CHECK(this->xyz_.size(0) == this->is_line_.size(0), "Global xyz and is_line out of sync!");
    TORCH_CHECK(this->xyz_.size(0) == this->ref_kf_id_.size(0), "Global xyz and ref_kf_id out of sync!");
    TORCH_CHECK(this->xyz_.size(0) == this->coord_event_id_.size(0), "Global xyz and coord_event_id out of sync!");

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
    //std::cerr << "\n[DEBUG SPLIT] Step 1: Entry" << std::endl;
    //std::cerr << "  > Global Rotation Shape: " << this->rotation_.sizes() << std::endl;
    //std::cerr << "  > Global Scaling Shape:  " << this->getScalingActivation().sizes() << std::endl;

    // 如果这里 Rotation 的第1维是 3，说明在上一帧（Clone步骤）就被 Scaling 覆盖了
    if (this->rotation_.dim() == 2 && this->rotation_.size(1) == 3) {
        std::cerr << "\033[1;31m[FATAL] Rotation is 3D at entry! It was corrupted in previous step.\033[0m" << std::endl;
    }

    // 1. [增加断言] 在函数入口确保影子张量处于同步状态
    TORCH_CHECK(this->xyz_.size(0) == this->xyz_init_.size(0), "xyz and xyz_init out of sync at Split entry!");

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
    //std::cerr << "[DEBUG SPLIT] Step 2: Selection. Selected M = " << M << std::endl;
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
    //std::cerr << "[DEBUG SPLIT] Step 3: Indexing" << std::endl;
    //std::cerr << "  > rot_sel Shape: " << rot_sel.sizes() << " (Numel: " << rot_sel.numel() << ")" << std::endl;

    if (rot_sel.dim() != 2 || rot_sel.size(1) != 4) {
        //std::cerr << "[Warning] Fixing rot_sel shape in Split: " << rot_sel.sizes() << std::endl;
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
    
    // Frozen-geometry ablation: with PHOTO_SLAM_ABL_DIRECTIONAL_DENSIFY=0 the
    // line-specific 1D split along the line axis is disabled, so every
    // Gaussian is split isotropically like a point (the line-aware routine is
    // still used, which keeps the metadata tensors aligned with xyz_).
    auto mask_line_subset = useDirectionalLineDensify()
        ? torch::logical_and(is_line_sel, geom_is_line)
        : torch::zeros_like(is_line_sel);
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
        //std::cerr << "  > [Point Branch] rot_p before reshape: " << rot_p.sizes() << std::endl;

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
    //std::cerr << "[DEBUG SPLIT] Step 4: Concatenation" << std::endl;
    //std::cerr << "  > new_xyz Shape:      " << new_xyz.sizes() << std::endl;
    //std::cerr << "  > new_scaling Shape:  " << new_scaling.sizes() << std::endl;
    //std::cerr << "  > new_rotation Shape: " << new_rotation.sizes() << std::endl;
    
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
    //std::cerr << "[DEBUG FLOW] Inside densifyAndSplit. Prepared tensors:" << std::endl;
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

    // 逻辑同上
    auto xyz_init_sel = this->xyz_init_.index({selected_mask});
    auto new_xyz_init = xyz_init_sel.index({combined_indices});
    ensureGbaMetadata();
    auto new_ref_kf_ids =
        this->ref_kf_id_.index({selected_mask}).index({combined_indices});

    // 3. [增加维度终检] 防止 Postfix 因为维度不一致而 crash
    TORCH_CHECK(new_xyz.size(0) == new_xyz_init.size(0), "new_xyz and new_xyz_init size mismatch!");

    // =========================================================================
    // 5. Postfix
    // =========================================================================
    this->densificationPostfixWithLineAwareness(
        new_xyz, new_features_dc, new_features_rest, new_opacity,
        new_scaling, new_rotation, new_exist_iter,
        new_is_line, new_line_dir_w, new_xyz_init, new_ref_kf_ids
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
    // Frozen-geometry ablation: with PHOTO_SLAM_ABL_DIRECTIONAL_DENSIFY=0 the
    // line-directed clone offset is disabled and every Gaussian is cloned like
    // a point.
    auto mask_line_subset = useDirectionalLineDensify()
        ? torch::logical_and(is_line_sel, geom_is_line)
        : torch::zeros_like(is_line_sel);
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

    // 在该函数内部获取被选中点的初始位置
    auto xyz_init_sel = this->xyz_init_.index({selected_mask});
    auto new_xyz_init = xyz_init_sel.index({combined_indices}); // 对应生成的 new_xyz
    ensureGbaMetadata();
    auto new_ref_kf_ids =
        this->ref_kf_id_.index({selected_mask}).index({combined_indices});

    // ================= [DEBUG PROBE 3] =================
    // std::cerr << "[DEBUG FLOW] Inside densifyAndClone. Prepared tensors:" << std::endl;
    // std::cerr << "  > Final new_rotation shape: " << new_rotation.sizes() << std::endl;
    // ============================================================

    // =========================================================================
    // 5. Postfix
    // =========================================================================
    this->densificationPostfixWithLineAwareness(
        new_xyz, new_features_dc, new_features_rest, new_opacity,
        new_scaling, new_rotation, new_exist_iter,
        new_is_line, new_line_dir_w, new_xyz_init, new_ref_kf_ids
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

    g_dens_prof.init();

    // 1. 获取归一化梯度 (此时长度为 N)
    auto grads = this->xyz_gradient_accum_ / this->denom_;
    grads.index_put_({grads.isnan()}, 0.0f);

    // =========================================================
    // 2. 核心改进：在点数增加前，先计算针对当前点的剪裁掩码
    // =========================================================
    // A. 基础透明度掩码 (line Gaussians get a separate, lower threshold
    // because thin structures naturally settle at lower opacity than points).
    auto opacity = this->getOpacityActivation().view({-1});
    auto opacity_mask = opacity < min_opacity;
    if (lineMinOpacity() >= 0.0f) {
        auto point_mask = torch::logical_and(~this->is_line_, opacity_mask);
        auto line_op_mask = torch::logical_and(this->is_line_, opacity < lineMinOpacity());
        opacity_mask = torch::logical_or(point_mask, line_op_mask);
    }

    // B. 激进线段剪裁 (只针对已经在图中存在一段时间的点, important)
    // Line Gaussians are thin (few pixels), so their per-point gradient is
    // inherently ~10-100x smaller than point Gaussians. Use a line-specific
    // factor (default 0.01, i.e. 1e-5 with grad_threshold 1e-3) so only
    // truly-lazy lines are pruned. factor==0 disables line lazy pruning.
    const float lazy_factor = lineLazyFactor();
    auto lazy_line_mask = torch::zeros_like(this->is_line_);
    if (lazy_factor > 0.0f) {
        float lazy_threshold = max_grad * lazy_factor;
        // 增加一个条件：denom_ > 0 确保这个点至少被投影过，且不是刚诞生的
        lazy_line_mask = torch::logical_and(
            this->is_line_, 
            torch::logical_and(grads.squeeze() < lazy_threshold, this->denom_.squeeze() > 0)
        );
    }

    // Instrument line-Gaussian retention (opacity / age / prune reason).
    // Gated behind PHOTO_SLAM_DEBUG_LINE_PRUNE (off by default: the .item()
    // calls force CPU-GPU syncs on every densify).
    torch::Tensor big_points_mask;
    if (linePruneDiag() && this->is_line_.any().item<bool>()) {
        auto line_mask = this->is_line_;
        auto op = torch::sigmoid(this->opacity_).squeeze();
        auto line_op = op.index({line_mask});
        auto line_birth = this->exist_since_iter_.index({line_mask}).to(torch::kFloat);
        std::cerr << "[LinePrune] line=" << line_mask.sum().item<int64_t>()
                  << " opMin=" << line_op.min().item<float>()
                  << " opMean=" << line_op.mean().item<float>()
                  << " opMax=" << line_op.max().item<float>()
                  << " birthMean=" << line_birth.mean().item<float>()
                  << " prunedByOpacity=" << torch::logical_and(line_mask, opacity_mask).sum().item<int64_t>()
                  << " prunedByLazy=" << lazy_line_mask.sum().item<int64_t>()
                  << std::endl;
    }

    auto prune_mask = torch::logical_or(opacity_mask, lazy_line_mask);

    // C. 屏幕空间与世界空间尺寸检查
    if (max_screen_size > 0) {
        auto big_points_vs = this->max_radii2D_ > max_screen_size;
        auto max_scaling = std::get<0>(this->getScalingActivation().max(1));
        // Point world-size check keeps the upstream 0.1*extent threshold.
        auto big_points_ws = torch::logical_and(~this->is_line_, max_scaling > 0.1f * extent);
        // Line world-size check uses a dedicated, larger factor (default 0.5)
        // because line Gaussians are elongated along the line direction by
        // design. This prunes only overgrown lines instead of exempting them.
        auto big_lines_ws = torch::logical_and(this->is_line_, max_scaling > lineSizeExtentFactor() * extent);
        big_points_mask = torch::logical_or(big_points_vs, torch::logical_or(big_points_ws, big_lines_ws));
        prune_mask = torch::logical_or(prune_mask, big_points_mask);
    }

    // =========================================================
    // 3. 执行物理剪裁 (先清减，再增补)
    // =========================================================
    // variable-topology audit: record the PRE-prune selection and population.
    // All .item() calls are gated behind PHOTO_SLAM_TOPO_AUDIT; a normal run
    // takes the early return in topoAuditRecord and performs no extra sync.
    topoAuditInitIfNeeded();
    const bool topo_on = topoAuditEnabled();
    const long long topo_total_before = topo_on ? auditTotal() : 0;
    long long topo_line_before = -1;
    long long topo_prune_opacity = -1, topo_prune_lazy = -1, topo_prune_big = -1;
    if (topo_on)
    {
        if (is_line_.defined() && is_line_.numel() == topo_total_before)
            topo_line_before = is_line_.to(torch::kCPU).to(torch::kBool)
                                   .sum().item<int64_t>();
        topo_prune_opacity = opacity_mask.sum().item<int64_t>();
        topo_prune_lazy = lazy_line_mask.sum().item<int64_t>();
        if (max_screen_size > 0)
            topo_prune_big = big_points_mask.sum().item<int64_t>();
        topoAuditRecord("entry", topo_total_before, topo_line_before,
                        topo_prune_opacity, topo_prune_lazy, topo_prune_big,
                        0, 0, max_grad, true);
        // Full selection sets, not just counts/top-16. These are the masks used
        // by the clone/split routines at the PRE-prune snapshot.
        auto gnorm = torch::frobenius_norm(
            (xyz_gradient_accum_ / denom_.clamp_min(1e-12)).to(torch::kCPU),
            /*dim=*/-1).flatten();
        auto max_scale = std::get<0>(getScalingActivation().max(1)).to(torch::kCPU);
        auto is_line_cpu = is_line_.to(torch::kCPU).to(torch::kBool).flatten();
        auto writeMask = [&](const char* stage, const char* kind,
                             const torch::Tensor& mask) {
            auto idx = mask.to(torch::kCPU).to(torch::kBool).flatten()
                           .nonzero().squeeze(1);
            for (int64_t q = 0; q < idx.numel(); ++q)
            {
                const int row = idx.index({q}).item<int>();
                topo_mask_out_ << topo_arm_ << ',' << topo_iter_ << ',' << stage
                               << ',' << kind << ',' << row << ','
                               << gnorm.index({row}).item<double>() << ','
                               << max_scale.index({row}).item<double>() << ','
                               << (row < is_line_cpu.numel()
                                       ? (is_line_cpu.index({row})
                                                  .item<float>() > 0.5f ? 1 : 0)
                                       : -1)
                               << '\n';
            }
            topo_mask_out_.flush();
        };
        writeMask("pre_prune", "prune", prune_mask);
        auto clone_cand = torch::logical_and(
            gnorm >= max_grad, max_scale <= percentDense() * extent);
        auto split_cand = torch::logical_and(
            gnorm >= max_grad, max_scale > percentDense() * extent);
        writeMask("pre_prune", "clone_candidate", clone_cand);
        writeMask("pre_prune", "split_candidate", split_cand);
    }

    // perf-stage3 audit: measure the three sub-step deltas from HOST metadata
    // (xyz_.size(0)) only -- at this point all pending device work for the
    // masks has already completed, so no extra synchronisation is introduced.
    auditInitIfNeeded();
    const bool audit_on = auditEnabled();
    const long long audit_before_prune = audit_on ? auditTotal() : 0;
    // perf-stage3 audit: label the clone/split/prune events emitted below.
    const std::string audit_reason_prev = audit_reason_;
    if (audit_on) audit_reason_ = "densify_prune";

    auto t_prune_start = std::chrono::steady_clock::now();
    this->prunePointsWithLineAwareness(prune_mask);
    auto t_prune_end = std::chrono::steady_clock::now();
    if (audit_on) {
        const long long after = auditTotal();
        auditEvent("prune", 0, audit_before_prune - after);
    }
    const long long audit_after_prune = audit_on ? auditTotal() : 0;
    const long long topo_after_prune = topo_on ? auditTotal() : 0;
    if (topo_on)
        topoAuditRecord("after_prune", topo_total_before, topo_line_before,
                        topo_prune_opacity, topo_prune_lazy, topo_prune_big,
                        0, 0, max_grad, false);

    if (linePruneDiag()) {
        auto lazy_count = lazy_line_mask.sum().item<int>();
        auto total_before = this->xyz_.size(0);
        auto line_before_total = this->is_line_.sum().item<int64_t>();
        std::cerr << "[Pruning Report] Total Gaussians: " << total_before 
              << " | lineBefore=" << line_before_total
              << " | lineAfter=" << this->is_line_.sum().item<int64_t>()
              << " | Pruned (Lazy Line): " << lazy_count 
              << " | Remaining: " << this->xyz_.size(0) << std::endl;
    }

    // 重新计算剪裁后的梯度，用于加密逻辑
    // 因为 prune 之后原有 grads 已经失效了
    auto grads_after_prune = this->xyz_gradient_accum_ / this->denom_;
    grads_after_prune.index_put_({grads_after_prune.isnan()}, 0.0f);
    if (topo_on)
    {
        auto gnorm = torch::frobenius_norm(grads_after_prune.to(torch::kCPU),
                                           /*dim=*/-1).flatten();
        auto max_scale =
            std::get<0>(getScalingActivation().max(1)).to(torch::kCPU);
        auto is_line_cpu = is_line_.to(torch::kCPU).to(torch::kBool).flatten();
        auto writeMask = [&](const char* kind, const torch::Tensor& mask) {
            auto idx = mask.to(torch::kCPU).to(torch::kBool).flatten()
                           .nonzero().squeeze(1);
            for (int64_t q = 0; q < idx.numel(); ++q)
            {
                const int row = idx.index({q}).item<int>();
                topo_mask_out_ << topo_arm_ << ',' << topo_iter_
                               << ",post_prune," << kind << ',' << row << ','
                               << gnorm.index({row}).item<double>() << ','
                               << max_scale.index({row}).item<double>() << ','
                               << (row < is_line_cpu.numel()
                                       ? (is_line_cpu.index({row})
                                                  .item<float>() > 0.5f ? 1 : 0)
                                       : -1)
                               << '\n';
            }
            topo_mask_out_.flush();
        };
        writeMask("clone", torch::logical_and(
                               gnorm >= max_grad,
                               max_scale <= percentDense() * extent));
        writeMask("split", torch::logical_and(
                               gnorm >= max_grad,
                               max_scale > percentDense() * extent));
    }

    // =========================================================
    // 4. 执行加密 (此时 N 已变，内部会更新状态)
    // =========================================================
    auto t_dens_start = std::chrono::steady_clock::now();
    // The line-aware routines are always used: they own the line metadata
    // invariants (is_line_ / line_dir_w_ / xyz_init_ must stay aligned with
    // xyz_). useDirectionalLineDensify() only decides whether they apply
    // line-specific geometry - see the classification inside the clone/split
    // bodies. Calling the point-only routines here would grow xyz_ without
    // extending that metadata and break the next forward pass.
    this->densifyAndCloneWithLineAwareness(grads_after_prune, max_grad, extent);
    const long long audit_after_clone = audit_on ? auditTotal() : 0;
    const long long topo_after_clone = topo_on ? auditTotal() : 0;
    if (topo_on)
        topoAuditRecord("after_clone", topo_total_before, topo_line_before,
                        topo_prune_opacity, topo_prune_lazy, topo_prune_big,
                        topo_after_clone - topo_after_prune, 0, max_grad,
                        false);
    this->densifyAndSplitWithLineAwareness(grads_after_prune, max_grad, extent, 2);
    const long long topo_after_split = topo_on ? auditTotal() : 0;
    if (topo_on)
        topoAuditRecord("after_split", topo_total_before, topo_line_before,
                        topo_prune_opacity, topo_prune_lazy, topo_prune_big,
                        topo_after_clone - topo_after_prune,
                        topo_after_split - topo_after_clone, max_grad, false);
    if (audit_on) {
        const long long after_split = auditTotal();
        auditEvent("clone", audit_after_clone - audit_after_prune, 0);
        auditEvent("split", after_split - audit_after_clone, 0);
        audit_reason_ = audit_reason_prev;   // restore caller's label
    }
    auto t_dens_end = std::chrono::steady_clock::now();

    // 5. 显存清理
    auto t_cache_start = std::chrono::steady_clock::now();
    c10::cuda::CUDACachingAllocator::emptyCache();
    auto t_cache_end = std::chrono::steady_clock::now();

    if (g_dens_prof.enabled) {
        g_dens_prof.prune_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_prune_end - t_prune_start).count();
        g_dens_prof.densify_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_dens_end - t_dens_start).count();
        g_dens_prof.emptycache_ms += std::chrono::duration_cast<std::chrono::milliseconds>(t_cache_end - t_cache_start).count();
        g_dens_prof.calls += 1;
    }
}

/*
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
    //std::cerr << "[DEBUG] Before Clone: Rot Shape = " << this->rotation_.sizes() << std::endl;
    this->densifyAndCloneWithLineAwareness(grads, max_grad, extent);
    //std::cerr << "[DEBUG] After Clone: Rot Shape = " << this->rotation_.sizes() << std::endl;
    const int split_N = 2; // recommended
    //std::cerr << "[DEBUG] Before Split: Rot Shape = " << this->rotation_.sizes() << std::endl;
    this->densifyAndSplitWithLineAwareness(grads, max_grad, extent, split_N);
    //std::cerr << "[DEBUG] After Split: Rot Shape = " << this->rotation_.sizes() << std::endl;
    // 3. 【核心改进：激进剪裁掩码】
    // =========================================================
    // 3. Base pruning criteria (opacity-driven)
    // =========================================================
    auto opacity_mask = (this->getOpacityActivation() < min_opacity).view({-1});
    // 策略 B: 梯度贡献度剪裁 (仅针对线特征)
    // 如果一个线点在当前周期的平均梯度低于阈值的 5%，认为其“无贡献”
    float lazy_threshold = max_grad * 0.05f;    //added by zdg
    auto lazy_line_mask = torch::logical_and(this->is_line_, grads.squeeze() < lazy_threshold);
    // 合并剪裁掩码
    auto prune_mask = torch::logical_or(opacity_mask, lazy_line_mask);
    //auto prune_mask =
    //    (this->getOpacityActivation() < min_opacity).view({-1});
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
*/

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

torch::Tensor GaussianModelLine::computeLineShapeConstraintRef(float lambda_ecc, float lambda_ori) {
    auto line_mask = this->is_line_;
    // 不再调用 line_mask.any().item<bool>()（每次迭代的 CPU-GPU 同步）；
    // 无线高斯时布尔索引得到空张量，sum 为 0，语义不变。
    // Defensive guard: line tensors must stay size-aligned with xyz_; otherwise
    // the boolean index below reads out of bounds (CUDA illegal access).
    if (this->line_dir_w_.size(0) != this->xyz_.size(0)
        || this->scaling_.size(0) != this->xyz_.size(0)
        || this->rotation_.size(0) != this->xyz_.size(0))
        return torch::zeros({}, xyz_.options());

    // 1. 提取 Scaling
    // 注意：scaling_ 存储的是 log 空间的值
    auto s = torch::exp(this->scaling_.index({line_mask})); // [N_line, 3]
    auto s_parallel = s.index({torch::indexing::Slice(), 0}); // 沿线方向
    auto s_perpendicular = s.index({torch::indexing::Slice(), torch::indexing::Slice(1, 3)}); // 法向方向

    // 约束 1：各向异性 (希望 s_perp / s_para 趋近于 0)
    // 我们设定一个目标比例，比如希望长宽比至少是 10:1
    auto eccentricity_loss = (s_perpendicular.mean(1) / (s_parallel + 1e-6)).sum();

    // 2. 提取 Rotation 并与初始方向对齐
    // Robust quaternion normalization: clamp sum-of-squares BEFORE sqrt so the
    // sqrt backward never sees 0 (avoids NaN gradient for a degenerate quat).
    auto rot_l = this->rotation_.index({line_mask});
    auto rot_norm = torch::sqrt(torch::sum(rot_l * rot_l, 1, true).clamp_min(1e-16f));
    auto q_current = rot_l / rot_norm;

    // Vectorized: local X axis (first column of the rotation matrix) computed
    // directly from the unit quaternion (w,x,y,z), avoiding the full [N,3,3]
    // build_rotation stack (3x less autograd graph for the orientation term).
    auto qw = q_current.index({torch::indexing::Slice(), 0});
    auto qx = q_current.index({torch::indexing::Slice(), 1});
    auto qy = q_current.index({torch::indexing::Slice(), 2});
    auto qz = q_current.index({torch::indexing::Slice(), 3});
    auto local_x = torch::stack({
        1 - 2 * (qy * qy + qz * qz),
        2 * (qx * qy + qw * qz),
        2 * (qx * qz - qw * qy)}, 1); // [N_line, 3]
    auto d_init = this->line_dir_w_.index({line_mask}); // 初始线方向

    // 约束 2：方向对齐 (余弦相似度接近 1)
    auto cos_sim = torch::abs(torch::sum(local_x * d_init, 1));
    auto orientation_loss = (1.0 - cos_sim).sum();

    auto ecc_term = lambda_ecc * eccentricity_loss;
    auto ori_term = lambda_ori * orientation_loss;
    if (ORB_SLAM3::IsLineLossDiag()) {
        // Diagnostic only (PHOTO_SLAM_DEBUG_LINE_LOSS=1).
        std::cerr << "[LINE-LOSS] L_ecc=" << ecc_term.item().toFloat()
                  << " L_shape_ori=" << ori_term.item().toFloat()
                  << " n_line_gaussians=" << s.size(0)
                  << " lambda_ecc=" << lambda_ecc << " lambda_ori=" << lambda_ori << std::endl;
    }
    return ecc_term + ori_term;
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

    // A PLY carries no anchor metadata: report every row as unknown instead of
    // inventing an association.
    this->ref_kf_id_ = torch::full(
        {num_points}, -1LL,
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    this->coord_event_id_ = torch::zeros(
        {num_points},
        torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    std::cerr << "[RefKf] loadPly: all " << num_points
              << " rows have unknown reference keyframes\n";

    GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE

    this->active_sh_degree_ = this->max_sh_degree_;
}

// ============================================================================
// Fixed-geometry convergence experiment: initialisation checkpoint.
// ============================================================================
namespace {
// FNV-1a over the raw bytes of a tensor (CPU, contiguous).
void fgmHashTensor(const torch::Tensor& t, std::uint64_t& h)
{
    if (!t.defined())
        return;
    auto c = t.detach().to(torch::kCPU).contiguous();
    if (c.dtype() == torch::kBool)
        c = c.to(torch::kUInt8);
    if (!c.is_floating_point())
        c = c.to(torch::kFloat64);
    const auto* p = reinterpret_cast<const unsigned char*>(c.data_ptr());
    const std::size_t n = static_cast<std::size_t>(c.numel()) * c.element_size();
    for (std::size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ULL;
    }
}
} // namespace

void GaussianModelLine::saveInitCheckpoint(const std::filesystem::path& path)
{
    ensureGbaMetadata();
    // Order is part of the format: do not reorder without bumping the reader.
    std::vector<torch::Tensor> ts = {
        xyz_.detach().to(torch::kCPU).contiguous(),
        features_dc_.detach().to(torch::kCPU).contiguous(),
        features_rest_.detach().to(torch::kCPU).contiguous(),
        scaling_.detach().to(torch::kCPU).contiguous(),
        opacity_.detach().to(torch::kCPU).contiguous(),
        rotation_.detach().to(torch::kCPU).contiguous(),
        is_line_.detach().to(torch::kCPU).contiguous(),
        line_dir_w_.detach().to(torch::kCPU).contiguous(),
        xyz_init_.detach().to(torch::kCPU).contiguous(),
        ref_kf_id_.detach().to(torch::kCPU).contiguous(),
        coord_event_id_.detach().to(torch::kCPU).contiguous(),
    };
    std::filesystem::create_directories(path.parent_path());
    torch::save(ts, path.string());
}

std::size_t GaussianModelLine::restoreInitCheckpoint(const std::filesystem::path& path)
{
    std::vector<torch::Tensor> ts;
    try {
        torch::load(ts, path.string());
    } catch (const std::exception& e) {
        std::cerr << "[FGM] restoreInitCheckpoint failed to read " << path
                  << " : " << e.what() << std::endl;
        return 0;
    }
    // 9 = legacy format; 10/11 add the ref-keyframe and coordinate-version
    // tensors. Accept all three so old checkpoints stay readable.
    if (ts.size() < 9) {
        std::cerr << "[FGM] restoreInitCheckpoint: expected >=9 groups, got "
                  << ts.size() << std::endl;
        return 0;
    }
    torch::NoGradGuard g;
    auto put = [&](torch::Tensor& dst, const torch::Tensor& src,
                   const char* name) {
        if (!dst.defined() || dst.numel() != src.numel()) {
            throw std::runtime_error(std::string("[FGM] checkpoint/param shape "
                "mismatch for ") + name + ": dst=" +
                std::to_string(dst.defined() ? dst.numel() : -1) +
                " src=" + std::to_string(src.numel()));
        }
        dst.copy_(src.to(dst.device()).to(dst.dtype()).view(dst.sizes()));
    };
    put(xyz_, ts[0], "xyz");
    put(features_dc_, ts[1], "features_dc");
    put(features_rest_, ts[2], "features_rest");
    put(scaling_, ts[3], "scaling");
    put(opacity_, ts[4], "opacity");
    put(rotation_, ts[5], "rotation");
    put(is_line_, ts[6], "is_line");
    invalidateLineCache();   // in-place copy_(): pointer does not change
    put(line_dir_w_, ts[7], "line_dir_w");
    invalidateLineCache();   // in-place copy_(): pointer does not change
    put(xyz_init_, ts[8], "xyz_init");
    invalidateLineCache();   // in-place copy_(): pointer does not change
    if (ts.size() > 9 && ts[9].numel() == xyz_.size(0)) {
        ensureGbaMetadata();
        put(ref_kf_id_, ts[9], "ref_kf_id");
    } else {
        ensureGbaMetadata();
        ref_kf_id_.fill_(-1);
    }
    if (ts.size() > 10 && ts[10].numel() == xyz_.size(0)) {
        ensureGbaMetadata();
        put(coord_event_id_, ts[10], "coord_event_id");
    }
    return ts.size();
}

std::string GaussianModelLine::paramFingerprint()
{
    std::uint64_t h = 1469598103934665603ULL;
    fgmHashTensor(xyz_, h);
    fgmHashTensor(opacity_, h);
    fgmHashTensor(scaling_, h);
    fgmHashTensor(rotation_, h);
    fgmHashTensor(features_dc_, h);
    fgmHashTensor(is_line_, h);
    fgmHashTensor(line_dir_w_, h);
    fgmHashTensor(xyz_init_, h);
    // Counts are part of the identity: same tensor hash but a different
    // Gaussian total or line-Gaussian total is NOT the same start state.
    const auto n_all = xyz_.defined() ? xyz_.size(0) : 0;
    long long n_line = -1;
    if (is_line_.defined() && is_line_.numel() > 0)
        n_line = is_line_.to(torch::kCPU).sum().item().toLong();
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%016llx_n%lld_nline%lld",
                  static_cast<unsigned long long>(h),
                  static_cast<long long>(n_all), n_line);
    return std::string(buf);
}

void GaussianModelLine::savePly(std::filesystem::path result_path)
{
    // Empty/uninitialized model guard: if SLAM never created any Gaussians
    // (e.g. a run that ended before mapping initialised), the tensor buffers
    // are undefined or zero-sized and transpose(1,2)/size(0) below would throw
    // (uncaught -> SIGABRT). Skip writing instead of crashing.
    if (this->xyz_.dim() == 0 || this->xyz_.size(0) == 0 ||
        this->features_dc_.dim() < 3 || this->features_rest_.dim() < 3 ||
        this->scaling_.dim() < 2 || this->rotation_.dim() < 2)
    {
        std::cerr << "[GaussianModelLine] Empty/uninitialized model; skipping PLY save to "
                  << result_path << std::endl;
        return;
    }

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
    // Empty/undefined sparse cloud guard (see savePly). size(0) on an undefined
    // tensor throws; skip in that case.
    if (this->sparse_points_xyz_.dim() == 0 || this->sparse_points_xyz_.size(0) == 0)
    {
        std::cerr << "[GaussianModelLine] No sparse points cached; skipping input.ply save to "
                  << result_path << std::endl;
        return;
    }

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

// ===========================================================================
// Stage-1 performance: cached line subset + optimised structural losses.
// Everything added below is implementation-only; the expressions, the
// normalisations and the gradient paths are unchanged (verified numerically by
// PHOTO_SLAM_FGM_EQCHECK against the *_Ref implementations).
// ===========================================================================
void GaussianModelLine::refreshLineCache()
{
    const long n_is = is_line_.defined() ? is_line_.size(0) : -1;
    const long n_init = xyz_init_.defined() ? xyz_init_.size(0) : -1;
    const long n_dir = line_dir_w_.defined() ? line_dir_w_.size(0) : -1;
    const void* p_is = is_line_.defined() ? is_line_.data_ptr() : nullptr;
    const void* p_init = xyz_init_.defined() ? xyz_init_.data_ptr() : nullptr;
    const void* p_dir = line_dir_w_.defined() ? line_dir_w_.data_ptr() : nullptr;

    // Primary: the explicit version. Secondary: the (numel, data_ptr) signature,
    // which catches a writer that forgot to bump the version.
    if (line_idx_.defined() && line_cache_version_ == line_partition_version_ &&
        n_is == lc_n_is_ && n_init == lc_n_init_ && n_dir == lc_n_dir_ &&
        p_is == lc_p_is_ && p_init == lc_p_init_ && p_dir == lc_p_dir_)
        return;   // fresh: no device work at all

    torch::NoGradGuard ng;
    if (n_is > 0) {
        // torch::nonzero is the only device-count read, and it now happens once
        // per topology instead of once per boolean-mask gather.
        line_idx_ = torch::nonzero(is_line_).flatten().to(torch::kLong).contiguous();
    } else {
        line_idx_ = torch::zeros(
            {0}, torch::TensorOptions().dtype(torch::kLong).device(device_type_));
    }
    line_anchor_cached_ = (xyz_init_.defined() && xyz_init_.size(0) == n_is)
                              ? xyz_init_.index_select(0, line_idx_).contiguous()
                              : torch::Tensor();
    line_dir_cached_ = (line_dir_w_.defined() && line_dir_w_.size(0) == n_is)
                           ? line_dir_w_.index_select(0, line_idx_).contiguous()
                           : torch::Tensor();
    lc_n_is_ = n_is; lc_n_init_ = n_init; lc_n_dir_ = n_dir;
    lc_p_is_ = p_is; lc_p_init_ = p_init; lc_p_dir_ = p_dir;
    line_cache_version_ = line_partition_version_;
}

torch::Tensor GaussianModelLine::computeVectorizedLineLoss(float lambda_coherence)
{
    // Mathematically identical to computeVectorizedLineLossRef():
    //   * a zero weight returns before building any graph node;
    //   * the 3 boolean-mask gathers are replaced by 1 cached int64 index and
    //     index_select, and the two constant tensors (xyz_init_, line_dir_w_)
    //     are gathered once instead of every call.
    // The index order returned by nonzero() is row order, the same order the
    // boolean mask gathered in, so the reduction order is unchanged.
    if (lambda_coherence == 0.0f)
        return torch::zeros({}, torch::TensorOptions().device(device_type_));
    if (!this->xyz_init_.defined() || this->xyz_init_.size(0) != this->xyz_.size(0)) {
        return torch::zeros({}, torch::TensorOptions().device(device_type_));
    }
    if (this->line_dir_w_.size(0) != this->xyz_.size(0)) {
        return torch::zeros({}, torch::TensorOptions().device(device_type_));
    }

    refreshLineCache();

    auto p_m   = this->xyz_.index_select(0, line_idx_);   // [Nl,3]
    auto p_ref = line_anchor_cached_;                     // [Nl,3] constant
    auto d_m   = line_dir_cached_;                        // [Nl,3] constant

    auto ap = p_m - p_ref;
    auto dist_parallel = torch::sum(ap * d_m, /*dim=*/1, /*keepdim=*/true);
    auto projection = dist_parallel * d_m;
    auto error_perp = ap - projection;

    const float delta = 0.01f;
    auto dist_perp = torch::norm(error_perp, 2, 1, true);
    auto is_small_error = dist_perp <= delta;
    auto loss_small = 0.5 * dist_perp.pow(2);
    auto loss_large = delta * (dist_perp - 0.5 * delta);
    auto huber_loss = torch::where(is_small_error, loss_small, loss_large);

    auto line_loss = lambda_coherence * huber_loss.sum();
    if (ORB_SLAM3::IsLineLossDiag()) {
        std::cerr << "[LINE-LOSS] L_ori=" << line_loss.item().toFloat()
                  << " n_line_gaussians=" << p_m.size(0)
                  << " lambda_coherence=" << lambda_coherence << std::endl;
    }
    return line_loss;
}

torch::Tensor GaussianModelLine::computeLineShapeConstraint(
    float lambda_ecc, float lambda_ori, const torch::Tensor& scaling_activation)
{
    // Mathematically identical to computeLineShapeConstraintRef():
    //   * a cached int64 line index replaces 3 boolean-mask gathers;
    //   * exp(scaling_) is reused from the caller when supplied (the mapper has
    //     already built it for the anisotropy penalty), so the activation is not
    //     constructed twice per iteration;
    //   * each term is only built when its own weight is non-zero; adding an
    //     exact 0.0f is the identity for a finite value, so the sum is unchanged.
    if (lambda_ecc == 0.0f && lambda_ori == 0.0f)
        return torch::zeros({}, xyz_.options());
    if (this->line_dir_w_.size(0) != this->xyz_.size(0)
        || this->scaling_.size(0) != this->xyz_.size(0)
        || this->rotation_.size(0) != this->xyz_.size(0))
        return torch::zeros({}, xyz_.options());

    refreshLineCache();

    torch::Tensor ecc_term, ori_term;
    if (lambda_ecc > 0.0f) {
        torch::Tensor s;
        if (scaling_activation.defined() &&
            scaling_activation.size(0) == this->scaling_.size(0)) {
            s = scaling_activation.index_select(0, line_idx_);
        } else {
            s = torch::exp(this->scaling_).index_select(0, line_idx_);
        }
        auto s_parallel = s.index({torch::indexing::Slice(), 0});
        auto s_perpendicular =
            s.index({torch::indexing::Slice(), torch::indexing::Slice(1, 3)});
        auto eccentricity_loss = (s_perpendicular.mean(1) / (s_parallel + 1e-6)).sum();
        ecc_term = lambda_ecc * eccentricity_loss;
    }

    if (lambda_ori > 0.0f) {
        auto rot_l = this->rotation_.index_select(0, line_idx_);
        auto rot_norm =
            torch::sqrt(torch::sum(rot_l * rot_l, 1, true).clamp_min(1e-16f));
        auto q_current = rot_l / rot_norm;
        auto qw = q_current.index({torch::indexing::Slice(), 0});
        auto qx = q_current.index({torch::indexing::Slice(), 1});
        auto qy = q_current.index({torch::indexing::Slice(), 2});
        auto qz = q_current.index({torch::indexing::Slice(), 3});
        auto local_x = torch::stack({
            1 - 2 * (qy * qy + qz * qz),
            2 * (qx * qy + qw * qz),
            2 * (qx * qz - qw * qy)}, 1);
        auto d_init = line_dir_cached_;
        auto cos_sim = torch::abs(torch::sum(local_x * d_init, 1));
        ori_term = lambda_ori * (1.0 - cos_sim).sum();
    }

    if (ORB_SLAM3::IsLineLossDiag()) {
        std::cerr << "[LINE-LOSS] L_ecc=" << (ecc_term.defined() ? ecc_term.item().toFloat() : 0.0f)
                  << " L_shape_ori=" << (ori_term.defined() ? ori_term.item().toFloat() : 0.0f)
                  << " lambda_ecc=" << lambda_ecc << " lambda_ori=" << lambda_ori
                  << std::endl;
    }
    if (ecc_term.defined() && ori_term.defined())
        return ecc_term + ori_term;
    if (ecc_term.defined())
        return ecc_term;
    return ori_term;
}

// ===========================================================================
// Line-density experiment: deterministic subsetting.
// ===========================================================================
void GaussianModelLine::keepSubset(const torch::Tensor& keep_mask)
{
    torch::NoGradGuard ng;
    auto m = keep_mask.to(xyz_.device()).to(torch::kBool).flatten();
    TORCH_CHECK(m.size(0) == xyz_.size(0),
                "keepSubset: mask/parameter size mismatch");
    ensureGbaMetadata();   // size must still equal the pre-subset xyz_
    auto sel = [&](torch::Tensor& t) {
        if (t.defined() && t.size(0) == m.size(0))
            t = t.index_select(0, torch::nonzero(m).flatten()).contiguous();
        else if (t.defined() && t.size(0) != m.size(0) && t.numel() > 0)
            t = torch::Tensor();   // size-mismatched auxiliary: drop it
    };
    sel(xyz_);           xyz_ = xyz_.detach().requires_grad_();
    sel(xyz_init_);
    sel(is_line_);
    sel(line_dir_w_);
    sel(point_ids_);
    sel(features_dc_);   features_dc_ = features_dc_.detach().requires_grad_();
    if (features_rest_.defined() && features_rest_.numel() > 0) {
        sel(features_rest_);
        features_rest_ = features_rest_.detach().requires_grad_();
    }
    sel(opacity_);       opacity_ = opacity_.detach().requires_grad_();
    sel(scaling_);       scaling_ = scaling_.detach().requires_grad_();
    sel(rotation_);      rotation_ = rotation_.detach().requires_grad_();
    sel(max_radii2D_);
    sel(exist_since_iter_);
    sel(xyz_gradient_accum_);
    sel(denom_);
    sel(ref_kf_id_);
    sel(coord_event_id_);
    invalidateLineCache();          // topology changed
    rebuildParamVectors();
}

// ---------------------------------------------------------------------------
// perf-stage3: online creation-source audit (host-only, no device sync).
// Accounting basis (so the total is fully explained):
//   current_total = initial_total + add_ordinary + add_line + clone + split
//                   - prune_removed
// Note: "clone"/"split" count INSERTED rows only; the parent row of a split is
// counted once in prune_removed iff the densify path actually removed it.
// reset_opacity is recorded for traceability but creates/removes nothing.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// mapper-budget-v1: creation-batch log.
// Records every ACCEPTED creation batch in the order the model received it, so
// that a later run can replay the same geometry input stream. The record is a
// flat binary stream of little-endian records:
//     <iteration:int64> <kind_len:int32> <kind bytes> <n:int64>
//     <xyz float32[n][3]> <rgb float32[n][3]> <dir float32[n][3]>
// `dir` is zero for point batches. Only writes when the switch is set.
// ---------------------------------------------------------------------------
void GaussianModelLine::creationLogInitIfNeeded()
{
    if (clog_initialised_) return;
    clog_initialised_ = true;
    const char* p = std::getenv("PHOTO_SLAM_CREATION_LOG");
    if (p == nullptr || p[0] == '\0') { clog_enabled_ = false; return; }
    clog_out_.open(p, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!clog_out_.good()) { clog_enabled_ = false; return; }
    clog_enabled_ = true;
    std::cerr << "[CreationLog] enabled -> " << p << std::endl;
}

void GaussianModelLine::creationLogAppend(const std::string& kind,
                                          const torch::Tensor& xyz,
                                          const torch::Tensor& rgb,
                                          const torch::Tensor& dir)
{
    if (!clog_enabled_ || !xyz.defined()) return;
    const long long n = xyz.size(0);
    if (n <= 0) return;
    auto x = xyz.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
    auto c = rgb.defined() ? rgb.detach().to(torch::kCPU).to(torch::kFloat32).contiguous()
                           : torch::zeros_like(x);
    auto d = dir.defined() ? dir.detach().to(torch::kCPU).to(torch::kFloat32).contiguous()
                           : torch::zeros_like(x);
    const std::int64_t it = static_cast<std::int64_t>(audit_iter_);
    const std::int32_t kl = static_cast<std::int32_t>(kind.size());
    clog_out_.write(reinterpret_cast<const char*>(&it), sizeof(it));
    clog_out_.write(reinterpret_cast<const char*>(&kl), sizeof(kl));
    clog_out_.write(kind.data(), kl);
    const std::int64_t nn = n;
    clog_out_.write(reinterpret_cast<const char*>(&nn), sizeof(nn));
    clog_out_.write(reinterpret_cast<const char*>(x.data_ptr<float>()),
                    n * 3 * sizeof(float));
    clog_out_.write(reinterpret_cast<const char*>(c.data_ptr<float>()),
                    n * 3 * sizeof(float));
    clog_out_.write(reinterpret_cast<const char*>(d.data_ptr<float>()),
                    n * 3 * sizeof(float));
}

// Full resumable snapshot: every tensor the mapper needs to continue, not just
// the point cloud. Written with torch::save so the format is a normal tensor
// archive (loadable from Python as well).
void GaussianModelLine::saveFullSnapshot(const std::string& path)
{
    ensureGbaMetadata();
    const auto cpu = [](const torch::Tensor& t) {
        return t.defined() ? t.detach().to(torch::kCPU).contiguous() : torch::Tensor();
    };
    std::vector<torch::Tensor> ts = {
        cpu(xyz_), cpu(features_dc_), cpu(features_rest_), cpu(scaling_),
        cpu(opacity_), cpu(rotation_), cpu(is_line_), cpu(line_dir_w_),
        cpu(xyz_init_), cpu(max_radii2D_), cpu(xyz_gradient_accum_),
        cpu(denom_), cpu(exist_since_iter_), cpu(point_ids_),
        cpu(ref_kf_id_), cpu(coord_event_id_),
    };
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    torch::save(ts, path);
    linePopRecord("snapshot");     // mapper-budget-v1: population at snapshot time
}

std::size_t GaussianModelLine::loadFullSnapshot(const std::string& path)
{
    std::vector<torch::Tensor> ts;
    try {
        torch::load(ts, path);
    } catch (const std::exception& e) {
        std::cerr << "[Snapshot] load failed " << path << " : " << e.what() << std::endl;
        return 0;
    }
    if (ts.size() < 9) {
        std::cerr << "[Snapshot] " << path << ": expected >=9 groups, got "
                  << ts.size() << std::endl;
        return 0;
    }
    torch::NoGradGuard ng;
    auto put = [&](torch::Tensor& dst, const torch::Tensor& src) {
        if (!src.defined()) return;
        dst = src.to(device_type_).detach();
        // Bool/long metadata tensors cannot require gradients; the old
        // unconditional requires_grad_() made every offline snapshot load
        // abort.  Floating-point geometry/colour tensors still do.
        if (dst.is_floating_point() || dst.is_complex())
            dst.requires_grad_();
    };
    put(xyz_, ts[0]);
    put(features_dc_, ts[1]);
    put(features_rest_, ts[2]);
    put(scaling_, ts[3]);
    put(opacity_, ts[4]);
    put(rotation_, ts[5]);
    put(is_line_, ts[6]);
    put(line_dir_w_, ts[7]);
    put(xyz_init_, ts[8]);
    if (ts.size() > 9)  { max_radii2D_ = ts[9].to(device_type_); }
    if (ts.size() > 10) { xyz_gradient_accum_ = ts[10].to(device_type_); }
    if (ts.size() > 11) { denom_ = ts[11].to(device_type_); }
    if (ts.size() > 12) { exist_since_iter_ = ts[12].to(device_type_); }
    if (ts.size() > 13) { point_ids_ = ts[13].to(device_type_); }
    if (ts.size() > 14) { ref_kf_id_ = ts[14].to(device_type_).to(torch::kLong); }
    if (ts.size() > 15) { coord_event_id_ = ts[15].to(device_type_).to(torch::kLong); }
    ensureGbaMetadata();
    invalidateLineCache();
    rebuildParamVectors();
    return static_cast<std::size_t>(xyz_.size(0));
}

// ---------------------------------------------------------------------------
// mapper-budget-v1: line-population + metadata-inheritance audit.
// Answers "how many line Gaussians exist at each stage" and "did the topology
// operations inherit is_line_ / anchor (xyz_init_) / direction (line_dir_w_)
// correctly". Each record reads a device tensor, so it is capture-only.
// ---------------------------------------------------------------------------
void GaussianModelLine::linePopInitIfNeeded()
{
    if (lpop_initialised_) return;
    lpop_initialised_ = true;
    const char* p = std::getenv("PHOTO_SLAM_LINE_POP_AUDIT");
    if (p == nullptr || p[0] == '\0') { lpop_enabled_ = false; return; }
    const std::string pre(p);
    lpop_out_.open(pre + "_linepop.csv", std::ios::out | std::ios::trunc);
    lpop_struct_out_.open(pre + "_lineinherit.csv", std::ios::out | std::ios::trunc);
    if (!lpop_out_.good()) { lpop_enabled_ = false; return; }
    lpop_out_ << "iteration,stage,total,line,ordinary\n";
    lpop_struct_out_
        << "iteration,stage,aligned_line,aligned_anchor,aligned_dir,line_zero_dir_rows\n";
    lpop_enabled_ = true;
    std::cerr << "[LinePopAudit] enabled -> " << pre << "_linepop.csv" << std::endl;
}

void GaussianModelLine::linePopRecord(const char* stage)
{
    linePopInitIfNeeded();
    if (!lpop_enabled_) return;
    torch::NoGradGuard ng;
    const long long tot = auditTotal();
    long long nl = -1;
    if (is_line_.defined() && is_line_.numel() == tot)
        nl = is_line_.to(torch::kCPU).to(torch::kBool).flatten().sum().item<std::int64_t>();
    lpop_out_ << audit_iter_ << ',' << stage << ',' << tot << ',' << nl << ','
              << (tot - (nl < 0 ? 0 : nl)) << '\n';
    lpop_out_.flush();

    const long long al_line = (is_line_.defined() && is_line_.numel() == tot) ? 1 : 0;
    const long long al_anchor =
        (xyz_init_.defined() && xyz_init_.size(0) == tot) ? 1 : 0;
    const long long al_dir =
        (line_dir_w_.defined() && line_dir_w_.size(0) == tot) ? 1 : 0;
    long long zero_dir = -1;
    if (al_line && al_dir && nl > 0) {
        auto lm = is_line_.to(torch::kCPU).to(torch::kBool).flatten();
        auto dd = line_dir_w_.to(torch::kCPU).to(torch::kFloat);
        auto nrm = dd.norm(2, 1);
        zero_dir = ((nrm < 1e-6f) & lm).sum().item<std::int64_t>();
    }
    lpop_struct_out_ << audit_iter_ << ',' << stage << ',' << al_line << ','
                     << al_anchor << ',' << al_dir << ',' << zero_dir << '\n';
    lpop_struct_out_.flush();
}

// ---------------------------------------------------------------------------
// variable-topology audit (capture-only).
// Records the pre-prune selection and the post-stage population at every
// densification. Row indices are only valid for the pre-prune snapshot; after a
// prune/clone/split they must not be used to pair rows (the order is new).
// ---------------------------------------------------------------------------
void GaussianModelLine::topoAuditInitIfNeeded()
{
    if (topo_initialised_) return;
    topo_initialised_ = true;
    const char* p = std::getenv("PHOTO_SLAM_TOPO_AUDIT");
    if (p == nullptr || p[0] == '\0') return;
    topo_enabled_ = true;
    topo_out_.open(std::string(p) + "_topology.csv", std::ios::out | std::ios::trunc);
    topo_selected_out_.open(std::string(p) + "_selected.csv",
                            std::ios::out | std::ios::trunc);
    topo_mask_out_.open(std::string(p) + "_mask.csv",
                        std::ios::out | std::ios::trunc);
    if (!topo_out_.good())
    {
        topo_enabled_ = false;
        return;
    }
    topo_out_ << "arm,iteration,stage,total_before,line_before,total_after,"
                 "line_after,prune_opacity,prune_lazy,prune_big,clone_delta,"
                 "split_delta,clone_cum,split_cum,prune_cum,grad_threshold,"
                 "grad_p50,grad_p95,grad_max,near_threshold,scale_mean,"
                 "scale_p50,opacity_mean,photo_loss,struct_loss\n";
    topo_selected_out_ << "arm,iteration,stage,rank,row,grad,ratio,is_line,"
                          "max_scale,opacity\n";
    topo_mask_out_ << "arm,iteration,stage,kind,row,grad,max_scale,is_line\n";
    std::cerr << "[TopoAudit] enabled -> " << p << "_topology.csv" << std::endl;
}

void GaussianModelLine::topoAuditRecord(const char* stage,
                                        long long total_before,
                                        long long line_before,
                                        long long prune_opacity,
                                        long long prune_lazy,
                                        long long prune_big,
                                        long long clone_delta,
                                        long long split_delta,
                                        float grad_threshold,
                                        bool write_selected)
{
    topoAuditInitIfNeeded();
    if (!topo_enabled_) return;
    torch::NoGradGuard ng;
    const long long total_after = auditTotal();
    long long line_after = -1;
    if (is_line_.defined() && is_line_.numel() == total_after)
        line_after = is_line_.to(torch::kCPU).to(torch::kBool).sum().item<int64_t>();

    double p50 = -1.0, p95 = -1.0, gmax = -1.0, near = -1.0;
    double scale_mean = -1.0, scale_p50 = -1.0, opacity_mean = -1.0;
    torch::Tensor gnorm, max_scale, opacity;
    if (xyz_gradient_accum_.defined() && denom_.defined() &&
        xyz_gradient_accum_.numel() > 0 && denom_.numel() == xyz_gradient_accum_.size(0))
    {
        auto grads = (xyz_gradient_accum_ / denom_.clamp_min(1e-12)).to(torch::kCPU);
        grads.index_put_({grads.isnan()}, 0.0f);
        gnorm = torch::frobenius_norm(grads, /*dim=*/-1).flatten();
        if (gnorm.numel() > 0)
        {
            gmax = gnorm.max().item<double>();
            p50 = gnorm.quantile(0.50).item<double>();
            p95 = gnorm.quantile(0.95).item<double>();
            if (grad_threshold > 0.0f)
                near = ((gnorm >= 0.9f * grad_threshold) &
                        (gnorm <= 1.1f * grad_threshold))
                           .to(torch::kCPU).sum().item<double>();
        }
    }
    if (scaling_.defined() && scaling_.numel() > 0)
    {
        max_scale = std::get<0>(getScalingActivation().max(1)).to(torch::kCPU);
        scale_mean = max_scale.mean().item<double>();
        scale_p50 = max_scale.quantile(0.50).item<double>();
    }
    if (opacity_.defined() && opacity_.numel() > 0)
        opacity_mean = getOpacityActivation().mean().item<double>();

    topo_out_ << topo_arm_ << ',' << topo_iter_ << ',' << stage << ',' << total_before << ','
              << line_before << ',' << total_after << ',' << line_after << ','
              << prune_opacity << ',' << prune_lazy << ',' << prune_big << ','
              << clone_delta << ',' << split_delta << ',' << audit_clone_ << ','
              << audit_split_ << ',' << audit_prune_ << ',' << grad_threshold
              << ',' << p50 << ',' << p95 << ',' << gmax << ',' << near << ','
              << scale_mean << ',' << scale_p50 << ',' << opacity_mean << ','
              << last_topology_photo_loss_ << ',' << last_topology_struct_loss_
              << '\n';
    topo_out_.flush();

    if (write_selected && gnorm.defined() && gnorm.numel() > 0)
    {
        const int64_t k = std::min<int64_t>(16, gnorm.numel());
        auto vals = std::get<0>(gnorm.topk(k));
        auto idxs = std::get<1>(gnorm.topk(k));
        auto op = opacity_.defined()
                      ? getOpacityActivation().flatten().to(torch::kCPU)
                      : torch::Tensor();
        for (int64_t r = 0; r < k; ++r)
        {
            const int row = idxs.index({r}).item<int>();
            const double g = vals.index({r}).item<double>();
            double ms = -1.0, opv = -1.0;
            if (max_scale.defined() && row < max_scale.numel())
                ms = max_scale.index({row}).item<double>();
            if (op.defined() && row < op.numel())
                opv = op.index({row}).item<double>();
            const int is_l = (is_line_.defined() && row < is_line_.numel())
                                 ? (is_line_.index({row})
                                            .to(torch::kCPU)
                                            .item<float>() > 0.5f
                                        ? 1 : 0)
                                 : -1;
            topo_selected_out_ << topo_arm_ << ',' << topo_iter_ << ',' << stage << ',' << r << ','
                               << row << ',' << g << ','
                               << (grad_threshold > 0.0f ? g / grad_threshold : -1.0)
                               << ',' << is_l << ',' << ms << ',' << opv << '\n';
        }
        topo_selected_out_.flush();
    }
}

void GaussianModelLine::auditInitIfNeeded()
{
    if (audit_initialised_) return;
    audit_initialised_ = true;
    const char* p = std::getenv("PHOTO_SLAM_CREATION_AUDIT");
    if (p == nullptr || p[0] == '\0') { audit_enabled_ = false; return; }
    audit_enabled_ = true;
    audit_prefix_ = p;
    audit_events_.open(audit_prefix_ + "_events.csv", std::ios::out | std::ios::trunc);
    audit_events_ << "iteration,reason,entry,delta_in,delta_out,total_after,peak_total\n";
    audit_peak_ = auditTotal();
    std::cerr << "[CreationAudit] enabled prefix=" << audit_prefix_ << std::endl;
}

void GaussianModelLine::auditEvent(const char* entry, long long delta_in, long long delta_out)
{
    if (!audit_enabled_) return;
    const long long total = auditTotal();
    if (total > audit_peak_) audit_peak_ = total;
    const std::string e(entry);
    if (e == "add_line") {
        audit_add_line_ += delta_in;
    } else if (e == "add_ordinary") {
        audit_add_ordinary_ += delta_in;
    } else if (e == "add") {
        // Classification for the float/tensor overloads, which carry no
        // per-point source: the mapper tags the call site.
        if (audit_reason_.find("line") != std::string::npos) audit_add_line_ += delta_in;
        else audit_add_ordinary_ += delta_in;
    } else if (e == "clone") {
        audit_clone_ += delta_in;
    } else if (e == "split") {
        audit_split_ += delta_in;
    } else if (e == "prune") {
        audit_prune_ += delta_out;
    } else if (e == "reset_opacity") {
        audit_reset_ += delta_in;
    }
    ++audit_events_n_;
    audit_events_ << audit_iter_ << ',' << audit_reason_ << ',' << entry << ','
                  << delta_in << ',' << delta_out << ',' << total << ',' << audit_peak_ << '\n';
    audit_events_.flush();
    // Running summary: rewritten on every event so it survives an interrupted
    // run. Host-only file I/O; no device synchronisation anywhere in this path.
    std::ofstream s(audit_prefix_ + "_summary.txt", std::ios::out | std::ios::trunc);
    const long long net = audit_add_ordinary_ + audit_add_line_ + audit_clone_ + audit_split_ - audit_prune_;
    s << "events " << audit_events_n_ << '\n'
      << "add_ordinary " << audit_add_ordinary_ << '\n'
      << "add_line " << audit_add_line_ << '\n'
      << "clone " << audit_clone_ << '\n'
      << "split " << audit_split_ << '\n'
      << "prune_removed " << audit_prune_ << '\n'
      << "reset_opacity " << audit_reset_ << '\n'
      << "net_from_audit " << net << '\n'
      << "peak_total " << audit_peak_ << '\n'
      << "current_total " << total << '\n';
    s.flush();
    linePopRecord(entry);          // mapper-budget-v1: population after this event
}

std::string GaussianModelLine::paramFingerprintNonLine(){
    std::uint64_t h = 1469598103934665603ULL;
    torch::Tensor keep;
    {
        torch::NoGradGuard ng;
        keep = ~(is_line_.defined() ? is_line_.to(torch::kBool) : torch::zeros_like(xyz_.select(1, 0)).to(torch::kBool));
    }
    auto sub = [&](const torch::Tensor& t) {
        if (t.defined() && t.size(0) == keep.size(0))
            fgmHashTensor(t.index_select(0, torch::nonzero(keep).flatten()), h);
    };
    sub(xyz_); sub(opacity_); sub(scaling_); sub(rotation_);
    sub(features_dc_); sub(xyz_init_);
    const long long n = keep.sum().item().toLong();
    char buf[128];
    std::snprintf(buf, sizeof(buf), "%016llx_nonlin%lld",
                  static_cast<unsigned long long>(h), n);
    return std::string(buf);
}
