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
    // Re-normalize rotation quaternions in-place (optimizer-aware). Keeps the
    // quaternion unit-norm so the normalize() backward in the line losses /
    // rasterizer never degenerates (avoids NaN gradients -> NaN rotation_).
    // in_place=true (append phase) keeps the SAME parameter tensor and Adam
    // moment buffers; the default false preserves the original behaviour.
    void normalizeRotationQuaternions(bool in_place = false);
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
    torch::Tensor computeVectorizedLineLoss(float lambda_coherence = 1.0f);
    // Reference (pre-optimisation) implementations. Kept ONLY so the
    // equivalence check can evaluate both on one parameter snapshot and compare
    // loss value and gradient exactly. Not called on any hot path.
    torch::Tensor computeVectorizedLineLossRef(float lambda_coherence);
    torch::Tensor computeLineShapeConstraintRef(float lambda_ecc, float lambda_ori);

    void oneUpShDegree();
    void setShDegree(const int sh);

    void createFromPcd(
        std::map<point3D_id_t, Point3D> pcd,
        const float spatial_lr_scale);
    void increasePcd(const std::vector<Point3D>& new_points, const int iteration);
    // ref_kf_ids is optional: one anchor keyframe id per new row. Empty/undefined
    // means "unknown" (-1) and is reported, never guessed.
    void increasePcd(std::vector<float> points, std::vector<float> colors,
                     const int iteration,
                     const torch::Tensor& ref_kf_ids = torch::Tensor());
    void increasePcd(torch::Tensor& new_point_cloud, torch::Tensor& new_colors,
                     const int iteration,
                     const torch::Tensor& ref_kf_ids = torch::Tensor());

    // ---- geom-sync-v3: apply a real LOOP/GBA correction to existing rows ----
    // world_deltas[i] = T_after^-1 * T_before maps the OLD world frame to the
    // NEW one for every Gaussian anchored at kf id world_deltas[i].first.
    // Ordinary rows: centre/rotation/anchor rigidly moved. Line rows also get
    // line_dir_w_ rotated. A non-unit world_scale applies the Sim(3) uniform
    // scale to centres/anchors and adds log(world_scale) to the log-scales.
    // Adam: the xyz first moment is transformed, the second moment is left in
    // place. `event_id` is a model-local coordination version: LOOP and GBA
    // share one monotonic sequence so each row is corrected exactly once.
    struct GbaCorrectionStats
    {
        long long moved_rows = 0;          // rows processed (selected)
        long long nonzero_moved_rows = 0;  // rows whose centre actually moved
        long long line_rows_moved = 0;     // subset of moved_rows with is_line_
        double translation_sum = 0.0;      // sum of |p' - p| over processed rows
        float max_translation = 0.0f;
        long long stale_rows = 0;         // coord_event_id_ >= event id
        long long unassigned_rows = 0;    // ref_kf_id_ < 0
        long long missing_event_rows = 0; // ref kf absent from the payload
        long long groups_applied = 0;
    };
    enum class GbaCorrectionMode { Full, MeansOnly };
    GbaCorrectionStats applyGbaCorrections(
        const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
        unsigned long event_id,
        GbaCorrectionMode mode = GbaCorrectionMode::Full,
        float world_scale = 1.0f);
    // Every coordinate-changing event (LOOP or GBA) gets a unique model-local
    // version. This is deliberately independent of ORB-SLAM3's GBA payload id,
    // which LOOP events do not carry.
    unsigned long allocateCoordEventId();
    unsigned long lastAppliedGbaEvent() const { return last_applied_gba_event_; }
    torch::Tensor refKfIds() const { return ref_kf_id_; }
    torch::Tensor coordEventIds() const { return coord_event_id_; }
    // FNV-1a over the six Adam (step, exp_avg, exp_avg_sq) states. Used to
    // prove that a recovery snapshot restored the optimiser, not just geometry.
    std::string adamFingerprint();

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
        torch::Tensor& new_line_dir_w,      // [N_new,3]
        torch::Tensor& new_xyz_init,        // [N_new, 3]
        // Optional deformation anchors [N_new] int64; empty -> -1 (unknown).
        const torch::Tensor& new_ref_kf_ids = torch::Tensor()
    );

    // Keep ref_kf_id_ / coord_event_id_ aligned with xyz_ (creates them if a
    // legacy checkpoint or PLY path did not provide them).
    void ensureGbaMetadata();

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

    void pruneTinyGaussians(float min_scale);

    void pruneIsolatedGaussians(float radius, int min_neighbors);

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

    torch::Tensor computeLineShapeConstraint(float lambda_ecc, float lambda_ori,
                                             const torch::Tensor& scaling_activation = {});

// void increasePointsIterationsOfExistence(const int i = 1);

    void loadPly(std::filesystem::path ply_path);
    void savePly(std::filesystem::path result_path);

    // ---- fixed-geometry convergence experiment: initialisation checkpoint ----
    // Serialises the initial Gaussian parameters AND the line structural
    // metadata the normal PLY does not carry (is_line_ mask, world line
    // directions, the anchor copy xyz_init_). A plain PLY cannot restore these,
    // which is exactly why the experiment must not rely on one.
    void saveInitCheckpoint(const std::filesystem::path& path);
    // Restores in place (copy_) so the optimiser keeps its references. Returns
    // the number of tensor groups restored (0 on failure).
    std::size_t restoreInitCheckpoint(const std::filesystem::path& path);
    // FNV-1a fingerprint over the live parameter tensors + structural counts.
    // Used to prove that two variants really start from the same state.
    std::string paramFingerprint();
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
    torch::Tensor xyz_init_;    // [N, 3] float
    torch::Tensor is_line_;   // [P] bool, check whether point is line sampled or not
    // 可选：每个 line Gaussian 的方向（世界系）
    torch::Tensor line_dir_w_;  // N x 3, float
    // Deformation anchor per Gaussian (keyframe id, -1 = unknown). Not a
    // deduplication key: clone/split inherit it and keep the topology legal.
    torch::Tensor ref_kf_id_;     // [N] int64
    // Coordinate version: last applied GBA event id at row creation / last
    // correction. Guards new points created in already-corrected coordinates
    // from a replayed old event.
    torch::Tensor coord_event_id_; // [N] int64
    unsigned long last_applied_gba_event_ = 0;
    unsigned long coord_event_seq_ = 0;

    // --- Stage-1 performance: cached line subset -----------------------------
    // Boolean-mask indexing on CUDA reads the true-count back from the device,
    // so every `t.index({is_line_})` costs a synchronisation. With a fixed
    // Gaussian topology the mask and the constant per-line anchors/directions do
    // not change, so the gathers are replaced by ONE cached int64 index plus
    // index_select.
    //
    // Cache key = (numel, data_ptr) of is_line_ / xyz_init_ / line_dir_w_.
    // Every writer of those three tensors REPLACES the tensor (verified: no
    // in-place mutation anywhere in this file), so a pointer change is a
    // reliable invalidation signal and a topology change cannot be missed.
    torch::Tensor line_idx_;             // [Nl] int64
    torch::Tensor line_anchor_cached_;   // [Nl,3] xyz_init_[line_idx_]
    torch::Tensor line_dir_cached_;      // [Nl,3] line_dir_w_[line_idx_]
    long lc_n_is_ = -1, lc_n_init_ = -1, lc_n_dir_ = -1;
    const void* lc_p_is_ = nullptr;
    const void* lc_p_init_ = nullptr;
    const void* lc_p_dir_ = nullptr;
    // PRIMARY invalidation mechanism: an explicit monotonic version that every
    // writer of is_line_ / xyz_init_ / line_dir_w_ bumps. numel+data_ptr alone
    // is NOT a reliable version: restoreInitCheckpoint() and loadPly() write
    // those tensors IN PLACE with copy_(), so the pointer stays the same while
    // the content changes. The pointer/size pair is kept only as a secondary
    // safety net for any writer that was missed.
    long line_partition_version_ = 0;
    long line_cache_version_ = -1;
    void invalidateLineCache() { ++line_partition_version_; }
    void refreshLineCache();
    // Replace the Gaussian set with a deterministic subset (see the density
    // experiment). Ordinary Gaussians keep their EXACT current parameters; only
    // rows are dropped. Parameter vectors are re-pointed so a caller can build a
    // fresh optimizer over the filtered tensors.
    void keepSubset(const torch::Tensor& keep_mask);
    // FNV-1a fingerprint over the NON-line Gaussians only, plus their count.
    // Used to prove that the line-density conditions share identical ordinary
    // Gaussians (they differ only in how many line Gaussians they carry).
    std::string paramFingerprintNonLine();
    // ===== Online creation-source audit (perf-stage3) =====================
    // Counts every Gaussian created / removed at the CPU entry points using
    // ONLY host-side metadata (tensor .size(0)), so it adds NO per-step device
    // synchronisation. Enabled by env PHOTO_SLAM_CREATION_AUDIT=<prefix>:
    //   <prefix>_events.csv   one row per creation event
    //   <prefix>_summary.txt  running totals (rewritten per event so the
    //                         summary survives an interrupted run)
    // Disabled by default -> zero overhead, online behaviour unchanged.
    void auditInitIfNeeded();
    void auditSetReason(const char* reason) { audit_reason_ = reason ? reason : "unknown"; }
    void auditEvent(const char* entry, long long delta_in, long long delta_out);
    long long auditTotal() const { return xyz_.defined() ? xyz_.size(0) : 0; }
    bool auditEnabled() const { return audit_enabled_; }
    long long auditIteration() const { return audit_iter_; }
    const std::string& auditReasonLabel() const { return audit_reason_; }
    // Cumulative creation-audit counters. Only meaningful when
    // PHOTO_SLAM_CREATION_AUDIT is enabled; zero otherwise.
    long long auditCloneCount() const { return audit_clone_; }
    long long auditSplitCount() const { return audit_split_; }
    long long auditPruneCount() const { return audit_prune_; }

    // ===== variable-topology audit (diagnostic only) ========================
    // PHOTO_SLAM_TOPO_AUDIT=<prefix> writes one row per densify stage plus the
    // top gradient-selected rows at the pre-prune snapshot. It is deliberately
    // capture-only: every statistic forces a device read, so it must stay off
    // in timing runs. `setLastTopologyLosses` is fed by the training loop.
    bool topoAuditEnabled() const { return topo_enabled_; }
    void setTopologyArm(const std::string& arm) { topo_arm_ = arm; }
    void setTopologyIteration(long long it) { topo_iter_ = it; }
    void setLastTopologyLosses(float photo, float structure)
    {
        last_topology_photo_loss_ = photo;
        last_topology_struct_loss_ = structure;
    }
    float lastTopologyPhotoLoss() const { return last_topology_photo_loss_; }
    float lastTopologyStructLoss() const { return last_topology_struct_loss_; }

    // ===== mapper-budget-v1: creation-batch log + full snapshot ==============
    // The creation log records every accepted creation batch (kind, iteration,
    // xyz, rgb, world direction) in the exact order the model received it, so a
    // later run can replay the SAME geometry input stream instead of
    // re-deriving it from SLAM. Switch: PHOTO_SLAM_CREATION_LOG=<path>.
    void creationLogInitIfNeeded();
    void creationLogAppend(const std::string& kind, const torch::Tensor& xyz,
                           const torch::Tensor& rgb, const torch::Tensor& dir);
    bool creationLogEnabled() const { return clog_enabled_; }
    // Full resumable snapshot (every tensor, not just a PLY).
    void saveFullSnapshot(const std::string& path);
    std::size_t loadFullSnapshot(const std::string& path);

    // ===== mapper-budget-v1: line-population + metadata inheritance ==========
    // Switch PHOTO_SLAM_LINE_POP_AUDIT=<prefix>. Each record costs one device
    // read of is_line_, so this is CAPTURE-ONLY and must stay off in any run
    // whose wall clock is used as a result. Writes two files:
    //   <prefix>_linepop.csv     iteration, stage, total, line, ordinary
    //   <prefix>_lineinherit.csv alignment of is_line_/xyz_init_/line_dir_w_
    //                            against xyz_, and line rows with a zero
    //                            direction (i.e. broken inheritance)
    void linePopInitIfNeeded();
    void linePopRecord(const char* stage);

    void rebuildParamVectors() { GAUSSIAN_MODEL_TENSORS_TO_VEC_LINE }
    // --- creation-audit state (only touched when the env switch is on) ---
    bool audit_enabled_ = false;
    bool audit_initialised_ = false;
    std::string audit_prefix_;
    std::string audit_reason_ = "map_point_seed";
    // Current training iteration, refreshed once per step by updateLearningRate
    // (a plain host-side store). Used to timestamp audit events without any
    // extra device synchronisation.
    long long audit_iter_ = -1;
    std::ofstream audit_events_;
    long long audit_add_ordinary_ = 0, audit_add_line_ = 0, audit_clone_ = 0,
              audit_split_ = 0, audit_prune_ = 0, audit_reset_ = 0,
              audit_peak_ = 0, audit_events_n_ = 0;
    // creation-batch log state
    bool clog_enabled_ = false, clog_initialised_ = false;
    std::ofstream clog_out_;
    // line-population audit state
    bool lpop_enabled_ = false, lpop_initialised_ = false;
    std::ofstream lpop_out_, lpop_struct_out_;
    // variable-topology audit state
    bool topo_enabled_ = false, topo_initialised_ = false;
    std::ofstream topo_out_, topo_selected_out_, topo_mask_out_;
    std::string topo_arm_;
    long long topo_iter_ = -1;
    float last_topology_photo_loss_ = 0.0f;
    float last_topology_struct_loss_ = 0.0f;
    void topoAuditInitIfNeeded();
    void topoAuditRecord(const char* stage, long long total_before,
                         long long line_before, long long prune_opacity,
                         long long prune_lazy, long long prune_big,
                         long long clone_delta, long long split_delta,
                         float grad_threshold, bool write_selected);
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

// 打印 densify 内部分解统计（prune/densify/emptyCache），由 GaussianMapperLine 在关闭时调用。
void printDensifyProfiler();
