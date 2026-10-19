/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2025-2026 ZDG, Zhejiang University.
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

#pragma once

#include <torch/torch.h>

#include <iostream>
#include <algorithm>
#include <functional>
#include <fstream>
#include <chrono>
#include <ctime>
#include <sstream>
#include <thread>
#include <filesystem>
#include <map>
#include <memory>
#include <random>
#include <mutex>

#include "append_refinement.h"

#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>

#include <jsoncpp/json/json.h>

#include "ORB-SLAM3/include/System.h"
#include "ORB-SLAM3/Thirdparty/Sophus/sophus/se3.hpp"

#include "operate_points.h"
#include "stereo_vision.h"
#include "tensor_utils.h"
#include "gaussian_keyframe_line.h"
#include "gaussian_scene_line.h"
#include "gaussian_trainer_line.h"

#define CHECK_DIRECTORY_AND_CREATE_IF_NOT_EXISTS_LINE(dir)                                       \
    if (!dir.empty() && !std::filesystem::exists(dir))                                      \
        if (!std::filesystem::create_directories(dir))                                      \
            throw std::runtime_error("Cannot create result directory at " + dir.string());
struct UndistortParamsLine
{
    UndistortParamsLine(
        const cv::Size& old_size,
        cv::Mat dist_coeff = (cv::Mat_<float>(1, 4) << 0.0f, 0.0f, 0.0f, 0.0f))
        : old_size_(old_size)
    {
        dist_coeff.copyTo(dist_coeff_);
    }

    cv::Size old_size_;
    cv::Mat dist_coeff_;
};

enum SystemSensorTypeLine
{
    INVALIDLINE = 0,
    MONOCULARLINE = 1,
    STEREOLINE = 2,
    RGBDLINE = 3
};

struct VariableParametersLine
{
    float position_lr_init;
    float feature_lr;
    float opacity_lr;
    float scaling_lr;
    float rotation_lr;
    float percent_dense;
    float lambda_dssim;
    int opacity_reset_interval;
    float densify_grad_th;
    int densify_interval;
    int new_kf_times_of_use;
    int stable_num_iter_existence; ///< loop closure correction

    bool keep_training;
    bool do_gaus_pyramid_training;
    bool do_inactive_geo_densify;
};

class GaussianMapperLine
{
public:
    GaussianMapperLine(
        std::shared_ptr<ORB_SLAM3::System> pSLAM,
        std::filesystem::path gaussian_config_file_path,
        std::filesystem::path result_dir = "",
        int seed = 0,
        torch::DeviceType device_type = torch::kCUDA);

    void readConfigFromFile(std::filesystem::path cfg_path);

    void run();
    void trainColmap();
    void trainForOneIteration();

    // ---- append refinement (PHOTO_SLAM_APPEND_STEPS) --------------------
    // Same control/logging mechanism as the point build, inserted after the
    // original final refinement and before shutdown export/signalStop().
    // Structural line losses stay OFF during append and the raster backend is
    // whatever the caller selected (C for the L acceptance run).
    bool appendRefinementEnabled();
    void runAppendRefinement();

    // Frozen-geometry mapper ablation. The scene produced by SLAM (keyframe
    // poses + images, sparse point cloud, line samples) is frozen once at the
    // end of the run and several mapper conditions are replayed on that
    // identical input, so that differences are attributable to the mapper
    // configuration only. Enabled by PHOTO_SLAM_FROZEN_ABLATION=<dir>.
    bool frozenAblationEnabled() const;
    void runFrozenAblation();

    //先用 Tensor 掩码法跑通流程，如果发现线段断裂严重，再引入 buildPntIdMap 做精细化的分组约束
    std::map<point3D_id_t, int> buildPntIdMap();    //To do next...

    bool isStopped();
    void signalStop(const bool going_to_stop = true);

    cv::Mat renderFromPose(
        const Sophus::SE3f &Tcw,
        const int width,
        const int height,
        const bool main_vision = false);
    
    //render with line
    cv::Mat renderFromPose_WithLine(
        const Sophus::SE3f &Tcw,
        const int width,
        const int height,
        const bool main_vision = false);

    int getIteration();
    void increaseIteration(const int inc = 1);

    float positionLearningRateInit();
    float featureLearningRate();
    float opacityLearningRate();
    float scalingLearningRate();
    float rotationLearningRate();
    float percentDense();
    float lambdaDssim();
    int opacityResetInterval();
    float densifyGradThreshold();
    int densifyInterval();
    int newKeyframeTimesOfUse();
    int stableNumIterExistence();
    bool isKeepingTraining();
    bool isdoingGausPyramidTraining();
    bool isdoingInactiveGeoDensify();

    void setPositionLearningRateInit(const float lr);
    void setFeatureLearningRate(const float lr);
    void setOpacityLearningRate(const float lr);
    void setScalingLearningRate(const float lr);
    void setRotationLearningRate(const float lr);
    void setPercentDense(const float percent_dense);
    void setLambdaDssim(const float lambda_dssim);
    void setOpacityResetInterval(const int interval);
    void setDensifyGradThreshold(const float th);
    void setDensifyInterval(const int interval);
    void setNewKeyframeTimesOfUse(const int times);
    void setStableNumIterExistence(const int niter);
    void setKeepTraining(const bool keep);
    void setDoGausPyramidTraining(const bool gaus_pyramid);
    void setDoInactiveGeoDensify(const bool inactive_geo_densify);

    VariableParametersLine getVaribleParameters();
    void setVaribleParameters(const VariableParametersLine &params);

    GaussianModelParamsLine& getGaussianModelParams() { return this->model_params_; }
    void setColmapDataPath(std::filesystem::path colmap_path) { this->model_params_.source_path_ = colmap_path; }
    void setSensorType(SystemSensorTypeLine sensor_type) { this->sensor_type_ = sensor_type; }

    void loadPly(std::filesystem::path ply_path, std::filesystem::path camera_path = "");

    void trainForOneIterationErrorGuided();

protected:
    bool hasMetInitialMappingConditions();
    bool hasMetIncrementalMappingConditions();

    void combineMappingOperations();

    void combineMappingOperations_withLine();   //Get Line 3D

    void handleNewKeyframe(std::tuple<unsigned long,
                                      unsigned long,
                                      Sophus::SE3f,
                                      cv::Mat,
                                      bool,
                                      cv::Mat,
                                      std::vector<float>,
                                      std::vector<float>,
                                      std::string,
                                      std::vector<float>> &kf);
    
    // Handle new keyframe with line information
    void handleNewKeyframe_WithLine(std::tuple<unsigned long,
                                      unsigned long,
                                      Sophus::SE3f,
                                      cv::Mat,
                                      bool,
                                      cv::Mat,
                                      std::vector<float>,
                                      std::vector<float>,
                                      std::string,
                                      std::vector<float>> &kf);                                  

    void generateKfidRandomShuffle();
    std::shared_ptr<GaussianKeyframeLine> useOneRandomSlidingWindowKeyframe();
    std::shared_ptr<GaussianKeyframeLine> useOneTrainingErrorSlidingWindowKeyframe();   //ADDED by ZDG
    std::shared_ptr<GaussianKeyframeLine> useOneRandomKeyframe();
    void increaseKeyframeTimesOfUse(std::shared_ptr<GaussianKeyframeLine> pkf, int times);
    void cullKeyframes();

    void increasePcdByKeyframeInactiveGeoDensify(
        std::shared_ptr<GaussianKeyframeLine> pkf);

    // Append-phase internals (same shared helpers as GaussianMapper).
    bool appendPreflight(std::string& reason, int& drained_batches);
    void prepareAppendViewSequence();
    std::shared_ptr<GaussianKeyframeLine> useOneAppendKeyframe();
    void appendRecordFirstStepAudit(const char* stage);
    void appendWriteBoundary(const std::filesystem::path& file, const char* stage);
    void appendWriteStatus(bool ok, const std::string& reason);
    long long appendValidatedCount();
    long long appendValidatedLineCount();

    // bool needInterruptTraining();
    // void setInterruptTraining(const bool interrupt_training);

    void recordKeyframeRendered(
        torch::Tensor &rendered,
        torch::Tensor &ground_truth,
        unsigned long kfid,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix = "");
    void renderAndRecordKeyframe(
        std::shared_ptr<GaussianKeyframeLine> pkf,
        float &dssim,
        float &psnr,
        float &psnr_gs,
        double &render_time,
        std::filesystem::path result_img_dir,
        std::filesystem::path result_gt_dir,
        std::filesystem::path result_loss_dir,
        std::string name_suffix = "");
    void renderAndRecordAllKeyframes(
        std::string name_suffix = "");
    // Stage-boundary evaluation/export wrapper: NoGradGuard + before/after
    // state comparison. Writes eval_guard_<suffix>.txt.
    void renderAndRecordAllKeyframesReadOnly(
        std::string name_suffix = "");

    void savePly(std::filesystem::path result_dir);
    void keyframesToJson(std::filesystem::path result_dir);
    void saveModelParams(std::filesystem::path result_dir);
    void writeKeyframeUsedTimes(std::filesystem::path result_dir, std::string name_suffix = "");

    // 🌟 新增：专门用于保存训练日志的函数
    void writeTrainingLog(int64_t iter_time, float ema_loss, int64_t num_points);

    void saveDebugMapToObj(const std::filesystem::path& path,
                            const std::vector<ORB_SLAM3::MapPoint*>& vpMPs,
                            const std::vector<ORB_SLAM3::MapLine*>& vpMPLs);
    

public:
    // Parameters
    std::filesystem::path config_file_path_;

    // Model
    std::shared_ptr<GaussianModelLine> gaussians_;
    std::shared_ptr<GaussianSceneLine> scene_;

    // SLAM system
    std::shared_ptr<ORB_SLAM3::System> pSLAM_;

    // Settings
    torch::DeviceType device_type_;
    int num_gaus_pyramid_sub_levels_ = 0;
    std::vector<int> kf_gaus_pyramid_times_of_use_;
    std::vector<float> kf_gaus_pyramid_factors_;

    bool viewer_camera_id_set_ = false;
    std::uint32_t viewer_camera_id_ = 0;
    float rendered_image_viewer_scale_ = 1.0f;
    float rendered_image_viewer_scale_main_ = 1.0f;

    float z_near_ = 0.01f;
    float z_far_ = 100.0f;
    float max_aniso_threshold_ = 1500.0f; // 线高斯的 anisotropic scale 不宜过大，过大会导致训练不稳定（尤其 early stage）
    float voxel_size_ = 0.25f;

    // 用于统计高斯建图的总耗时
    double total_training_time_ms_ = 0.0;
    //int training_global_refined_iters_ = 0;

    // Data
    bool kfid_shuffled_ = false;
    // Reproducibility: <0 = PHOTO_SLAM_SEED not set (original behaviour).
    int random_seed_ = -1;
    unsigned long shuffle_counter_ = 0;
    // Frozen-geometry mapper ablation state (see runFrozenAblation()).
    struct FrozenAblationCondition {
        std::string tag;
        bool use_line_gaussians = true;   // seed line-sampled Gaussians
        bool line_loss = true;            // structural (line) regularisation
        bool directional_densify = true;  // line-aware vs ordinary densify
        // Effective structural weights; < 0 means "use the configuration value".
        // Used to ablate one structural term at a time without touching the
        // formulas (each term is already scaled by its own weight).
        float w_position = -1.0f;   // weight_line_coherence_  -> perpendicular offset term
        float w_ecc = -1.0f;        // weight_line_shape_ecc_  -> eccentricity term
        float w_orient = -1.0f;     // weight_line_shape_ori_  -> local-X vs fixed line-dir term
        // -1 = follow PHOTO_SLAM_FGM_USE_REF, 1 = force the pre-optimisation
        // implementation, 0 = force the optimised one. Lets S-original and
        // S-fast run as two conditions of ONE process on ONE frozen scene.
        int force_ref = -1;
        // Line-density experiment: fraction of the EXISTING line Gaussians to
        // keep. 1.0 = keep all (default), 0.0 = drop every line Gaussian. Only
        // line rows are affected; ordinary Gaussians always keep their exact
        // initial parameters. Sets are deterministic and NESTED across fractions.
        float keep_line_frac = 1.0f;
        // mapper-budget-v2-hash: fraction of the inactive-geo candidate pool the
        // arm accepts. 1.0 = R100 (accept all), 0.5 = R50 (deterministic hash
        // half of the SAME pool). This is the ONLY difference between the arms.
        // raster-perf-v5: pin the rasteriser backward backend per condition so one
        // frozen process can compare A and C on the same scene/init/schedule.
        // -1 = do not pin (use PHOTO_SLAM_RASTER_BACKEND, default A).
        int raster_backend = -1;
        float inactive_keep = 1.0f;    };
    std::vector<FrozenAblationCondition> frozen_conditions_;
    std::filesystem::path frozen_ablation_dir_;
    int frozen_ablation_steps_ = 0;
    float frozen_line_w_coherence_ = 0.0f;
    float frozen_line_w_ecc_ = 0.0f;
    float frozen_line_w_ori_ = 0.0f;
    // REAL optimizer steps executed (counted next to optimizer_->step(),
    // independent of the loop/iteration counter).
    long long frozen_opt_steps_ = 0;

    // ---- append refinement state (PHOTO_SLAM_APPEND_STEPS) --------------
    append_refinement::Config append_config_;
    append_refinement::TimeBudgetConfig time_budget_config_;
    bool time_budget_mode_ = false;
    bool append_diag_mode_ = false;             // PHOTO_SLAM_APPEND_DIAG
    bool append_phase_ = false;                 // gated INSIDE trainForOneIterationErrorGuided()
    bool append_ok_ = false;
    std::string append_fail_reason_;
    std::string budget_status_ = "disabled";
    bool budget_unmatched_ = false;
    long long append_completed_steps_ = 0;      // after real optimizer.step()
    long long append_iterations_ = 0;           // training calls in append loop
    long long natural_completed_steps_ = 0;
    int append_start_iteration_ = 0;
    double cumulative_train_seconds_ = 0.0;     // natural + refinement + append
    double natural_cumulative_train_seconds_ = 0.0;
    double budget_overshoot_seconds_ = 0.0;
    double natural_train_seconds_ = 0.0;
    double append_train_seconds_ = 0.0;
    double append_preflight_seconds_ = 0.0;
    double append_queue_seconds_ = 0.0;
    double natural_save_seconds_ = 0.0;
    double budget_save_seconds_ = 0.0;
    double append_validation_seconds_ = 0.0;
    double shutdown_export_seconds_ = 0.0;
    std::filesystem::path append_dir_;
    std::filesystem::path natural_end_ply_dir_;
    std::filesystem::path budget_end_ply_dir_;
    int natural_end_iteration_ = -1;
    int budget_end_iteration_ = -1;
    std::vector<std::size_t> append_view_seq_;
    std::size_t append_view_cursor_ = 0;
    std::vector<std::array<long long, 3>> append_actual_views_;
    std::vector<long long> append_adam_steps_before_;
    std::vector<append_refinement::GroupGradInfo> append_first_step_grads_;
    append_refinement::ParamSnapshot append_param_snapshot_;
    std::vector<append_refinement::KeyframeState> append_kf_before_;
    std::uint64_t append_camera_hash_before_ = 0;
    long long append_gaussian_count_before_ = -1;
    long long append_line_count_before_ = -1;
    std::unique_ptr<std::ofstream> append_adam_audit_;
    // Pre-generated keyframe sampling sequence (indices into scene keyframe
    // order), generated once and replayed identically by every condition and
    // every repeat.
    std::vector<std::size_t> frozen_kf_seq_;
    bool frozen_kf_replay_ = false;
    std::size_t frozen_kf_step_ = 0;
    void preGenerateFrozenKfSequence(int n_steps);

    // ---- fixed-geometry convergence experiment (see runFrozenAblation) ----
    // Freeze the Gaussian COUNT: when true, every entry point that can change
    // the number of Gaussians (densify/clone/split/prune, opacity reset,
    // inactive-geo densify, repeated line sampling, increasePcd) is skipped.
    bool frozen_fixed_count_ = false;
    // Common evaluation views + the ordered list of completed-optimizer-step
    // values at which both variants are evaluated. Eval never mutates state.
    std::vector<std::size_t> frozen_eval_kf_seq_;
    std::vector<long long> frozen_eval_steps_;
    std::filesystem::path frozen_checkpoint_path_;
    // Renders every eval view (no grad, no optimizer/stat updates) and returns
    // mean PSNR / mean SSIM over those fixed views.
    void evalAtFrozenStep(long long opt_steps, const std::string& tag,
                          double& psnr_out, double& ssim_out, double& l1_out,
                          double& struct_ori_out, double& struct_shape_out,
                          double& eval_seconds_out);
    void assertFrozenCounts(const char* where);
    // Per-eval-point mechanism diagnostics (reads only, plus an env-gated
    // separated-gradient probe that restores the gradient buffers afterwards).
    void diagAtFrozenStep(long long opt_steps, const std::string& tag);
    // Controlled micro-benchmark (PHOTO_SLAM_FGM_MICROBENCH=1): attributes the
    // per-iteration cost between render, photometric loss, each structural term
    // and the backward pass, on ONE FIXED parameter snapshot, without ever
    // calling optimizer.step(). Used only to locate cost, never as a speed
    // result.
    void runMicroBench(const std::string& tag, long long step);
    // perf-stage3 / 三: backward decomposition (PHOTO_SLAM_FGM_BWD_DECOMP=1).
    // Measures the image loss, the renderer backward, the public constraints and
    // the full original training step as separate non-nested blocks, reports
    // host vs device time per stage, and verifies that the split parameter
    // gradients match the original full path (tolerance stated in the source).
    void runBackwardDecomposition(const std::string& tag, long long step);
    // mapper-budget-v1 capture support (all default off).
    //   PHOTO_SLAM_SNAPSHOT_AT=<a,b,...> + PHOTO_SLAM_SNAPSHOT_DIR=<dir>
    //     -> full resumable model snapshots (all tensors, not a PLY).
    //   PHOTO_SLAM_LINE_SAMPLE_AUDIT=<prefix>
    //     -> <prefix>_linesample.csv: candidate counts and per-reason rejections
    //        at the line-sampling entry, so "only 3 line seeds" is explained by
    //        data rather than by relaxing a threshold.
    void maybeSnapshotAndAudit();
    void lineSampleAuditLog(long long raw_points, long long candidates,
                            long long rejected_voxel, long long accepted);
    // ===== mapper-budget-v2-hash =============================================
    // Deterministic acceptance budget for the inactive-geo补点 pool.
    //   PHOTO_SLAM_INACTIVE_KEEP   : 1.0 = R100 (accept all), 0.5 = R50
    //   PHOTO_SLAM_INACTIVE_SEED   : fixed seed for the hash ranking
    //   PHOTO_SLAM_BUDGET_DIR      : where the kept-index CSV + hashes go
    // The ranking key is splitmix64(batch_kf_id, candidate_index, seed): a STABLE
    // hash sample, NOT image-space stratification, and it makes no claim about
    // surface coverage. Original candidate order is preserved for insertion.
    void budgetInit();
    void budgetSetCondition(float keep_frac, const std::string& dir);
    bool applyInactiveBudget(torch::Tensor& pts, torch::Tensor& cols,
                             torch::Tensor* ref_kf_ids = nullptr);
    void budgetSummary();
    // Replay driver: feeds the inactive-geo entry one frozen keyframe at a time,
    // in ascending id order, at a fixed optimizer-step period, so the SAME
    // external geometry stream is produced in every arm without the SLAM
    // keyframe thread. Switch: PHOTO_SLAM_REPLAY_INACTIVE=1.
    void replayInitInactiveDriver();
    void replayDriveInactiveGeo();
    void recordTrainStep();
    // Numerical equivalence check (PHOTO_SLAM_FGM_EQCHECK=1): evaluates the
    // optimised and the reference structural-loss implementations on ONE
    // parameter snapshot and reports max|d loss| and max|d grad| per target
    // parameter, so "mathematically equivalent" is measured, not asserted.
    void runEquivCheck(const std::string& tag, long long step);
    // Deterministic, nested, direction-stratified subset of the line Gaussians.
    // No explicit MapLine->sample association exists at freeze time (the
    // line_to_sample_ids path is unused), so line membership is approximated by
    // the quantised world direction; lines sharing a direction are
    // indistinguishable. Documented as a limitation.
    torch::Tensor makeLineSubsetMask(float keep_frac) const;
    long long frozen_count_ref_all_ = -1;
    long long frozen_count_ref_line_ = -1;
    int frozen_count_violation_ = 0;
    // Effective selector for the structural-loss implementation, resolved from
    // PHOTO_SLAM_FGM_USE_REF and overridable per condition.
    int frozen_use_ref_ = -1;
    bool useRefStructLoss() const;
    double frozen_eval_seconds_total_ = 0.0;
    // Reference init fingerprint captured from the canonical checkpoint; every
    // variant must reproduce it after restoring + before the first step.
    std::string frozen_ref_fingerprint_;
    // mapper-budget-v1 capture state
    bool lsample_initialised_ = false;
    std::ofstream lsample_out_;
    long long lsample_calls_ = 0;
    // mapper-budget-v2-hash state
    bool budget_initialised_ = false;
    float budget_keep_frac_ = 1.0f;
    std::uint64_t budget_seed_ = 20260101ULL;
    std::string budget_dir_;
    std::ofstream budget_kept_out_, budget_evt_out_, budget_step_out_;
    long long budget_cur_kf_fid_ = -1;
    long long budget_batches_ = 0, budget_cand_ = 0, budget_kept_ = 0;
    bool replay_inactive_enabled_ = false;
    std::size_t replay_kf_cursor_ = 0;
    long long replay_inactive_period_ = 200;
    std::vector<std::shared_ptr<GaussianKeyframeLine>> replay_kf_order_;
    long long budget_flush_seconds_us_ = 0;
    long long frozen_raw_iter_ref_ = -1;
    // Actually-visited training views, recorded from the live training function
    // (not from the pre-generated sequence): 4 ints per step =
    // {completed_optimizer_steps, keyframe fid, gaussian pyramid level, default_sh_}.
    std::vector<int> frozen_train_views_flat_;

    // ---- gba-recovery-v1: in-process state snapshot + two-arm experiment ----
public:
    struct RecoveryAdamState
    {
        long long step = -1;
        torch::Tensor exp_avg, exp_avg_sq;
    };
    struct RecoveryKfState
    {
        int remaining_times_of_use = 0;
        std::vector<int> pyramid_times_of_use;
        float tracking_error = 1.0f;
        bool done_inactive_densify = false;
        bool has_pose = false;
        Eigen::Quaterniond q = Eigen::Quaterniond::Identity();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
    };
    struct RecoverySnapshot
    {
        torch::Tensor xyz, xyz_init, is_line, line_dir_w, ref_kf_id, coord_event_id;
        torch::Tensor features_dc, features_rest, scaling, rotation, opacity;
        torch::Tensor point_ids, exist_since_iter, max_radii2D;
        torch::Tensor xyz_grad_accum, denom;
        std::vector<RecoveryAdamState> adam;   // 6 optimizer groups
        std::vector<float> lrs;                // 6 param-group learning rates
        long long iteration = 0;
        long long frozen_opt_steps = 0;
        int active_sh = 0;
        int default_sh = 0;
        float ema_loss = 0.0f;
        std::map<std::size_t, int> kfs_used_times;
        std::map<std::size_t, RecoveryKfState> kfs;
    };
    struct RecoveryArmResult
    {
        bool ok = false;
        double sync_seconds = 0.0;
        double train_seconds = 0.0;
        long long steps = 0;
        long long updated_rows = 0;
        long long gaussians = 0;
        long long line_gaussians = 0;
        float max_translation = 0.0f;
        double mean_translation = 0.0;
        std::string param_fp;
        std::string adam_fp;
        std::string param_fp0;   // immediately after arm sync, before step 0
        std::string adam_fp0;
        std::string schedule_fp; // iteration/SH/keyframe usage/training views
        // completed_optimizer_step -> (PSNR, SSIM)
        std::map<long long, std::pair<double, double>> eval_quality;
        // Only filled for short repeat arms (steps <= 10).
        std::map<long long, std::string> step_param_fp;
    };

    struct EvalViewRenders
    {
        std::vector<torch::Tensor> images;   // CPU float [3,H,W]
        std::vector<double> psnr, ssim;
        std::vector<std::size_t> fids;
    };
    struct DiagPoint
    {
        std::string label;
        long long step = 0, iteration = 0;
        int active_sh = 0, default_sh = 0;
        std::vector<torch::Tensor> params;    // 6 CPU clones
        std::vector<torch::Tensor> grads;     // 6 CPU clones (backward point)
        std::vector<float> lrs;               // 6 param-group LRs
        std::vector<long long> adam_steps;    // 6 Adam step counters
    };
    struct DiagArm
    {
        bool ok = false;
        std::vector<DiagPoint> points;
        EvalViewRenders render0, render1, render10;
        std::string param_fp0, adam_fp0, schedule_fp0;
        std::vector<int> view_seq;
    };
    struct LoopScopeKf
    {
        unsigned long kf_id = 0;
        double rot_deg = 0.0;
        double trans = 0.0;
        bool scene_present = false;
        Sophus::SE3f delta;
    };
    // Exactly what the first optimizer.step() of a replay arm consumes for the
    // rotation group, taken from the real parameter object (not from the model
    // member) right after the first backward.
    struct RotationAdamSample
    {
        bool valid = false;
        torch::Tensor param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq;
        torch::Tensor scaling;          // for covariance comparison
        long long step = -1;            // Adam step counter BEFORE the update
        bool state_found = false;
        std::uintptr_t param_impl = 0;
        std::uintptr_t param_data = 0;
        std::uintptr_t member_impl = 0;
        std::uintptr_t member_data = 0;
        bool grad_defined = false;
        double lr = 0.0, beta1 = 0.0, beta2 = 0.0, eps = 0.0, weight_decay = 0.0;
        bool amsgrad = false;
    };

    EvalViewRenders renderEvalViews();
    DiagPoint captureDiagPoint(const std::string& label, bool with_grads);
    std::string scheduleFingerprint() const;
    DiagArm runRepeatDiagArm(const std::string& tag, const RecoverySnapshot& base,
                             const RecoverySnapshot& kf_usage,
                             const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses);
    EvalViewRenders runStep0DiagArm(const std::string& tag, const RecoverySnapshot& base,
                                    const RecoverySnapshot& kf_usage,
                                    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
                                    const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
                                    unsigned long evt, int mode);
    void runGbaRecoveryDiagnosis(
        const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
        const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
        unsigned long evt);

    RecoverySnapshot captureRecoverySnapshot();
    void restoreRecoverySnapshot(const RecoverySnapshot& snap);
    // gba-state-audit: records the state that is actually fed to Adam, keyed by
    // the parameter's TensorImpl, plus identity checks against the model member.
    void writeAdamAudit(const char* stage);
    // gba-state-audit: replays one real Adam update offline (no render, loss or
    // event) from deep copies, and compares it with the training result.
    void runFirstUpdateReplay(const std::string& tag, const RotationAdamSample& s,
                              const DiagArm& arm);
    RecoveryArmResult runRecoveryArm(
        const std::string& tag, const RecoverySnapshot& base,
        const RecoverySnapshot& kf_usage,
        const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
        const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
        unsigned long evt, bool apply_map, int steps,
        const std::vector<long long>& eval_steps,
        const std::filesystem::path& out_dir);
    void runGbaRecoveryExperiment(
        const std::vector<std::pair<unsigned long, Sophus::SE3f>>& world_deltas,
        const std::vector<std::pair<unsigned long, Sophus::SE3f>>& after_poses,
        unsigned long evt);

    bool gba_recovery_done_ = false;
    // Diagnostic hook: ("after_backward" | "after_update"). Used only by the
    // short gba-recovery mechanism diagnosis; null in production.
    std::function<void(const char*)> recovery_diag_hook_;
    std::vector<LoopScopeKf> loop_scope_kfs_;
    bool loop_scope_available_ = false;

    // gba-state-audit (PHOTO_SLAM_GBA_UPDATE_AUDIT=<csv>): state actually fed
    // to Adam, one row per param group per captured update stage.
    std::unique_ptr<std::ofstream> adam_audit_out_;
    int adam_audit_groups_seen_ = 0;
    long long adam_audit_last_iteration_ = -1;
    long long adam_audit_step_calls_this_iter_ = 0;
    long long adam_audit_updates_ = 0;          // captured optimizer steps
    long long adam_audit_step_calls_total_ = 0;
    std::uintptr_t diag_snapshot_rot_data_ = 0; // snapshot vs live storage check
    RotationAdamSample first_bw_rot_;
    RotationAdamSample first_bw_rot_ref_;   // first arm's sample (cross-arm)
    std::filesystem::path diag_root_;       // PHOTO_SLAM_GBA_RECOVERY_DIAG root
    void writeCrossArmInputDiff(const std::string& tag,
                                const RotationAdamSample& s);

protected:
    std::map<camera_id_t, torch::Tensor> undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_main_undistort_mask_;
    std::map<camera_id_t, torch::Tensor> viewer_sub_undistort_mask_;

protected:
    // Parameters
    GaussianModelParamsLine model_params_;
    GaussianOptimizationParamsLine opt_params_;
    GaussianPipelineParamsLine pipe_params_;

    // Data
    std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>> viewpoint_sliding_window_;
    std::vector<std::size_t> kfid_shuffle_;
    std::size_t kfid_shuffle_idx_ = 0;
    std::map<std::size_t, int> kfs_used_times_;

    // Status
    bool initial_mapped_;
    bool interrupt_training_;
    bool stopped_;
    int iteration_;
    float ema_loss_for_log_;
    bool SLAM_ended_;
    bool loop_closure_iteration_;
    bool keep_training_ = false;
    int default_sh_ = 0;
    bool prof_refine_phase_ = false; ///< CUDA 事件剖分：是否处于最终精修阶段
    // Shutdown-only single-frame PSNR diagnostic (PHOTO_SLAM_PSNR_DIAG).
    bool shutdown_record_phase_ = false;
    bool psnr_diag_done_ = false;

    // Wall-clock phase timing (关键路径诊断，std::chrono 统计)
    double phase_slam_end_s_ = 0.0;    ///< SLAM 结束时刻(相对 mapper 启动)
    double phase_mapper_end_s_ = 0.0;  ///< 增量建图结束时刻
    double phase_drain_ms_ = 0.0;      ///< SLAM 结束后 queue drain 耗时
    double phase_refine_s_ = 0.0;      ///< 最终精修耗时

    // Settings
    SystemSensorTypeLine sensor_type_;

    float monocular_inactive_geo_densify_max_pixel_dist_ = 20.0;
    float stereo_baseline_length_ = 0.0f;
    int stereo_min_disparity_ = 0;
    int stereo_num_disparity_ = 128;
    cv::Mat stereo_Q_;
    cv::Ptr<cv::cuda::StereoSGM> stereo_cv_sgm_;
    float RGBD_min_depth_ = 0.0f;
    float RGBD_max_depth_ = 100.0f;

    bool inactive_geo_densify_ = true;
    int depth_cached_ = 0;
    int max_depth_cached_ = 1;
    torch::Tensor depth_cache_points_;
    torch::Tensor depth_cache_colors_;
    // Per-cached-point deformation anchor, appended in lockstep with the caches
    // above and filtered by the same inactive budget mask.
    torch::Tensor depth_cache_ref_kf_ids_;

    // geom-sync-v3: last GlobalBA event actually applied to the Gaussians.
    // Used for duplicate/stale rejection and for stamping new rows.
    unsigned long last_applied_gba_event_ = 0;
    long long lm_created_rows_total_ = 0;  // ordinary LocalMapping additions

    unsigned long min_num_initial_map_kfs_;
    torch::Tensor background_;
    float large_rot_th_;
    float large_trans_th_;
    torch::Tensor override_color_;

    int new_keyframe_times_of_use_;
    int local_BA_increased_times_of_use_;
    int loop_closure_increased_times_of_use_;

    bool cull_keyframes_;
    int stable_num_iter_existence_;

    bool do_gaus_pyramid_training_;

    std::filesystem::path result_dir_;
    int keyframe_record_interval_;
    int all_keyframes_record_interval_;
    bool record_rendered_image_;
    bool record_ground_truth_image_;
    bool record_loss_image_;

    int training_report_interval_;
    bool record_loop_ply_;

    int prune_big_point_after_iter_;
    float densify_min_opacity_ = 20;

    // Tools
    std::random_device rd_;

    // Mutex
    std::mutex mutex_status_;
    std::mutex mutex_settings_;
    std::mutex mutex_render_; ///< the model is suppose to be read-only from outside
};