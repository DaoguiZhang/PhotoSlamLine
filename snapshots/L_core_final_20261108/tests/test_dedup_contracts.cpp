// Production-path contract tests for the second round of the de-duplication /
// map-coordinate fixes. Complements tests/test_dedup_transform.cpp (which covers
// the quaternion/log-scale/direction/anchor math and is also runnable against the
// unfixed baseline tree).
//
// This file only builds against the FIXED tree because it exercises APIs that do
// not exist in the baseline:
//   * explicit rejection of invalid map transforms
//   * replaceTensorToOptimizer() with no / null / present Adam state
//   * live-occupancy voxel maintenance (invalidate / rebuild / commit)
//   * MappingOperation carrying the new write-back types
//
// Every case prints PASS/FAIL; the exit code is the number of failures.

#include <torch/torch.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "include/gaussian_model_line.h"
#include "include/gaussian_parameters_line.h"
#include "include/gaussian_scene_line.h"
#include "include/point3d.h"
#include "include/tensor_utils.h"

#include "Atlas.h"

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& name, const std::string& detail = "")
{
    if (ok) { ++g_pass; std::printf("[PASS] %s\n", name.c_str()); }
    else    { ++g_fail; std::printf("[FAIL] %s   %s\n", name.c_str(), detail.c_str()); }
}

static torch::Tensor cpuF(const torch::Tensor& t)
{
    return t.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
}

static bool sameTensor(const torch::Tensor& a, const torch::Tensor& b, double tol = 0.0)
{
    if (a.sizes() != b.sizes()) return false;
    return (cpuF(a) - cpuF(b)).abs().max().item<double>() <= tol;
}

struct Fixture
{
    GaussianModelLine model;
    explicit Fixture(bool with_step)
        : model(GaussianModelParamsLine("", "", "", 3, "images", -1.0f, false, "cuda", false))
    {
        std::map<point3D_id_t, Point3D> pcd;
        const float pts[4][3] = {{1.f, 2.f, 3.f}, {-1.5f, 0.5f, 2.f},
                                 {0.25f, -2.f, 1.f}, {2.f, 1.f, -1.5f}};
        for (int i = 0; i < 4; ++i) {
            Point3D p;
            p.xyz_ = Eigen::Vector3d(pts[i][0], pts[i][1], pts[i][2]);
            p.color_ = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
            p.source_ = (i % 2 == 0) ? PointSourceType::LINE_SAMPLED : PointSourceType::MAP_POINT;
            p.line_dir_ = (i % 2 == 0) ? Eigen::Vector3f(0.f, 0.f, 1.f) : Eigen::Vector3f::Zero();
            p.sample_step_ = 0.1f;
            pcd[static_cast<point3D_id_t>(i)] = p;
        }
        model.createFromPcd(pcd, 1.0f);
        GaussianOptimizationParamsLine opt;
        opt.iterations_ = 1000;
        opt.densify_from_iter_ = 100;
        opt.densify_until_iter_ = 500;
        model.trainingSetup(opt);
        if (with_step) {
            torch::NoGradGuard ng;
            for (auto& g : model.optimizer_->param_groups())
                for (auto& p : g.params())
                    p.mutable_grad() = torch::zeros_like(p);
            model.optimizer_->step();
            model.optimizer_->zero_grad(true);
        }
        model.exist_since_iter_ = torch::zeros(
            {model.xyz_.size(0)},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }

    bool refsOk()
    {
        auto& g = model.optimizer_->param_groups();
        return g[0].params()[0].data_ptr() == model.xyz_.data_ptr()
            && g[4].params()[0].data_ptr() == model.scaling_.data_ptr()
            && g[5].params()[0].data_ptr() == model.rotation_.data_ptr();
    }
};

int main()
{
    torch::manual_seed(11);
    if (!torch::cuda::is_available()) { std::printf("CUDA required\n"); return 99; }

    const Eigen::Matrix3f I3 = Eigen::Matrix3f::Identity();

    // ---------------------------------------------------------------- case A
    // Invalid transforms must be REJECTED (exception) and leave every piece of
    // state untouched: xyz, rotation, scaling, anchor, line dir, caches, refs.
    for (float bad : {0.0f, -1.0f, std::nanf(""), std::numeric_limits<float>::infinity()})
    {
        Fixture f(/*with_step=*/true);
        auto xyz0 = f.model.xyz_.clone();
        auto rot0 = f.model.rotation_.clone();
        auto scl0 = f.model.scaling_.clone();
        auto anc0 = f.model.xyz_init_.clone();
        auto dir0 = f.model.line_dir_w_.clone();

        bool threw = false;
        try {
            f.model.applyScaledTransformation(bad, Sophus::SE3f(I3, Eigen::Vector3f(0.3f, 0.2f, 0.1f)));
        }
        catch (const std::exception&) { threw = true; }

        bool unchanged = sameTensor(f.model.xyz_, xyz0) && sameTensor(f.model.rotation_, rot0)
                      && sameTensor(f.model.scaling_, scl0) && sameTensor(f.model.xyz_init_, anc0)
                      && sameTensor(f.model.line_dir_w_, dir0) && f.refsOk();
        char buf[160];
        std::snprintf(buf, sizeof(buf), "s=%.1f threw=%d state_unchanged=%d",
                      static_cast<double>(bad), static_cast<int>(threw), static_cast<int>(unchanged));
        check(threw && unchanged, "caseA invalid scale rejected, state untouched", buf);

        // same contract on the local (loop-correction) entry point
        torch::Tensor fl = torch::full({f.model.xyz_.size(0)}, true,
                                       torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        torch::Tensor diff = tensor_utils::EigenMatrix2TorchTensor(
            Eigen::MatrixXf(Eigen::Matrix4f::Identity()), torch::kCUDA).transpose(0, 1).contiguous();
        Eigen::Matrix4f V = Eigen::Matrix4f::Identity(); V(2, 3) = 100.0f;
        torch::Tensor view = tensor_utils::EigenMatrix2TorchTensor(
            Eigen::MatrixXf(V), torch::kCUDA).transpose(0, 1).contiguous();
        torch::Tensor proj = tensor_utils::EigenMatrix2TorchTensor(
            Eigen::MatrixXf(Eigen::Matrix4f::Identity()), torch::kCUDA).transpose(0, 1).contiguous();
        int n = 0;
        bool threw2 = false;
        try {
            f.model.scaledTransformVisiblePointsOfKeyframe(fl, diff, view, proj, 0, 100, n, bad);
        }
        catch (const std::exception&) { threw2 = true; }
        bool fl_untouched = fl.to(torch::kCPU).all().item<bool>();
        check(threw2 && fl_untouched && sameTensor(f.model.xyz_, xyz0),
              "caseA invalid scale rejected on the loop path", buf);
    }

    // ---------------------------------------------------------------- case B
    // replaceTensorToOptimizer with (a) no Adam state at all, (b) real state,
    // (c) a parameter group whose state entry is null.
    {
        Fixture f(/*with_step=*/false);      // (a)+(c): step() never ran
        bool ok = false;
        try {
            f.model.applyScaledTransformation(2.0f, Sophus::SE3f(I3, Eigen::Vector3f::Zero()));
            ok = true;
        }
        catch (const std::exception& e) { std::printf("  [info] %s\n", e.what()); }
        check(ok && f.refsOk(), "caseB no Adam state yet: update works, refs valid");
        bool finite = cpuF(f.model.xyz_).isfinite().all().item<bool>()
                   && cpuF(f.model.scaling_).isfinite().all().item<bool>();
        check(finite, "caseB no Adam state yet: no NaN/Inf");

        // the optimizer must still be usable afterwards
        torch::NoGradGuard ng;
        for (auto& g : f.model.optimizer_->param_groups())
            for (auto& p : g.params()) p.mutable_grad() = torch::zeros_like(p);
        bool stepped = true;
        try { f.model.optimizer_->step(); f.model.optimizer_->zero_grad(true); }
        catch (const std::exception&) { stepped = false; }
        check(stepped && f.refsOk(), "caseB optimizer still usable after lazy state creation");

        // (b) real state
        Fixture g(/*with_step=*/true);
        auto before = g.model.scaling_.clone();
        g.model.applyScaledTransformation(0.5f, Sophus::SE3f(I3, Eigen::Vector3f::Zero()));
        bool half_ok = (cpuF(g.model.scaling_) - (cpuF(before) + std::log(0.5f))).abs().max().item<double>() < 1e-5;
        check(half_ok && g.refsOk(), "caseB real Adam state: log-scale += log(s), refs valid");
    }

    // ---------------------------------------------------------------- case C
    // Live-occupancy voxel maintenance.
    {
        GaussianModelParamsLine args("", "", "", 3, "images", -1.0f, false, "cuda", false);
        GaussianSceneLine* scene = new GaussianSceneLine(args);   // leaked: CUDA teardown
        const float vox = 1.0f;

        // 4 gaussians: (1.3,2.6,3.1) line, (-1.5,0.5,2.0) point, (0.25,-2,1) line, (2,1,-1.5) point
        torch::Tensor xyz = torch::tensor({{1.3f, 2.6f, 3.1f}, {-1.5f, 0.5f, 2.0f},
                                           {0.25f, -2.0f, 1.0f}, {2.0f, 1.0f, -1.5f}},
                                          torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        torch::Tensor is_line = torch::tensor({true, false, true, false},
                                              torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));

        check(scene->voxelGridStale(), "caseC grid starts stale");
        scene->ensureVoxelGridFresh(xyz, is_line, vox, "test-initial");
        check(!scene->voxelGridStale() && scene->getVoxelCount() == 2,
              "caseC rebuild keeps only line gaussians (filter scope unchanged)",
              "cells=" + std::to_string(scene->getVoxelCount()));
        check(scene->isVoxelOccupied(Eigen::Vector3f(1.3f, 2.6f, 3.1f), vox),
              "caseC line gaussian position is occupied");
        check(!scene->isVoxelOccupied(Eigen::Vector3f(-1.5f, 0.5f, 2.0f), vox),
              "caseC plain point is NOT a filtering target");

        // a fresh query must not rebuild again
        auto st1 = scene->voxel_stats();
        scene->ensureVoxelGridFresh(xyz, is_line, vox, "test-again");
        check(scene->voxel_stats().rebuilds == st1.rebuilds, "caseC no rebuild when fresh");

        // invalidation is O(1) and forces exactly one rebuild
        scene->invalidateVoxelGrid("test-move");
        check(scene->voxelGridStale(), "caseC transform/prune invalidates");
        torch::Tensor moved = xyz.clone();
        moved.index_put_({0}, torch::tensor({4.3f, 5.6f, 6.1f},
                          torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA)));
        scene->ensureVoxelGridFresh(moved, is_line, vox, "test-rebuild");
        bool moved_ok = scene->isVoxelOccupied(Eigen::Vector3f(4.3f, 5.6f, 6.1f), vox)
                     && !scene->isVoxelOccupied(Eigen::Vector3f(1.3f, 2.6f, 3.1f), vox);
        check(moved_ok, "caseC rebuild follows the new coordinates");

        // commit-after-success only; duplicate commit is idempotent
        int before = scene->getVoxelCount();
        std::vector<Eigen::Vector3f> batch = {{9.3f, 9.6f, 9.1f}, {9.4f, 9.5f, 9.2f}}; // same cell
        scene->commitVoxelKeys(batch, vox);
        check(scene->getVoxelCount() == before + 1,
              "caseC same-batch duplicates collapse to one cell",
              "before=" + std::to_string(before) + " after=" + std::to_string(scene->getVoxelCount()));

        auto st2 = scene->voxel_stats();
        check(st2.rebuilds >= 2 && st2.rebuild_ms >= 0.0 && st2.rebuild_points > 0,
              "caseC rebuild cost is counted",
              "rebuilds=" + std::to_string(st2.rebuilds) + " points=" +
              std::to_string(st2.rebuild_points) + " ms=" + std::to_string(st2.rebuild_ms));
        std::printf("  [voxel-stats] rebuilds=%ld points=%ld ms=%.3f bytes=%.0f commits=%ld\n",
                    st2.rebuilds, st2.rebuild_points, st2.rebuild_ms, st2.rebuild_bytes,
                    st2.committed_keys);
    }

    // ---------------------------------------------------------------- case D
    // Producer contract: the two new write-back types really travel in a
    // MappingOperation (type + scale), and are distinct from the old ones.
    {
        ORB_SLAM3::MappingOperation gba(ORB_SLAM3::MappingOperation::OprType::GlobalBAWriteback, 1.0f);
        ORB_SLAM3::MappingOperation merge(ORB_SLAM3::MappingOperation::OprType::MapMerge, 1.0f);
        check(gba.meOperationType == ORB_SLAM3::MappingOperation::OprType::GlobalBAWriteback
              && merge.meOperationType == ORB_SLAM3::MappingOperation::OprType::MapMerge,
              "caseD write-back op types exist and are distinct");
        check(gba.mfScale == 1.0f && merge.mfScale == 1.0f,
              "caseD write-back ops carry scale 1 (rigid)");
        check(static_cast<int>(ORB_SLAM3::MappingOperation::OprType::MapMerge) !=
              static_cast<int>(ORB_SLAM3::MappingOperation::OprType::LoopClosingBA),
              "caseD MapMerge does not alias LoopClosingBA");
    }

    std::printf("\n==== summary: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail;
}
