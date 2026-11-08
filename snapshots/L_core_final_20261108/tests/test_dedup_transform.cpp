// Production-path regression test for the Gaussian de-duplication / map
// coordinate-update consistency fixes.
//
// Unlike tests/test_line_loop_gaussian.cpp (a pure Eigen copy of the math), this
// test drives the REAL implementations:
//   * cuda_rasterizer/operate_points.h::insert_rot_to_rots  (through the kernel
//     launched by src/operate_points.cu::scaleAndTransformThenMarkVisiblePoints)
//   * GaussianModelLine::scaledTransformVisiblePointsOfKeyframe (loop correction)
//   * GaussianModelLine::applyScaledTransformation              (global map scale)
//   * GaussianSceneLine voxel grid maintenance
//
// Every case prints PASS/FAIL; the process exit code is the number of failures.
//
// Build: cmake --build build --target test_dedup_transform

#include <torch/torch.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "include/gaussian_model_line.h"
#include "include/gaussian_parameters_line.h"
#include "include/gaussian_scene_line.h"
#include "include/operate_points.h"
#include "include/point3d.h"
#include "include/tensor_utils.h"

static int g_pass = 0;
static int g_fail = 0;

static void check(bool ok, const std::string& name, const std::string& detail = "")
{
    if (ok) {
        ++g_pass;
        std::printf("[PASS] %s\n", name.c_str());
    }
    else {
        ++g_fail;
        std::printf("[FAIL] %s   %s\n", name.c_str(), detail.c_str());
    }
}

static void checkClose(const torch::Tensor& a, const torch::Tensor& b, double tol,
                       const std::string& name)
{
    auto x = a.detach().to(torch::kCPU).to(torch::kFloat64);
    auto y = b.detach().to(torch::kCPU).to(torch::kFloat64);
    if (x.sizes() != y.sizes()) {
        check(false, name, "shape mismatch");
        return;
    }
    double maxerr = (x - y).abs().max().item<double>();
    char buf[160];
    std::snprintf(buf, sizeof(buf), "max|diff|=%.3e tol=%.1e", maxerr, tol);
    check(maxerr <= tol, name, buf);
}

// Quaternion (w,x,y,z) -> rotation matrix, Hamilton convention (same as
// auxiliary.h/operate_points.h and Eigen).
static Eigen::Matrix3f quatToMat(float w, float x, float y, float z)
{
    Eigen::Quaternionf q(w, x, y, z);
    q.normalize();
    return q.toRotationMatrix();
}

static torch::Tensor mat4RowMajorTensor(const Eigen::Matrix4f& M)
{
    // Production convention: EigenMatrix2TorchTensor(...).transpose(0,1) gives a
    // row-major math matrix, which is what the CUDA kernels index.
    return tensor_utils::EigenMatrix2TorchTensor(
               Eigen::MatrixXf(M), torch::kCUDA).transpose(0, 1).contiguous();
}

static torch::Tensor affineTensor(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M.block<3, 3>(0, 0) = R;
    M.block<3, 1>(0, 3) = t;
    return mat4RowMajorTensor(M);
}

static torch::Tensor viewKeepAll()
{
    // in_frustum() only checks p_view.z > 0.2 -> translate everything +100 in z.
    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V(2, 3) = 100.0f;
    return mat4RowMajorTensor(V);
}

static torch::Tensor projIdentity()
{
    return mat4RowMajorTensor(Eigen::Matrix4f::Identity());
}

// ---------------------------------------------------------------------------
// Model fixture: 5 points, 3 of them line-sampled with distinct directions and
// deliberately non-zero z quaternion components.
// ---------------------------------------------------------------------------
struct Fixture
{
    GaussianModelLine model;
    std::vector<Eigen::Vector3f> p0;

    explicit Fixture(const torch::Tensor& rot_override = torch::Tensor())
        : model(GaussianModelParamsLine("", "", "", 3, "images", -1.0f, false, "cuda", false))
    {
        std::map<point3D_id_t, Point3D> pcd;
        p0 = { { 1.00f,  2.00f,  3.00f},
               {-1.50f,  0.50f,  2.00f},
               { 0.25f, -2.00f,  1.00f},
               { 2.00f,  1.00f, -1.50f},
               {-0.50f, -0.50f,  0.75f} };
        const bool is_line[5] = { true, true, false, true, false };
        const Eigen::Vector3f dirs[5] = {
            {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
            {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f} };

        for (int i = 0; i < 5; ++i) {
            Point3D p;
            p.xyz_ = p0[i].cast<double>();
            p.color_ = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
            p.source_ = is_line[i] ? PointSourceType::LINE_SAMPLED
                                   : PointSourceType::MAP_POINT;
            p.line_dir_ = dirs[i];
            p.sample_step_ = 0.1f;
            pcd[static_cast<point3D_id_t>(i)] = p;
        }
        model.createFromPcd(pcd, 1.0f);

        if (rot_override.defined()) {
            model.rotation_ = rot_override;
            model.Tensor_vec_rotation_ = {model.rotation_};
        }

        GaussianOptimizationParamsLine opt;
        opt.iterations_ = 30000;
        opt.densify_from_iter_ = 500;
        opt.densify_until_iter_ = 15000;
        model.trainingSetup(opt);

        // One dummy Adam step so the optimizer owns real per-parameter state;
        // replaceTensorToOptimizer() (used by the update paths) dereferences it.
        {
            torch::NoGradGuard no_grad;
            for (auto& group : model.optimizer_->param_groups())
                for (auto& p : group.params())
                    p.mutable_grad() = torch::zeros_like(p);
            model.optimizer_->step();
            model.optimizer_->zero_grad(true);
        }

        // Make every gaussian "stable" for the loop-correction selection.
        model.exist_since_iter_ = torch::zeros(
            {model.xyz_.size(0)},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }

    torch::Tensor flags()
    {
        return torch::full({model.xyz_.size(0)}, true,
                           torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    }

    void loopTransform(torch::Tensor& fl, const Eigen::Matrix3f& R,
                       const Eigen::Vector3f& t, float s, int& n_transformed)
    {
        torch::Tensor diff = affineTensor(R, t);
        torch::Tensor view = viewKeepAll();
        torch::Tensor proj = projIdentity();
        model.scaledTransformVisiblePointsOfKeyframe(
            fl, diff, view, proj, /*kf_creation_iter=*/0,
            /*stable_num_iter_existence=*/100, n_transformed, s);
    }
};

static torch::Tensor cpu(const torch::Tensor& t)
{
    return t.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
}

static Eigen::Matrix3f rotFromRow(const torch::Tensor& q)
{
    auto c = cpu(q).contiguous();
    return quatToMat(c[0].item<float>(), c[1].item<float>(),
                     c[2].item<float>(), c[3].item<float>());
}

static bool optimizerRefsMatch(GaussianModelLine& m)
{
    auto& g = m.optimizer_->param_groups();
    return g[0].params()[0].data_ptr() == m.xyz_.data_ptr()
        && g[4].params()[0].data_ptr() == m.scaling_.data_ptr()
        && g[5].params()[0].data_ptr() == m.rotation_.data_ptr();
}

int main()
{
    torch::manual_seed(7);
    if (!torch::cuda::is_available()) {
        std::printf("CUDA required\n");
        return 99;
    }

    const Eigen::Matrix3f I3 = Eigen::Matrix3f::Identity();

    // ---------------------------------------------------------------- case 1
    // Kernel write-back order: with an identity transform the stored quaternion
    // must round-trip, and it must stay unit-norm. A duplicated `rotidx + 2`
    // write loses z and makes |q| != 1.
    {
        // (w, x, y, z) with all components different and z != 0
        Eigen::Quaternionf q(0.4f, 0.5f, 0.6f, 0.678233f);
        q.normalize();
        torch::Tensor qs = torch::tensor({{q.w(), q.x(), q.y(), q.z()}},
                                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
                                .repeat({5, 1}).contiguous();
        Fixture f(qs);
        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, I3, Eigen::Vector3f::Zero(), 1.0f, n);

        Eigen::Matrix3f R_in = quatToMat(q.w(), q.x(), q.y(), q.z());
        Eigen::Matrix3f R_out = rotFromRow(f.model.rotation_[0]);
        float rot_err = (R_in - R_out).cwiseAbs().maxCoeff();
        float norm = cpu(f.model.rotation_[0]).norm().item<float>();
        char buf[200];
        std::snprintf(buf, sizeof(buf),
                      "rotation-matrix max err=%.3e, |q'|=%.6f (want 1)", rot_err, norm);
        check(rot_err < 1e-4f && std::abs(norm - 1.0f) < 1e-3f,
              "case1 identity transform keeps the quaternion (write order w,x,y,z)",
              buf);
    }

    // ---------------------------------------------------------------- case 2
    // Explicit z-only quaternion: exposed directly by the wrong index.
    {
        Eigen::Quaternionf q(0.70710678f, 0.0f, 0.0f, 0.70710678f);  // 90 deg about z
        torch::Tensor qs = torch::tensor({{q.w(), q.x(), q.y(), q.z()}},
                                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
                                .repeat({5, 1}).contiguous();
        Fixture f(qs);
        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, I3, Eigen::Vector3f::Zero(), 1.0f, n);
        auto v = cpu(f.model.rotation_[0]).contiguous();
        float y_err = std::abs(v[2].item<float>() - q.y());
        float z_err = std::abs(v[3].item<float>() - q.z());
        char buf[200];
        std::snprintf(buf, sizeof(buf), "stored=(%.4f,%.4f,%.4f,%.4f) want=(%.4f,%.4f,%.4f,%.4f)",
                      v[0].item<float>(), v[1].item<float>(), v[2].item<float>(), v[3].item<float>(),
                      q.w(), q.x(), q.y(), q.z());
        check(y_err < 1e-4f && z_err < 1e-4f,
              "case2 quaternion z component reaches rots[4i+3]", buf);
    }

    // ---------------------------------------------------------------- case 3
    // Identity: nothing changes (rotation, scale, direction, anchor, xyz).
    {
        Fixture f;
        auto xyz_before = f.model.xyz_.clone();
        auto rot_before = f.model.rotation_.clone();
        auto scl_before = f.model.getScalingActivation().clone();
        auto dir_before = f.model.line_dir_w_.clone();
        auto anc_before = f.model.xyz_init_.clone();
        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, I3, Eigen::Vector3f::Zero(), 1.0f, n);

        checkClose(f.model.xyz_, xyz_before, 1e-6, "case3 identity: xyz unchanged");
        checkClose(f.model.rotation_, rot_before, 1e-5, "case3 identity: rotation unchanged");
        checkClose(f.model.getScalingActivation(), scl_before, 1e-6, "case3 identity: sigma unchanged");
        checkClose(f.model.line_dir_w_, dir_before, 1e-6, "case3 identity: line dir unchanged");
        checkClose(f.model.xyz_init_, anc_before, 1e-6, "case3 identity: anchor unchanged");
        {
            auto fl_cpu = fl.to(torch::kCPU).contiguous();
            std::string dbg = "flags_after=";
            for (int i = 0; i < 5; ++i) dbg += (fl_cpu[i].item<bool>() ? "1" : "0");
            torch::Tensor xyz_l = f.model.xyz_;
            torch::Tensor view_l = viewKeepAll();
            torch::Tensor proj_l = projIdentity();
            torch::Tensor vis = markVisible(xyz_l, view_l, proj_l);
            auto vcpu = vis.to(torch::kCPU).contiguous();
            dbg += " visible=";
            for (int i = 0; i < 5; ++i) dbg += (vcpu[i].item<bool>() ? "1" : "0");
            auto e = f.model.exist_since_iter_.to(torch::kCPU).contiguous();
            dbg += " exist=";
            for (int i = 0; i < 5; ++i) dbg += std::to_string(e[i].item<int>()) + ",";
            check(n == 5, "case3 identity: all points selected", "n=" + std::to_string(n) + " " + dbg);
        }
        check(optimizerRefsMatch(f.model), "case3 identity: optimizer refs still valid");
    }

    // ---------------------------------------------------------------- case 4
    // Pure translation: positions and anchors move, direction/scale/rotation do not.
    {
        Fixture f;
        Eigen::Vector3f t(1.0f, -2.0f, 0.5f);
        auto xyz_before = f.model.xyz_.clone();
        auto anc_before = f.model.xyz_init_.clone();
        auto scl_before = f.model.getScalingActivation().clone();
        auto dir_before = f.model.line_dir_w_.clone();
        auto rot_before = f.model.rotation_.clone();
        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, I3, t, 1.0f, n);

        auto xyz_after = cpu(f.model.xyz_);
        auto anc_after = cpu(f.model.xyz_init_);
        auto xyz_want = cpu(xyz_before);
        auto anc_want = cpu(anc_before);
        for (int i = 0; i < 5; ++i)
            for (int d = 0; d < 3; ++d) {
                xyz_want[i][d] += t[d];
                anc_want[i][d] += t[d];
            }
        checkClose(xyz_after, xyz_want, 1e-5, "case4 translation: xyz moves");
        checkClose(anc_after, anc_want, 1e-5, "case4 translation: anchor moves with xyz");
        checkClose(f.model.line_dir_w_, dir_before, 1e-6, "case4 translation: dir unchanged");
        checkClose(f.model.getScalingActivation(), scl_before, 1e-6, "case4 translation: sigma unchanged");
        checkClose(f.model.rotation_, rot_before, 1e-5, "case4 translation: rotation unchanged");
    }

    // ---------------------------------------------------------------- case 5/6
    // 90 deg about z and a general rotation: xyz, quaternion and line dir all use
    // the same physical R (no transpose slip).
    for (int variant = 0; variant < 2; ++variant)
    {
        Fixture f;
        Eigen::Matrix3f R = (variant == 0)
            ? Eigen::Matrix3f(Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitZ()))
            : Eigen::Matrix3f(Eigen::AngleAxisf(0.7f, Eigen::Vector3f(1, 2, 3).normalized()));

        auto xyz_before = cpu(f.model.xyz_).contiguous();
        auto anc_before = cpu(f.model.xyz_init_).contiguous();
        auto rot_before = cpu(f.model.rotation_).contiguous();
        auto dir_before = cpu(f.model.line_dir_w_).contiguous();
        auto scl_before = f.model.getScalingActivation().clone();

        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, R, Eigen::Vector3f::Zero(), 1.0f, n);

        auto xyz_after = cpu(f.model.xyz_).contiguous();
        auto anc_after = cpu(f.model.xyz_init_).contiguous();
        auto rot_after = cpu(f.model.rotation_).contiguous();
        auto dir_after = cpu(f.model.line_dir_w_).contiguous();

        float e_xyz = 0, e_anc = 0, e_rot = 0, e_dir = 0;
        for (int i = 0; i < 5; ++i) {
            Eigen::Vector3f p(xyz_before[i][0].item<float>(), xyz_before[i][1].item<float>(), xyz_before[i][2].item<float>());
            Eigen::Vector3f pw = R * p;
            for (int d = 0; d < 3; ++d) {
                e_xyz = std::max(e_xyz, std::abs(pw[d] - xyz_after[i][d].item<float>()));
                e_anc = std::max(e_anc, std::abs(pw[d] - anc_after[i][d].item<float>()));
            }
            // rotation: R(q') == R * R(q)
            Eigen::Matrix3f Rq(quatToMat(rot_before[i][0].item<float>(), rot_before[i][1].item<float>(),
                                         rot_before[i][2].item<float>(), rot_before[i][3].item<float>()));
            Eigen::Matrix3f Rq2(quatToMat(rot_after[i][0].item<float>(), rot_after[i][1].item<float>(),
                                          rot_after[i][2].item<float>(), rot_after[i][3].item<float>()));
            e_rot = std::max(e_rot, (R * Rq - Rq2).cwiseAbs().maxCoeff());
            // Non-line gaussians keep the (1,0,0) placeholder direction; only
            // line-sampled ones must follow the frame change.
            if (f.model.is_line_[i].item<bool>()) {
                Eigen::Vector3f dvec(dir_before[i][0].item<float>(), dir_before[i][1].item<float>(), dir_before[i][2].item<float>());
                Eigen::Vector3f dvec2(dir_after[i][0].item<float>(), dir_after[i][1].item<float>(), dir_after[i][2].item<float>());
                e_dir = std::max(e_dir, ((R * dvec) - dvec2).cwiseAbs().maxCoeff());
            }
        }
        char buf[220];
        check(e_xyz < 1e-5f && e_anc < 1e-5f && e_rot < 1e-4f && e_dir < 1e-5f,
              variant == 0 ? "case5 rotation 90deg(z): xyz/anchor/quat/dir consistent"
                           : "case6 general rotation: xyz/anchor/quat/dir consistent", buf);
        checkClose(f.model.getScalingActivation(), scl_before, 1e-6,
                   "case5/6 rotation keeps sigma");
    }

    // ---------------------------------------------------------------- case 7
    // Loop scale 0.5 / 2: sigma must scale linearly (log-scale update).
    for (float s : {0.5f, 2.0f})
    {
        Fixture f;
        auto scl_before = cpu(f.model.getScalingActivation()).clone();
        auto xyz_before = cpu(f.model.xyz_).contiguous();
        auto anc_before = cpu(f.model.xyz_init_).contiguous();
        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, I3, Eigen::Vector3f::Zero(), s, n);
        auto scl_after = cpu(f.model.getScalingActivation()).contiguous();
        auto xyz_after = cpu(f.model.xyz_).contiguous();
        auto anc_after = cpu(f.model.xyz_init_).contiguous();

        float e_s = 0, e_xyz = 0, e_anc = 0;
        for (int i = 0; i < 5; ++i) {
            for (int d = 0; d < 3; ++d) {
                e_s = std::max(e_s, std::abs(scl_before[i][d].item<float>() * s - scl_after[i][d].item<float>()));
                e_xyz = std::max(e_xyz, std::abs(xyz_before[i][d].item<float>() * s - xyz_after[i][d].item<float>()));
                e_anc = std::max(e_anc, std::abs(anc_before[i][d].item<float>() * s - anc_after[i][d].item<float>()));
            }
        }
        char buf[200];
        std::snprintf(buf, sizeof(buf), "sigma=%.2e xyz=%.2e anchor=%.2e (s=%.2f)", e_s, e_xyz, e_anc, s);
        check(e_s < 1e-5f && e_xyz < 1e-5f && e_anc < 1e-5f,
              s < 1.0f ? "case7a scale 0.5: sigma_new = 0.5*sigma_old (log-scale)"
                       : "case7b scale 2.0: sigma_new = 2*sigma_old (log-scale)", buf);
        check(optimizerRefsMatch(f.model), "case7 scaling/rotation optimizer refs valid");
    }

    // ---------------------------------------------------------------- case 8
    // Mixed selection: pre-cleared flags must stay untouched.
    {
        Fixture f;
        torch::Tensor fl = f.flags();
        fl.index_put_({2}, false);
        fl.index_put_({4}, false);
        auto xyz_before = cpu(f.model.xyz_).contiguous();
        auto anc_before = cpu(f.model.xyz_init_).contiguous();
        auto scl_before = cpu(f.model.getScalingActivation()).contiguous();
        torch::Tensor fl2 = fl.clone();
        int n = 0;
        f.loopTransform(fl2, I3, Eigen::Vector3f(5, 5, 5), 2.0f, n);
        auto xyz_after = cpu(f.model.xyz_).contiguous();
        auto anc_after = cpu(f.model.xyz_init_).contiguous();
        auto scl_after = cpu(f.model.getScalingActivation()).contiguous();

        bool sel_ok = true, unsel_ok = true;
        for (int i = 0; i < 5; ++i) {
            bool selected = fl[i].item<bool>();
            for (int d = 0; d < 3; ++d) {
                float before = xyz_before[i][d].item<float>();
                float after = xyz_after[i][d].item<float>();
                float want = selected ? (2.0f * before + 5.0f) : before;
                if (std::abs(after - want) > 1e-5f) selected ? (sel_ok = false) : (unsel_ok = false);
                float ab = anc_before[i][d].item<float>();
                float aa = anc_after[i][d].item<float>();
                float awant = selected ? (2.0f * ab + 5.0f) : ab;
                if (std::abs(aa - awant) > 1e-5f) selected ? (sel_ok = false) : (unsel_ok = false);
                float sb = scl_before[i][d].item<float>();
                float sa = scl_after[i][d].item<float>();
                float swant = selected ? 2.0f * sb : sb;
                if (std::abs(sa - swant) > 1e-5f) selected ? (sel_ok = false) : (unsel_ok = false);
            }
        }
        check(sel_ok, "case8 selected gaussians transformed");
        check(unsel_ok, "case8 unselected gaussians untouched");
        check(n == 3, "case8 only 3 points reported transformed", "n=" + std::to_string(n));
    }

    // ---------------------------------------------------------------- case 9
    // Same flags twice: the second call must not transform anything again.
    {
        Fixture f;
        torch::Tensor fl = f.flags();
        int n1 = 0, n2 = 0;
        f.loopTransform(fl, I3, Eigen::Vector3f(1, 2, 3), 1.0f, n1);
        auto xyz_once = f.model.xyz_.clone();
        auto scl_once = f.model.scaling_.clone();
        auto anc_once = f.model.xyz_init_.clone();
        f.loopTransform(fl, I3, Eigen::Vector3f(1, 2, 3), 1.0f, n2);
        check(n1 == 5 && n2 == 0, "case9 second call transforms nothing",
              "n1=" + std::to_string(n1) + " n2=" + std::to_string(n2));
        checkClose(f.model.xyz_, xyz_once, 1e-6, "case9 xyz not transformed twice");
        checkClose(f.model.scaling_, scl_once, 1e-6, "case9 sigma not scaled twice");
        checkClose(f.model.xyz_init_, anc_once, 1e-6, "case9 anchor not transformed twice");
    }

    // ---------------------------------------------------------------- case 10
    // Transform then inverse transform recovers the original state.
    {
        Fixture f;
        Eigen::Matrix3f R(Eigen::AngleAxisf(0.6f, Eigen::Vector3f(0.3f, -1.0f, 0.5f).normalized()));
        Eigen::Vector3f t(0.5f, -0.25f, 1.0f);
        auto xyz0 = f.model.xyz_.clone();
        auto anc0 = f.model.xyz_init_.clone();
        auto scl0 = f.model.scaling_.clone();
        auto rot0 = f.model.rotation_.clone();
        auto dir0 = f.model.line_dir_w_.clone();

        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, R, t, 1.0f, n);
        torch::Tensor fl2 = f.flags();
        f.loopTransform(fl2, R.transpose(), -(R.transpose() * t), 1.0f, n);

        checkClose(f.model.xyz_, xyz0, 1e-4, "case10 roundtrip xyz");
        checkClose(f.model.xyz_init_, anc0, 1e-4, "case10 roundtrip anchor");
        checkClose(f.model.scaling_, scl0, 1e-5, "case10 roundtrip log-scale");
        auto r0 = cpu(rot0).contiguous(), r1 = cpu(f.model.rotation_).contiguous();
        float rot_err = 0;
        for (int i = 0; i < 5; ++i) {
            Eigen::Matrix3f A(quatToMat(r0[i][0].item<float>(), r0[i][1].item<float>(), r0[i][2].item<float>(), r0[i][3].item<float>()));
            Eigen::Matrix3f B(quatToMat(r1[i][0].item<float>(), r1[i][1].item<float>(), r1[i][2].item<float>(), r1[i][3].item<float>()));
            rot_err = std::max(rot_err, (A - B).cwiseAbs().maxCoeff());
        }
        char buf[120];
        std::snprintf(buf, sizeof(buf), "rot err=%.2e", rot_err);
        check(rot_err < 1e-4f, "case10 roundtrip rotation", buf);
        checkClose(f.model.line_dir_w_, dir0, 1e-4, "case10 roundtrip line direction");
    }

    // ---------------------------------------------------------------- case 11
    // Point-to-reference-line relation is preserved (loss invariance).
    {
        Fixture f;
        Eigen::Matrix3f R(Eigen::AngleAxisf(1.1f, Eigen::Vector3f(0, 1, 0)));
        Eigen::Vector3f t(0.3f, 0.2f, -0.7f);
        auto xyz0 = cpu(f.model.xyz_).contiguous();
        auto anc0 = cpu(f.model.xyz_init_).contiguous();
        auto dir0 = cpu(f.model.line_dir_w_).contiguous();
        torch::Tensor fl = f.flags();
        int n = 0;
        f.loopTransform(fl, R, t, 1.0f, n);
        auto xyz1 = cpu(f.model.xyz_).contiguous();
        auto anc1 = cpu(f.model.xyz_init_).contiguous();
        auto dir1 = cpu(f.model.line_dir_w_).contiguous();

        float max_perp_err = 0;
        for (int i = 0; i < 5; ++i) {
            Eigen::Vector3f d0(dir0[i][0].item<float>(), dir0[i][1].item<float>(), dir0[i][2].item<float>());
            if (d0.norm() < 1e-6f) continue;
            d0.normalize();
            Eigen::Vector3f P0(xyz0[i][0].item<float>(), xyz0[i][1].item<float>(), xyz0[i][2].item<float>());
            Eigen::Vector3f A0(anc0[i][0].item<float>(), anc0[i][1].item<float>(), anc0[i][2].item<float>());
            Eigen::Vector3f P1(xyz1[i][0].item<float>(), xyz1[i][1].item<float>(), xyz1[i][2].item<float>());
            Eigen::Vector3f A1(anc1[i][0].item<float>(), anc1[i][1].item<float>(), anc1[i][2].item<float>());
            Eigen::Vector3f d1(dir1[i][0].item<float>(), dir1[i][1].item<float>(), dir1[i][2].item<float>());
            d1.normalize();
            Eigen::Vector3f ap0 = P0 - A0, ap1 = P1 - A1;
            Eigen::Vector3f perp0 = ap0 - (ap0.dot(d0)) * d0;
            Eigen::Vector3f perp1 = ap1 - (ap1.dot(d1)) * d1;
            max_perp_err = std::max(max_perp_err, (perp0 - perp1).norm());
        }
        char buf[120];
        std::snprintf(buf, sizeof(buf), "max perpendicular drift=%.3e", max_perp_err);
        check(max_perp_err < 1e-5f, "case11 point-to-line perpendicular offset invariant", buf);
    }

    // ---------------------------------------------------------------- case 12
    // Global map transform (ScaleRefinement path).
    {
        Fixture f;
        Eigen::Matrix3f R(Eigen::AngleAxisf(0.4f, Eigen::Vector3f(0, 0, 1).normalized()));
        Eigen::Vector3f t(0.1f, 0.2f, 0.3f);
        Sophus::SE3f T(R, t);
        auto xyz0 = cpu(f.model.xyz_).contiguous();
        auto anc0 = cpu(f.model.xyz_init_).contiguous();
        auto scl0 = cpu(f.model.getScalingActivation()).contiguous();
        auto rot0 = cpu(f.model.rotation_).contiguous();
        auto dir0 = cpu(f.model.line_dir_w_).contiguous();
        const float s = 2.0f;

        f.model.applyScaledTransformation(s, T);

        auto xyz1 = cpu(f.model.xyz_).contiguous();
        auto anc1 = cpu(f.model.xyz_init_).contiguous();
        auto scl1 = cpu(f.model.getScalingActivation()).contiguous();
        auto rot1 = cpu(f.model.rotation_).contiguous();
        auto dir1 = cpu(f.model.line_dir_w_).contiguous();

        float e_xyz = 0, e_anc = 0, e_s = 0, e_rot = 0, e_dir = 0;
        for (int i = 0; i < 5; ++i) {
            Eigen::Vector3f P0(xyz0[i][0].item<float>(), xyz0[i][1].item<float>(), xyz0[i][2].item<float>());
            Eigen::Vector3f Pw = s * (R * P0) + t;
            Eigen::Vector3f A0(anc0[i][0].item<float>(), anc0[i][1].item<float>(), anc0[i][2].item<float>());
            Eigen::Vector3f Aw = s * (R * A0) + t;
            for (int d = 0; d < 3; ++d) {
                e_xyz = std::max(e_xyz, std::abs(Pw[d] - xyz1[i][d].item<float>()));
                e_anc = std::max(e_anc, std::abs(Aw[d] - anc1[i][d].item<float>()));
                e_s = std::max(e_s, std::abs(s * scl0[i][d].item<float>() - scl1[i][d].item<float>()));
            }
            Eigen::Matrix3f Rq(quatToMat(rot0[i][0].item<float>(), rot0[i][1].item<float>(), rot0[i][2].item<float>(), rot0[i][3].item<float>()));
            Eigen::Matrix3f Rq2(quatToMat(rot1[i][0].item<float>(), rot1[i][1].item<float>(), rot1[i][2].item<float>(), rot1[i][3].item<float>()));
            e_rot = std::max(e_rot, (R * Rq - Rq2).cwiseAbs().maxCoeff());
            Eigen::Vector3f d0(dir0[i][0].item<float>(), dir0[i][1].item<float>(), dir0[i][2].item<float>());
            Eigen::Vector3f d1(dir1[i][0].item<float>(), dir1[i][1].item<float>(), dir1[i][2].item<float>());
            if (d0.norm() > 1e-6f)
                e_dir = std::max(e_dir, ((R * d0) - d1).cwiseAbs().maxCoeff());
        }
        char buf[240];
        std::snprintf(buf, sizeof(buf), "xyz=%.2e anchor=%.2e sigma=%.2e rot=%.2e dir=%.2e",
                      e_xyz, e_anc, e_s, e_rot, e_dir);
        check(e_xyz < 1e-5f && e_anc < 1e-5f && e_s < 1e-5f && e_rot < 1e-4f && e_dir < 1e-5f,
              "case12 global scale+rotation: xyz/anchor/sigma/quat/dir consistent", buf);
        check(optimizerRefsMatch(f.model), "case12 optimizer refs valid after global transform");
    }

    // ---------------------------------------------------------------- case 13
    // Invalid scale must be REFUSED explicitly and leave no partially updated
    // state (the baseline silently degraded to a rigid transform / produced NaN).
    for (float bad : {0.0f, -1.0f, std::nanf("")})
    {
        Fixture f;
        Eigen::Matrix3f R(Eigen::AngleAxisf(0.3f, Eigen::Vector3f(1, 0, 0)));
        Eigen::Vector3f t(0.2f, 0.0f, 0.1f);
        Sophus::SE3f T(R, t);
        auto xyz0 = f.model.xyz_.clone();
        auto anc0 = f.model.xyz_init_.clone();
        auto scl0 = f.model.scaling_.clone();
        auto rot0 = f.model.rotation_.clone();

        bool threw = false;
        try { f.model.applyScaledTransformation(bad, T); }
        catch (const std::exception&) { threw = true; }

        bool unchanged = (cpu(f.model.xyz_) - cpu(xyz0)).abs().max().item<float>() == 0.0f
                      && (cpu(f.model.xyz_init_) - cpu(anc0)).abs().max().item<float>() == 0.0f
                      && (cpu(f.model.scaling_) - cpu(scl0)).abs().max().item<float>() == 0.0f
                      && (cpu(f.model.rotation_) - cpu(rot0)).abs().max().item<float>() == 0.0f;
        char buf[160];
        std::snprintf(buf, sizeof(buf), "s=%.1f threw=%d unchanged=%d",
                      static_cast<double>(bad), static_cast<int>(threw), static_cast<int>(unchanged));
        check(threw && unchanged && optimizerRefsMatch(f.model),
              "case13 invalid scale refused, nobody half-updated", buf);
    }

    // ---------------------------------------------------------------- case 14
    // Voxel occupancy semantics: live-occupancy, idempotent insert, invalidated
    // (not silently kept) when the map frame changes, rebuilt from live data.
    {
        GaussianModelParamsLine args("", "", "", 3, "images", -1.0f, false, "cuda", false);
        // Leaked on purpose: the scene owns CUDA tensors and this standalone test
        // tears the CUDA context down at exit.
        GaussianSceneLine* scene = new GaussianSceneLine(args);
        const float vox = 1.0f;
        Eigen::Vector3f p(1.3f, 2.6f, 3.1f);

        torch::Tensor xyz = torch::tensor({{p.x(), p.y(), p.z()}},
                                          torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        torch::Tensor is_line = torch::tensor({true},
                                              torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
        scene->ensureVoxelGridFresh(xyz, is_line, vox, "case14");
        check(scene->isVoxelOccupied(p, vox), "case14 voxel: accepted sample is occupied");
        check(scene->getVoxelCount() == 1, "case14 voxel: one live line gaussian -> one cell",
              "count=" + std::to_string(scene->getVoxelCount()));

        Eigen::Vector3f q(1.3f + 1.5f * vox, 2.6f, 3.1f);
        check(!scene->isVoxelOccupied(q, vox), "case14 voxel: neighbouring cell stays free");

        Eigen::Matrix3f R(Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitZ()));
        Eigen::Vector3f t(0.0f, 0.0f, 0.5f);
        Sophus::SE3f T(R, t);
        const float s = 2.0f;
        scene->applyScaledTransformation(s, T);

        Eigen::Vector3f p_moved = s * (R * p) + t;
        bool invalidated = !scene->isVoxelOccupied(p, vox) && scene->voxelGridStale();
        check(invalidated, "case14 voxel: global transform invalidates instead of keeping stale keys");

        // Rebuild from the LIVE (moved) gaussians, exactly what the mapper does
        // before the next de-duplication query.
        torch::Tensor xyz_moved = torch::tensor({{p_moved.x(), p_moved.y(), p_moved.z()}},
                                                torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA));
        scene->ensureVoxelGridFresh(xyz_moved, is_line, vox, "case14-rebuild");
        check(scene->isVoxelOccupied(p_moved, vox),
              "case14 voxel: rebuilt grid follows the new frame");

        scene->clearVoxelGrid();
        check(scene->getVoxelCount() == 0 && scene->voxelGridStale(),
              "case14 voxel: clearVoxelGrid empties keys and marks stale");
    }

    std::printf("\n==== summary: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail;
}
