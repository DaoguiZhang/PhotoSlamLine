// Independent test for the OFFICIAL Photo-SLAM GaussianModel (f8bfb2f).
//
// The official tree has no GaussianModelLine, so this test uses only what upstream
// actually ships:
//   * GaussianModel::createFromPcd / trainingSetup / getScalingActivation /
//     getCovarianceActivation
//   * GaussianModel::applyScaledTransformation
//   * GaussianModel::scaledTransformVisiblePointsOfKeyframe
//   * the public CUDA helper used by the loop path
//
// Expectations come from the definition of a similarity transform
// (p' = s R p + t, sigma' = s sigma, Sigma' = s^2 R Sigma R^T), Eigen quaternions,
// and one formula-free invariant (world x X, camera x X^-1 => camera-space fixed).
// Nothing here calls project-side helpers to build the expected values.
//
// One case per process:  ./review_official --case=N   (a crash only loses that case)

#include <torch/torch.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sys/stat.h>
#include <map>
#include <string>
#include <vector>

#include "include/gaussian_model.h"
#include "include/gaussian_parameters.h"
#include "include/point3d.h"
#include "include/operate_points.h"
#include "include/tensor_utils.h"

// The library actually mapped into this process. Printed so the driver can
// prove which binary produced each result instead of trusting the build tree.
static std::string loadedMapperLibrary()
{
    std::ifstream maps("/proc/self/maps");
    std::string line, best;
    while (std::getline(maps, line)) {
        if (line.find("libgaussian_mapper.so") != std::string::npos) {
            const auto sp = line.find('/');
            if (sp != std::string::npos)
                best = line.substr(sp);
        }
    }
    // strip the " (deleted)" suffix glibc can append
    const auto del = best.find(" (deleted)");
    if (del != std::string::npos)
        best = best.substr(0, del);
    return best;
}

static void reportLibraryIdentity()
{
    const std::string p = loadedMapperLibrary();
    struct stat st {};
    unsigned long long size = 0;
    long long mtime = 0;
    if (!p.empty() && ::stat(p.c_str(), &st) == 0) {
        size = static_cast<unsigned long long>(st.st_size);
        mtime = static_cast<long long>(st.st_mtime);
    }
    std::printf("[lib] mapped=%s size=%llu mtime=%lld\n", p.c_str(), size, mtime);
    std::fflush(stdout);
}

static int g_fail = 0;

static void check(bool ok, const std::string& name, const std::string& detail = "")
{
    if (ok) std::printf("[PASS] %s\n", name.c_str());
    else  { ++g_fail; std::printf("[FAIL] %s   %s\n", name.c_str(), detail.c_str()); }
}

static torch::Tensor cpuF(const torch::Tensor& t)
{
    return t.detach().to(torch::kCPU).to(torch::kFloat32).contiguous();
}

static Eigen::Matrix3f quatToMat(const torch::Tensor& q4)
{
    auto c = cpuF(q4).contiguous();
    Eigen::Quaternionf q(c[0].item<float>(), c[1].item<float>(),
                         c[2].item<float>(), c[3].item<float>());
    q.normalize();
    return q.toRotationMatrix();
}

static torch::Tensor affine(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M.block<3, 3>(0, 0) = R;
    M.block<3, 1>(0, 3) = t;
    return tensor_utils::EigenMatrix2TorchTensor(Eigen::MatrixXf(M), torch::kCUDA)
        .transpose(0, 1).contiguous();
}

static torch::Tensor viewKeepAll()
{
    Eigen::Matrix4f V = Eigen::Matrix4f::Identity();
    V(2, 3) = 100.0f;
    return tensor_utils::EigenMatrix2TorchTensor(Eigen::MatrixXf(V), torch::kCUDA)
        .transpose(0, 1).contiguous();
}

struct Fixture
{
    GaussianModel model;
    std::vector<Eigen::Vector3f> p0;

    Fixture(bool with_step)
        : model(GaussianModelParams("", "", "", 3, "images", -1.0f, false, "cuda", false))
    {
        p0 = {{1.0f, 2.0f, 3.0f}, {-1.5f, 0.5f, 2.0f},
              {0.25f, -2.0f, 1.0f}, {2.0f, 1.0f, -1.5f}};
        std::map<point3D_id_t, Point3D> pcd;
        for (int i = 0; i < 4; ++i) {
            Point3D p;
            p.xyz_ = Eigen::Vector3d(p0[i].x(), p0[i].y(), p0[i].z());
            p.color_ = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
            pcd[static_cast<point3D_id_t>(i)] = p;
        }
        model.createFromPcd(pcd, 1.0f);

        GaussianOptimizationParams opt;
        opt.iterations_ = 1000;
        opt.densify_from_iter_ = 100;
        opt.densify_until_iter_ = 500;
        // NOTE: upstream GaussianOptimizationParams has a self-initialising
        // member (`opacity_lr_(opacity_lr_)` in gaussian_parameters.cpp), so the
        // default-constructed object leaves opacity_lr_ indeterminate. The
        // production mapper always overwrites it from the YAML, but a direct
        // construction must not depend on that garbage: an out-of-float-range
        // value makes torch's Adam abort on its first step with
        // "value cannot be converted to type float without overflow".
        opt.opacity_lr_ = 0.05f;
        model.trainingSetup(opt);

        if (with_step) {
            torch::NoGradGuard ng;
            for (auto& g : model.optimizer_->param_groups())
                for (auto& p : g.params()) p.mutable_grad() = torch::zeros_like(p);
            model.optimizer_->step();
            model.optimizer_->zero_grad(true);
        }
        model.exist_since_iter_ = torch::zeros(
            {model.xyz_.size(0)},
            torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    }

    torch::Tensor allFlags()
    {
        return torch::full({model.xyz_.size(0)}, true,
                           torch::TensorOptions().dtype(torch::kBool).device(torch::kCUDA));
    }

    void loopTransform(torch::Tensor& fl, const Eigen::Matrix3f& R,
                       const Eigen::Vector3f& t, float s, int& n)
    {
        torch::Tensor diff = affine(R, t);
        torch::Tensor view = viewKeepAll();
        torch::Tensor proj = affine(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
        model.scaledTransformVisiblePointsOfKeyframe(fl, diff, view, proj, 0, 100, n, s);
    }

    bool refsOk()
    {
        auto& g = model.optimizer_->param_groups();
        return g[0].params()[0].data_ptr() == model.xyz_.data_ptr()
            && g[4].params()[0].data_ptr() == model.scaling_.data_ptr()
            && g[5].params()[0].data_ptr() == model.rotation_.data_ptr();
    }
};

int main(int argc, char** argv)
{
    torch::manual_seed(5);
    int c = -1;
    for (int i = 1; i < argc; ++i)
        if (std::strncmp(argv[i], "--case=", 7) == 0) c = std::atoi(argv[i] + 7);
    reportLibraryIdentity();
    if (!torch::cuda::is_available()) { std::printf("CUDA required\n"); return 99; }
    const Eigen::Matrix3f I3 = Eigen::Matrix3f::Identity();

    if (c == 1) {   // quaternion write-back, identity transform
        Eigen::Quaternionf q(0.4f, 0.5f, 0.6f, 0.678233f);
        q.normalize();
        Fixture f(true);
        torch::Tensor qs = torch::tensor({{q.w(), q.x(), q.y(), q.z()}},
                                         torch::TensorOptions().dtype(torch::kFloat).device(torch::kCUDA))
                                .repeat({4, 1}).contiguous();
        f.model.rotation_ = qs.clone();
        f.model.Tensor_vec_rotation_ = {f.model.rotation_};
        torch::Tensor fl = f.allFlags();
        int n = 0;
        f.loopTransform(fl, I3, Eigen::Vector3f::Zero(), 1.0f, n);
        auto v = cpuF(f.model.rotation_[0]).contiguous();
        Eigen::Matrix3f Rin = quatToMat(qs[0]);
        Eigen::Matrix3f Rout = quatToMat(f.model.rotation_[0]);
        float err = (Rin - Rout).cwiseAbs().maxCoeff();
        float norm = cpuF(f.model.rotation_[0]).norm().item<float>();
        char buf[200];
        std::snprintf(buf, sizeof(buf), "stored=(%.4f,%.4f,%.4f,%.4f) rotErr=%.3e |q'|=%.4f",
                      v[0].item<float>(), v[1].item<float>(), v[2].item<float>(), v[3].item<float>(),
                      err, norm);
        check(err < 1e-4f && std::abs(norm - 1.f) < 1e-3f,
              "O1 identity transform keeps the quaternion (w,x,y,z)", buf);
    }
    else if (c == 2) {   // identity: nothing changes
        Fixture f(true);
        auto xyz0 = f.model.xyz_.clone(); auto rot0 = f.model.rotation_.clone();
        auto sig0 = f.model.getScalingActivation().clone();
        torch::Tensor fl = f.allFlags(); int n = 0;
        f.loopTransform(fl, I3, Eigen::Vector3f::Zero(), 1.0f, n);
        float ex = (cpuF(f.model.xyz_) - cpuF(xyz0)).abs().max().item<float>();
        float es = (cpuF(f.model.getScalingActivation()) - cpuF(sig0)).abs().max().item<float>();
        float er = (cpuF(f.model.rotation_) - cpuF(rot0)).abs().max().item<float>();
        char buf[160];
        std::snprintf(buf, sizeof(buf), "xyz=%.2e sigma=%.2e rot=%.2e", ex, es, er);
        check(ex < 1e-6f && es < 1e-6f && er < 1e-5f, "O2 identity is a no-op", buf);
    }
    else if (c == 3) {   // pure translation
        Fixture f(true);
        Eigen::Vector3f t(1.f, -2.f, 0.5f);
        auto xyz0 = cpuF(f.model.xyz_).clone();
        auto sig0 = cpuF(f.model.getScalingActivation()).clone();
        auto rot0 = cpuF(f.model.rotation_).clone();
        torch::Tensor fl = f.allFlags(); int n = 0;
        f.loopTransform(fl, I3, t, 1.0f, n);
        auto xyz1 = cpuF(f.model.xyz_).contiguous();
        float ex = 0;
        for (int i = 0; i < 4; ++i)
            for (int d = 0; d < 3; ++d)
                ex = std::max(ex, std::abs(xyz0[i][d].item<float>() + t[d] - xyz1[i][d].item<float>()));
        float es = (cpuF(f.model.getScalingActivation()) - sig0).abs().max().item<float>();
        float er = (cpuF(f.model.rotation_) - rot0).abs().max().item<float>();
        char buf[160];
        std::snprintf(buf, sizeof(buf), "xyz=%.2e sigma=%.2e rot=%.2e", ex, es, er);
        check(ex < 1e-5f && es < 1e-6f && er < 1e-5f, "O3 translation moves centres only", buf);
    }
    else if (c == 4) {   // global s=2 (applyScaledTransformation)
        Fixture f(true);
        auto xyz0 = cpuF(f.model.xyz_).clone();
        auto sig0 = cpuF(f.model.getScalingActivation()).clone();
        auto cov0 = cpuF(f.model.getCovarianceActivation()).clone();
        f.model.applyScaledTransformation(2.0f, Sophus::SE3f(I3, Eigen::Vector3f::Zero()));
        auto xyz1 = cpuF(f.model.xyz_).contiguous();
        auto sig1 = cpuF(f.model.getScalingActivation()).contiguous();
        auto cov1 = cpuF(f.model.getCovarianceActivation()).contiguous();
        float ex = 0, es = 0, ec = 0;
        for (int i = 0; i < 4; ++i) {
            for (int d = 0; d < 3; ++d) {
                ex = std::max(ex, std::abs(2.f * xyz0[i][d].item<float>() - xyz1[i][d].item<float>()));
                es = std::max(es, std::abs(2.f * sig0[i][d].item<float>() - sig1[i][d].item<float>()));
            }
            for (int a = 0; a < 6; ++a)
                ec = std::max(ec, std::abs(4.f * cov0[i][a].item<float>() - cov1[i][a].item<float>()));
        }
        char buf[200];
        std::snprintf(buf, sizeof(buf), "xyz=%.2e sigma=%.2e cov=%.2e", ex, es, ec);
        check(ex < 1e-5f && es < 1e-5f && ec < 1e-4f,
              "O4 global s=2: centres x2, sigma x2, covariance x4", buf);
    }
    else if (c == 5) {   // loop path s=0.5 with a mixed selection
        Fixture f(true);
        Eigen::Matrix3f R(Eigen::AngleAxisf(0.45f, Eigen::Vector3f(0.f, 1.f, 0.f)));
        Eigen::Vector3f t(0.2f, 0.3f, -0.1f);
        const float s = 0.5f;
        torch::Tensor fl = f.allFlags();
        fl.index_put_({1}, false);
        const torch::Tensor selection = fl.clone();
        auto xyz0 = cpuF(f.model.xyz_).clone();
        auto sig0 = cpuF(f.model.getScalingActivation()).clone();
        auto rot0 = cpuF(f.model.rotation_).clone();
        int n = 0;
        f.loopTransform(fl, R, t, s, n);
        auto xyz1 = cpuF(f.model.xyz_).contiguous();
        auto sig1 = cpuF(f.model.getScalingActivation()).contiguous();
        float ex = 0, es = 0, er = 0;
        for (int i = 0; i < 4; ++i) {
            const bool sel = selection.to(torch::kCPU)[i].item<bool>();
            Eigen::Vector3f P(xyz0[i][0].item<float>(), xyz0[i][1].item<float>(), xyz0[i][2].item<float>());
            Eigen::Vector3f Pw = s * (R * P) + t;
            for (int d = 0; d < 3; ++d) {
                float want_x = sel ? Pw[d] : P[d];
                float want_s = sel ? s * sig0[i][d].item<float>() : sig0[i][d].item<float>();
                ex = std::max(ex, std::abs(want_x - xyz1[i][d].item<float>()));
                es = std::max(es, std::abs(want_s - sig1[i][d].item<float>()));
            }
            Eigen::Matrix3f Rq0 = quatToMat(rot0[i]);
            Eigen::Matrix3f want = sel ? Eigen::Matrix3f(R * Rq0) : Rq0;
            er = std::max(er, (want - quatToMat(cpuF(f.model.rotation_)[i])).cwiseAbs().maxCoeff());
        }
        char buf[200];
        std::snprintf(buf, sizeof(buf), "xyz=%.2e sigma=%.2e rot=%.2e", ex, es, er);
        check(ex < 1e-5f && es < 1e-5f && er < 1e-4f,
              "O5 loop s=0.5, mixed selection: selected follow contract, others untouched", buf);
    }
    else if (c == 6) {   // transform + inverse
        Fixture f(true);
        Eigen::Matrix3f R(Eigen::AngleAxisf(0.6f, Eigen::Vector3f(-0.3f, 1.f, 0.4f).normalized()));
        Eigen::Vector3f t(0.5f, -0.25f, 1.f);
        auto xyz0 = f.model.xyz_.clone();
        auto sig0 = cpuF(f.model.getScalingActivation()).clone();
        auto rot0 = f.model.rotation_.clone();
        torch::Tensor fl = f.allFlags(); int n = 0;
        f.loopTransform(fl, R, t, 1.0f, n);
        torch::Tensor fl2 = f.allFlags();
        f.loopTransform(fl2, R.transpose(), -(R.transpose() * t), 1.0f, n);
        float ex = (cpuF(f.model.xyz_) - cpuF(xyz0)).abs().max().item<float>();
        float es = (cpuF(f.model.getScalingActivation()) - sig0).abs().max().item<float>();
        float er = (cpuF(f.model.rotation_) - cpuF(rot0)).abs().max().item<float>();
        char buf[160];
        std::snprintf(buf, sizeof(buf), "xyz=%.2e sigma=%.2e rot=%.2e", ex, es, er);
        check(ex < 1e-4f && es < 1e-4f && er < 1e-4f, "O6 transform + inverse recovers state", buf);
    }
    else if (c == 7) {   // Adam state not created yet (no optimizer.step() ever)
        Fixture f(false);                       // note: no dummy step
        bool ok = false; std::string what;
        try {
            f.model.applyScaledTransformation(2.0f, Sophus::SE3f(I3, Eigen::Vector3f::Zero()));
            ok = true;
        } catch (const std::exception& e) { what = e.what(); }
        bool finite = cpuF(f.model.xyz_).isfinite().all().item<bool>()
                   && cpuF(f.model.scaling_).isfinite().all().item<bool>();
        check(ok && finite && f.refsOk(),
              "O7 update before the first optimizer.step() (null Adam state)", what);
    }
    else if (c == 8) {   // invalid scale must be refused, and nothing may change
        Fixture f(true);
        auto xyz0 = f.model.xyz_.clone();
        auto sig0 = f.model.scaling_.clone();
        auto rot0 = f.model.rotation_.clone();
        bool threw = false;
        try { f.model.applyScaledTransformation(std::nanf(""),
                Sophus::SE3f(I3, Eigen::Vector3f(0.3f, 0.2f, 0.1f))); }
        catch (const std::exception&) { threw = true; }
        float ex = (cpuF(f.model.xyz_) - cpuF(xyz0)).abs().max().item<float>();
        float es = (cpuF(f.model.scaling_) - cpuF(sig0)).abs().max().item<float>();
        float er = (cpuF(f.model.rotation_) - cpuF(rot0)).abs().max().item<float>();
        char buf[200];
        std::snprintf(buf, sizeof(buf), "threw=%d dxyz=%.2e dsigma=%.2e drot=%.2e",
                      (int)threw, ex, es, er);
        check(threw && ex == 0.f && es == 0.f && er == 0.f,
              "O8 invalid scale refused, state untouched", buf);
    }
    else {
        std::printf("usage: --case=1..8\n");
        return 97;
    }
    std::printf("case %d: %s\n", c, g_fail == 0 ? "OK" : "FAILED");
    return g_fail;
}
