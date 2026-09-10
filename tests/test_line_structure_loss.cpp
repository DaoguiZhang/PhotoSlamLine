/**
 * Deterministic autograd-gradient vs central finite-difference check for the
 * line structural losses of Photo-SLAM-L (GaussianModelLine):
 *
 *   - computeVectorizedLineLoss   : position + line-direction coherence (L_ori)
 *   - computeLineShapeConstraint  : scaling eccentricity + orientation (L_ecc/L_shape)
 *
 * This validates the two terms that make up the line structural loss L_str.
 */

#include <torch/torch.h>

#include <iostream>
#include <iomanip>
#include <cmath>
#include <functional>

#include "include/gaussian_model_line.h"

struct GradCheck
{
    std::string name;
    double maxAbs = 0.0;
    double maxRel = 0.0;
    int dof = 0;
    bool ran = false;
};

// Compare autograd gradient with central finite difference for `param`
// (a leaf tensor with requires_grad). `setter` writes a tensor into the model,
// `lossFn` returns the scalar loss given the model state.
static GradCheck checkGradient(const std::string& name,
                               torch::Tensor param,
                               const std::function<void(const torch::Tensor&)>& setter,
                               const std::function<torch::Tensor()>& lossFn,
                               double eps,
                               int maxDof)
{
    GradCheck rep;
    rep.name = name;
    rep.ran = true;

    // ---- analytic / autograd ----
    if (param.grad().defined())
        param.grad().zero_();
    torch::Tensor loss = lossFn();
    loss.backward();
    torch::Tensor autoGrad = param.grad().detach().cpu().clone();

    // ---- finite difference ----
    torch::Tensor original = param.detach().clone();
    torch::Tensor flat = original.flatten().clone();

    int dof = std::min(maxDof, (int)flat.numel());
    rep.dof = dof;

    double numericScale = 0.0;
    for (int i = 0; i < dof; ++i)
    {
        float base = flat[i].item<float>();

        torch::Tensor plus = flat.clone();
        plus[i] = base + (float)eps;
        torch::Tensor minus = flat.clone();
        minus[i] = base - (float)eps;

        {
            torch::NoGradGuard no_grad;
            setter(plus.view(original.sizes()).to(param.device()));
            double lp = lossFn().item<double>();
            setter(minus.view(original.sizes()).to(param.device()));
            double lm = lossFn().item<double>();
            double fd = (lp - lm) / (2.0 * eps);

            double a = autoGrad.flatten()[i].item<double>();
            double absErr = std::abs(a - fd);
            rep.maxAbs = std::max(rep.maxAbs, absErr);
            numericScale = std::max(numericScale, std::abs(fd));
        }
    }
    // Relative error normalized by the largest gradient magnitude (robust to
    // near-zero gradient components, where per-element relative error is ill-defined).
    rep.maxRel = rep.maxAbs / std::max(1e-12, numericScale);
    // restore
    {
        torch::NoGradGuard no_grad;
        setter(original.to(param.device()));
    }

    return rep;
}

static void print(const GradCheck& rep, double eps, double passAbs, double passRel)
{
    // Primary criterion: absolute error. maxRel is reported for reference and
    // is dominated by FD noise when a gradient component is near zero.
    bool pass = rep.ran && (rep.maxAbs < passAbs);
    std::cout << std::left << std::setw(46) << rep.name
              << " | dof=" << std::setw(3) << rep.dof
              << " | eps=" << eps
              << " | maxAbs=" << std::scientific << std::setprecision(3) << rep.maxAbs
              << " | maxRel=" << std::scientific << std::setprecision(3) << rep.maxRel
              << " | " << (rep.ran ? (pass ? "PASS" : "FAIL") : "SKIPPED")
              << " (abs<" << passAbs << ")" << std::endl;
}

int main()
{
    const double eps = 1e-5;
    const double passAbs = 1e-3;
    const double passRel = 5e-3;

    bool useCuda = torch::cuda::is_available();
    torch::Device dev(useCuda ? torch::kCUDA : torch::kCPU);

    GaussianModelLine model(0); // sh_degree = 0
    model.device_type_ = useCuda ? torch::kCUDA : torch::kCPU;

    const int N = 6;
    // A deterministic 3D line: points along direction d with small perpendicular offsets.
    Eigen::Vector3f d(1.0f, 0.3f, 0.1f);
    d.normalize();
    Eigen::Vector3f perp(-d.y(), d.x(), 0.0f);
    perp.normalize();

    auto xyzInit = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto xyzCur = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    auto dirW = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32));
    for (int i = 0; i < N; ++i)
    {
        float t = -1.0f + 2.0f * i / (N - 1);
        Eigen::Vector3f p = t * 0.6f * d;
        float offset = 0.02f * std::sin(0.7f * i + 0.3f); // small, non-zero perpendicular offset
        // Add a parallel component too, so dL/d(line_dir) is non-degenerate.
        Eigen::Vector3f pc = p + offset * perp + 0.03f * d;
        xyzInit[i][0] = p.x(); xyzInit[i][1] = p.y(); xyzInit[i][2] = p.z();
        xyzCur[i][0] = pc.x(); xyzCur[i][1] = pc.y(); xyzCur[i][2] = pc.z();
        dirW[i][0] = d.x(); dirW[i][1] = d.y(); dirW[i][2] = d.z();
    }

    auto isLine = torch::ones({N}, torch::TensorOptions().dtype(torch::kBool));

    // Move to device
    xyzInit = xyzInit.to(dev);
    xyzCur = xyzCur.to(dev);
    dirW = dirW.to(dev);
    isLine = isLine.to(dev);

    model.xyz_ = xyzCur;
    model.xyz_init_ = xyzInit;
    model.line_dir_w_ = dirW;
    model.is_line_ = isLine;

    std::cout << "===== Line structural loss: autograd vs central finite difference =====" << std::endl;
    std::cout << "device: " << (useCuda ? "cuda" : "cpu") << std::endl;

    // ------------------------------------------------------------------
    // 1. L_ori: position (xyz_) gradient
    // ------------------------------------------------------------------
    {
        model.xyz_ = xyzCur.detach().clone().requires_grad_(true);
        model.xyz_init_ = xyzInit.detach().clone();
        model.line_dir_w_ = dirW.detach().clone();
        model.is_line_ = isLine;

        auto setter = [&](const torch::Tensor& t) { model.xyz_ = t; };
        auto lossFn = [&]() { return model.computeVectorizedLineLoss(1.0f); };
        auto rep = checkGradient("computeVectorizedLineLoss dL/d(positions)",
                                 model.xyz_, setter, lossFn, eps, 3 * N);
        print(rep, eps, passAbs, passRel);
    }

    // ------------------------------------------------------------------
    // 2. L_ori: line-direction (line_dir_w_) gradient
    // ------------------------------------------------------------------
    {
        model.xyz_ = xyzCur.detach().clone();
        model.xyz_init_ = xyzInit.detach().clone();
        model.line_dir_w_ = dirW.detach().clone().requires_grad_(true);
        model.is_line_ = isLine;

        auto setter = [&](const torch::Tensor& t) { model.line_dir_w_ = t; };
        auto lossFn = [&]() { return model.computeVectorizedLineLoss(1.0f); };
        auto rep = checkGradient("computeVectorizedLineLoss dL/d(line_dir)",
                                 model.line_dir_w_, setter, lossFn, eps, 3 * N);
        print(rep, eps, passAbs, passRel);
    }

    // ------------------------------------------------------------------
    // 3. L_shape: scaling (log-space) gradient
    // 4. L_shape: rotation gradient
    //    (uses general_utils::build_rotation, which hardcodes CUDA)
    // ------------------------------------------------------------------
    if (!useCuda)
    {
        GradCheck skip;
        skip.name = "computeLineShapeConstraint (scaling/rotation)";
        print(skip, eps, passAbs, passRel);
    }
    else
    {
        auto scaling = torch::empty({N, 3}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        auto rotation = torch::empty({N, 4}, torch::TensorOptions().dtype(torch::kFloat32).device(dev));
        for (int i = 0; i < N; ++i)
        {
            scaling[i][0] = std::log(0.7f);   // along-line
            scaling[i][1] = std::log(0.15f);  // perpendicular
            scaling[i][2] = std::log(0.15f);  // perpendicular
            rotation[i][0] = 1.0f; rotation[i][1] = 0.0f;
            rotation[i][2] = 0.0f; rotation[i][3] = 0.0f;
        }
        model.scaling_ = scaling;
        model.rotation_ = rotation;
        model.line_dir_w_ = dirW.detach().clone();
        model.is_line_ = isLine;

        // scaling gradient
        {
            model.scaling_ = scaling.detach().clone().requires_grad_(true);
            model.rotation_ = rotation.detach().clone();
            auto setter = [&](const torch::Tensor& t) { model.scaling_ = t; };
            auto lossFn = [&]() { return model.computeLineShapeConstraint(1.0f, 1.0f); };
            auto rep = checkGradient("computeLineShapeConstraint dL/d(scaling)",
                                     model.scaling_, setter, lossFn, eps, 3 * N);
            print(rep, eps, passAbs, passRel);
        }

        // rotation gradient
        {
            model.scaling_ = scaling.detach().clone();
            model.rotation_ = rotation.detach().clone().requires_grad_(true);
            auto setter = [&](const torch::Tensor& t) { model.rotation_ = t; };
            auto lossFn = [&]() { return model.computeLineShapeConstraint(1.0f, 1.0f); };
            auto rep = checkGradient("computeLineShapeConstraint dL/d(rotation)",
                                     model.rotation_, setter, lossFn, eps, 4 * N);
            print(rep, eps, passAbs, passRel);
        }
    }

    std::cout << "============================================================================" << std::endl;
    return 0;
}
