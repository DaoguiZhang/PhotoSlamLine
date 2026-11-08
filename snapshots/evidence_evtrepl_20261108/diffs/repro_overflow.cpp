// Minimal reproducer for the intermittent
//   std::runtime_error: value cannot be converted to type float without overflow
// seen once in a while on the official test cases.
//
// It does NOT touch any geometry: it only builds a GaussianModel, calls
// trainingSetup() and performs the first optimizer step on zero gradients --
// exactly what the test fixture does before the first assertion. If the abort
// reproduces here, the exception cannot be attributed to the code under test.
//
// usage: repro_overflow [n_steps]
#include <torch/torch.h>

#include <cstdio>
#include <map>
#include <string>

#include "include/gaussian_model.h"
#include "include/gaussian_parameters.h"
#include "include/point3d.h"

int main(int argc, char **argv)
{
    torch::manual_seed(5);
    int n_steps = 1;
    if (argc > 1)
        n_steps = std::atoi(argv[1]);
    if (!torch::cuda::is_available()) {
        std::printf("CUDA required\n");
        return 99;
    }
    {
        // same library identity report as review_official.cpp
        std::ifstream maps("/proc/self/maps");
        std::string line, best;
        while (std::getline(maps, line))
            if (line.find("libgaussian_mapper.so") != std::string::npos) {
                const auto sp = line.find('/');
                if (sp != std::string::npos)
                    best = line.substr(sp);
            }
        std::printf("[lib] mapped=%s\n", best.c_str());
    }

    GaussianModel model(
        GaussianModelParams("", "", "", 3, "images", -1.0f, false, "cuda", false));
    std::map<point3D_id_t, Point3D> pcd;
    for (int i = 0; i < 4; ++i) {
        Point3D p;
        p.xyz_ = Eigen::Vector3d(1.0 + i, 2.0 - i, 3.0 + 0.5 * i);
        p.color_ = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
        pcd[static_cast<point3D_id_t>(i)] = p;
    }
    model.createFromPcd(pcd, 1.0f);

    GaussianOptimizationParams opt;
    opt.iterations_ = 1000;
    opt.densify_from_iter_ = 100;
    opt.densify_until_iter_ = 500;
    // upstream self-initialises opacity_lr_ (opacity_lr_(opacity_lr_)); optional
    // but the honest thing is to give the optimiser a value instead of garbage.
    opt.opacity_lr_ = 0.05f;
    model.trainingSetup(opt);

    {
        int gi = 0;
        for (auto &g : model.optimizer_->param_groups()) {
            auto *a = dynamic_cast<torch::optim::AdamOptions *>(&g.options());
            std::printf("[opt] group %d nparams=%zu lr=%.17g eps=%.17g "
                        "beta1=%.17g beta2=%.17g weight_decay=%.17g\n",
                        gi++, g.params().size(),
                        a ? a->lr() : -1.0, a ? a->eps() : -1.0,
                        a ? std::get<0>(a->betas()) : -1.0,
                        a ? std::get<1>(a->betas()) : -1.0,
                        a ? a->weight_decay() : -1.0);
        }
    }

    torch::NoGradGuard ng;
    for (auto &g : model.optimizer_->param_groups())
        for (auto &p : g.params())
            p.mutable_grad() = torch::zeros_like(p);
    for (int i = 0; i < n_steps; ++i) {
        model.optimizer_->step();
        model.optimizer_->zero_grad(true);
    }
    std::printf("steps=%d ok xyz0=%.6f\n", n_steps,
                model.xyz_.select(0, 0).select(0, 0).item<float>());
    return 0;
}
