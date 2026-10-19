#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cmath>

static void report(const char* name, const torch::Tensor& g_ref, const torch::Tensor& g_new)
{
    double d = (g_ref - g_new).abs().max().item<double>();
    double r = g_ref.abs().max().item<double>();
    std::cout << "  " << name << ": max|dgrad|=" << std::setprecision(6) << d
              << "  rel=" << (r > 0 ? d / r : d) << (d == 0.0 ? "   IDENTICAL" : "   *** DIFFERS ***")
              << std::endl;
}

// max(dim).values / min(dim).values  (with indices -> gradient to the single argmax/argmin)
static torch::Tensor aniso_ref(const torch::Tensor& sc, double th, double lam)
{
    auto mx = std::get<0>(torch::max(sc, 1));
    auto mn = std::get<0>(torch::min(sc, 1));
    auto ratio = mx / (mn + 1e-6);
    return lam * torch::pow(torch::clamp(ratio - th, 0), 2).mean();
}
// amax / amin  (values only -> gradient spread over all tied extrema)
static torch::Tensor aniso_new(const torch::Tensor& sc, double th, double lam)
{
    auto mx = torch::amax(sc, 1);
    auto mn = torch::amin(sc, 1);
    auto ratio = mx / (mn + 1e-6);
    return lam * torch::pow(torch::clamp(ratio - th, 0), 2).mean();
}

static double TH = 1.0;
static void oneCase(std::string name, torch::Tensor sc, torch::Device dev)
{
    std::cout << "[" << name << "] device=" << dev.str() << "  threshold=" << TH << std::endl;
    for (auto& mk : {std::make_pair("reference max/min(dim)", 0), std::make_pair("optimised amax/amin", 1)}) {
        sc.detach_().requires_grad_();
        if (sc.grad().defined()) sc.grad().zero_();
        auto l = (mk.second == 0) ? aniso_ref(sc, TH, 0.01) : aniso_new(sc, TH, 0.01);
        l.backward();
        static torch::Tensor g0, l0;
        if (mk.second == 0) { g0 = sc.grad().detach().clone(); l0 = l.detach().clone(); }
        else {
            std::cout << "  loss: ref=" << l0.item<double>() << " new=" << l.item<double>()
                      << "  max|dloss|=" << std::abs(l0.item<double>() - l.item<double>()) << std::endl;
            report("grad wrt scaling_", g0, sc.grad().detach());
        }
    }
    std::cout << std::endl;
}

int main()
{
    std::cout << "LibTorch " << TORCH_VERSION << "  CUDA available=" << torch::cuda::is_available() << "\n\n";
    for (torch::Device dev : {torch::Device(torch::kCPU), torch::Device(torch::kCUDA)}) {
        if (dev.is_cuda() && !torch::cuda::is_available()) continue;
        auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(dev);

        // 1. unique max and unique min
        oneCase("unique max / unique min", torch::tensor({{0.14f, 0.08f, 0.05f},
                                                          {0.30f, 0.01f, 0.02f}}, opt), dev);
        // 2. tied max
        oneCase("tied max (two axes equal to max)", torch::tensor({{0.20f, 0.20f, 0.01f},
                                                                   {0.50f, 0.50f, 0.02f}}, opt), dev);
        // 3. tied min
        oneCase("tied min (two axes equal to min)", torch::tensor({{0.30f, 0.08f, 0.08f},
                                                                   {0.60f, 0.03f, 0.03f}}, opt), dev);
        // 4. the REAL init pattern: s_para=0.14 unique max, s_perp1=s_perp2=0.08 exactly tied min
        {
            auto base = torch::tensor({{0.14f, 0.08f, 0.08f}}, opt).repeat({64, 1});
            oneCase("actual init pattern (tied perpendicular axes)", base, dev);
        }
        // 5. realistic trained pattern, mostly untied
        {
            auto r = torch::rand({200, 3}, opt) * 0.3f + 0.01f;
            oneCase("realistic random scales (no ties expected)", r, dev);
        }
    }
    return 0;
}
