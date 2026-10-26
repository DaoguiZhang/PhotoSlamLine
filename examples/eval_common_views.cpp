// Headless evaluation of a saved Gaussian map at a fixed camera list.
// No ImGui, no training, no optimizer state: loads a PLY, renders the given
// views with the existing renderer/metrics-free path, and writes PNGs plus a
// per-view CSV. Metrics are computed offline from the PNGs by one shared
// Python implementation (scripts/eval_common_views.py).
//
// usage: eval_common_views <gaussian_cfg> <ply> <orb_cfg> <cameras.txt> <out_dir>
// cameras.txt lines: <fid> <qw> <qx> <qy> <qz> <tx> <ty> <tz>   (Tcw)
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "include/gaussian_model_line.h"
#include "include/gaussian_keyframe_line.h"
#include "include/gaussian_renderer_line.h"
#include "include/gaussian_parameters_line.h"
#include "include/camera.h"

int main(int argc, char** argv)
{
    if (argc != 6)
    {
        std::cerr << "usage: " << argv[0]
                  << " <gaussian_cfg> <ply> <orb_cfg> <cameras.txt> <out_dir>\n";
        return 2;
    }
    torch::DeviceType device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    // Camera intrinsics / size from the ORB settings (same file as training).
    cv::FileStorage fs(argv[3], cv::FileStorage::READ);
    if (!fs.isOpened()) { std::cerr << "cannot open " << argv[3] << "\n"; return 2; }
    const int W = fs["Camera.w"], H = fs["Camera.h"];
    const float fx = fs["Camera.fx"], fy = fs["Camera.fy"];
    const float cx = fs["Camera.cx"], cy = fs["Camera.cy"];
    std::cout << "[EvalViews] W=" << W << " H=" << H << " fx=" << fx << " fy=" << fy
              << " cx=" << cx << " cy=" << cy << "\n";

    // Load the map with the line-capable model (reads point PLYs too).
    GaussianModelParamsLine mp{std::filesystem::path(argv[1])};
    auto model = std::make_shared<GaussianModelLine>(mp);
    model->device_type_ = device;
    model->loadPly(std::filesystem::path(argv[2]));
    std::cout << "[EvalViews] loaded gaussians=" << model->xyz_.size(0)
              << " line=" << (model->is_line_.defined() && model->is_line_.numel()
                                      ? model->is_line_.to(torch::kCPU).sum().item<int64_t>()
                                      : 0)
              << " active_sh=" << model->active_sh_degree_ << "\n";

    GaussianPipelineParamsLine pipe(true, true);
    torch::Tensor bg = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    torch::Tensor empty_color;
    std::filesystem::path out(argv[5]);
    std::filesystem::create_directories(out / "image");
    std::ofstream csv(out / "render_meta.csv");
    csv << "fid,image\n";

    std::ifstream cam(argv[4]);
    std::string line;
    const float fovx = 2.0f * std::atan(W / (2.0f * fx));
    const float fovy = 2.0f * std::atan(H / (2.0f * fy));
    while (std::getline(cam, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::size_t fid; double qw, qx, qy, qz, tx, ty, tz;
        ss >> fid >> qw >> qx >> qy >> qz >> tx >> ty >> tz;
        auto kf = std::make_shared<GaussianKeyframeLine>(fid, 0);
        kf->image_width_ = W;
        kf->image_height_ = H;
        kf->camera_id_ = 0;
        kf->FoVx_ = fovx;
        kf->FoVy_ = fovy;
        kf->set_camera_ = true;
        kf->setPose(qw, qx, qy, qz, tx, ty, tz);
        kf->computeTransformTensors();

        auto pkg = GaussianRendererWithLine::renderWithLine(
            kf, H, W, model, pipe, bg, empty_color);
        torch::Tensor img = std::get<0>(pkg).detach().clamp(0.0, 1.0).to(torch::kCPU)
                                .permute({1, 2, 0}).contiguous();
        cv::Mat rgb(H, W, CV_32FC3, img.data_ptr<float>());
        cv::Mat rgb8, bgr;
        rgb.convertTo(rgb8, CV_8UC3, 255.0);
        cv::cvtColor(rgb8, bgr, cv::COLOR_RGB2BGR);
        char name[32]; std::snprintf(name, sizeof(name), "%06zu.png", fid);
        cv::imwrite((out / "image" / name).string(), bgr);
        csv << fid << "," << name << "\n";
    }
    std::cout << "[EvalViews] done -> " << out << "\n";
    return 0;
}
