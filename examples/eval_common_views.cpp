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
    // Replica ORB settings put intrinsics under Camera1.* and often omit w/h.
    auto pick = [&](const char* k1, const char* k2, double dflt) -> double {
        if (!fs[k1].empty()) return (double)fs[k1];
        if (!fs[k2].empty()) return (double)fs[k2];
        std::cerr << "[EvalViews] WARN missing " << k1 << "/" << k2
                  << " -> using " << dflt << "\n";
        return dflt;
    };
    const int W = (int)pick("Camera.w", "Camera1.w", 1200.0);
    const int H = (int)pick("Camera.h", "Camera1.h", 680.0);
    const float fx = (float)pick("Camera1.fx", "Camera.fx", 600.0);
    const float fy = (float)pick("Camera1.fy", "Camera.fy", 600.0);
    const float cx = (float)pick("Camera1.cx", "Camera.cx", 599.5);
    const float cy = (float)pick("Camera1.cy", "Camera.cy", 339.5);
    std::cout << "[EvalViews] W=" << W << " H=" << H << " fx=" << fx << " fy=" << fy
              << " cx=" << cx << " cy=" << cy << "\n";

    // Load the map with the line-capable model (reads point PLYs too).
    GaussianModelParamsLine mp{std::filesystem::path(argv[1])};
    auto model = std::make_shared<GaussianModelLine>(mp);
    model->device_type_ = device;
    model->loadPly(std::filesystem::path(argv[2]));
    // Render-time buffers are not part of the PLY (they are training state).
    // The renderer reads max_radii2D_ when building the visibility filter, so
    // an undefined buffer produces a 0-element CUDA kernel launch.
    {
        const auto N = model->xyz_.size(0);
        const auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
        if (!model->max_radii2D_.defined() || model->max_radii2D_.numel() != N)
            model->max_radii2D_ = torch::zeros({N, 1}, opt);
        if (!model->xyz_gradient_accum_.defined() || model->xyz_gradient_accum_.numel() != N)
            model->xyz_gradient_accum_ = torch::zeros({N, 1}, opt);
        if (!model->denom_.defined() || model->denom_.numel() != N)
            model->denom_ = torch::zeros({N, 1}, opt);
        std::cout << "[EvalViews] buffers max_radii2D=" << model->max_radii2D_.sizes()
                  << " device=" << model->max_radii2D_.device() << "\n";
    }
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

    // Same Camera construction as GaussianMapperLine::loadPly (online path).
    Camera camera;
    camera.camera_id_ = 0;
    camera.width_ = W;
    camera.height_ = H;
    camera.setModelId(Camera::CameraModelType::PINHOLE);
    camera.params_[0] = fx; camera.params_[1] = fy;
    camera.params_[2] = cx; camera.params_[3] = cy;
    camera.dist_coeff_ = cv::Mat::zeros(5, 1, CV_32F);
    {
        cv::Mat K = (cv::Mat_<float>(3, 3) << fx, 0.f, cx, 0.f, fy, cy, 0.f, 0.f, 1.f);
        camera.initUndistortRectifyMapAndMask(K, cv::Size(W, H), K, false);
    }
    std::cout << "[EvalViews] camera " << camera.width_ << "x" << camera.height_
              << " params=" << camera.params_[0] << "," << camera.params_[1] << ","
              << camera.params_[2] << "," << camera.params_[3]
              << " model=" << int(camera.model_id_) << "\n";

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
        // Mirror the online construction: z-clip planes must be set BEFORE
        // computeTransformTensors(), otherwise the projection is degenerate.
        kf->znear_ = 0.01f;
        kf->zfar_ = 100.0f;
        kf->setCameraParams(camera);   // same as the online keyframe path
        kf->setPose(qw, qx, qy, qz, tx, ty, tz);
        kf->computeTransformTensors();
        std::cout << "[EvalViews] fid=" << fid << " W=" << kf->image_width_
                  << " H=" << kf->image_height_ << " FoVx=" << kf->FoVx_
                  << " FoVy=" << kf->FoVy_ << " znear=" << kf->znear_
                  << " zfar=" << kf->zfar_ << "\n"
                  << "[EvalViews] world_view=" << kf->world_view_transform_.sizes()
                  << kf->world_view_transform_.device()
                  << " full_proj=" << kf->full_proj_transform_.sizes()
                  << kf->full_proj_transform_.device()
                  << " xyz=" << model->xyz_.sizes() << model->xyz_.device()
                  << " rot=" << model->rotation_.sizes() << model->rotation_.device()
                  << " sh=" << model->features_rest_.sizes()
                  << " active_sh=" << model->active_sh_degree_ << "\n";

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
