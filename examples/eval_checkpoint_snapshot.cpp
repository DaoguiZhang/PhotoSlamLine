// Offline re-render of a diagnostic online checkpoint.
//
// Loads the exact model tensors saved by GaussianModelLine::saveFullSnapshot,
// the exact training cameras and undistort masks captured with the checkpoint,
// and recomputes the native C++ GS-PSNR.  No new GT camera is created here.
//
// usage:
//   eval_checkpoint_snapshot <gaussian_cfg> <snapshot.pt> <meta.txt>
//     <cameras.csv> <masks.pt> <mask_cameras.txt> <orb_cfg> <out_dir>
//
// The checkpoint-time psnr_gaussian_splatting.txt is only compared afterwards;
// it is not an input to this program.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <torch/torch.h>
#include <opencv2/opencv.hpp>

#include "include/camera.h"
#include "include/gaussian_keyframe_line.h"
#include "include/gaussian_model_line.h"
#include "include/gaussian_parameters_line.h"
#include "include/gaussian_renderer_line.h"
#include "include/loss_utils.h"

namespace {

std::map<std::string, std::string> readMeta(const std::filesystem::path& p)
{
    std::map<std::string, std::string> m;
    std::ifstream in(p);
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) continue;
        m[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return m;
}

struct CamRow {
    std::size_t fid = 0;
    std::string image;
    double qw = 1, qx = 0, qy = 0, qz = 0;
    double tx = 0, ty = 0, tz = 0;
    float fx = 0, fy = 0, cx = 0, cy = 0;
    int width = 0, height = 0;
};

std::vector<CamRow> readCameras(const std::filesystem::path& p)
{
    std::vector<CamRow> rows;
    std::ifstream in(p);
    std::string line;
    std::getline(in, line); // header
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string f;
        std::vector<std::string> v;
        while (std::getline(ss, f, ',')) v.push_back(f);
        if (v.size() < 15) continue;
        CamRow r;
        r.fid = std::stoull(v[0]);
        r.image = v[1];
        r.qw = std::stod(v[2]);
        r.qx = std::stod(v[3]);
        r.qy = std::stod(v[4]);
        r.qz = std::stod(v[5]);
        r.tx = std::stod(v[6]);
        r.ty = std::stod(v[7]);
        r.tz = std::stod(v[8]);
        r.fx = std::stof(v[9]);
        r.fy = std::stof(v[10]);
        r.cx = std::stof(v[11]);
        r.cy = std::stof(v[12]);
        r.width = std::stoi(v[13]);
        r.height = std::stoi(v[14]);
        rows.push_back(std::move(r));
    }
    return rows;
}

std::vector<int> readMaskIds(const std::filesystem::path& p)
{
    std::vector<int> ids;
    std::ifstream in(p);
    int id;
    while (in >> id) ids.push_back(id);
    return ids;
}

std::string valueOr(const std::map<std::string, std::string>& m,
                    const std::string& k, const std::string& d)
{
    auto it = m.find(k);
    return it == m.end() ? d : it->second;
}

std::vector<long long> readZeroIndices(const char* path)
{
    std::vector<long long> out;
    if (!path || !path[0]) return out;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        std::string first;
        std::getline(ss, first, ',');
        try {
            out.push_back(std::stoll(first));
        } catch (...) {
            // header or malformed line
        }
    }
    return out;
}

void saveDiffImage(const torch::Tensor& diff, const std::filesystem::path& path)
{
    auto t = diff.detach().to(torch::kCPU).clamp(0.0, 1.0).mul(255.0)
                 .to(torch::kUInt8).contiguous();
    if (t.dim() != 3 || t.size(0) != 3) return;
    const int H = static_cast<int>(t.size(1));
    const int W = static_cast<int>(t.size(2));
    std::vector<cv::Mat> ch(3);
    for (int c = 0; c < 3; ++c)
        ch[c] = cv::Mat(H, W, CV_8UC1, t[c].data_ptr<uint8_t>());
    cv::Mat rgb, bgr;
    cv::merge(ch, rgb);
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    cv::imwrite(path.string(), bgr);
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 9) {
        std::cerr << "usage: " << argv[0]
                  << " <gaussian_cfg> <snapshot.pt> <meta.txt> <cameras.csv>"
                     " <masks.pt> <mask_cameras.txt> <orb_cfg> <out_dir>\n";
        return 2;
    }
    const torch::DeviceType device =
        torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::filesystem::path out(argv[8]);
    std::filesystem::create_directories(out);

    // Read the same model parameters that the mapper used.
    cv::FileStorage gcfg(argv[1], cv::FileStorage::READ);
    if (!gcfg.isOpened()) {
        std::cerr << "cannot open " << argv[1] << "\n";
        return 2;
    }
    GaussianModelParamsLine mp;
    mp.sh_degree_ = static_cast<int>(gcfg["Model.sh_degree"]);
    mp.white_background_ =
        static_cast<int>(gcfg["Model.white_background"]) != 0;
    mp.resolution_ = static_cast<float>(gcfg["Model.resolution"]);
    mp.data_device_ = device == torch::kCUDA ? "cuda" : "cpu";

    auto model = std::make_shared<GaussianModelLine>(mp);
    model->device_type_ = device;
    if (model->loadFullSnapshot(argv[2]) == 0) {
        std::cerr << "[OfflineCkpt] failed to load snapshot " << argv[2] << "\n";
        return 3;
    }
    const auto meta = readMeta(argv[3]);
    const int active_sh = std::stoi(valueOr(meta, "active_sh", "3"));
    model->active_sh_degree_ = active_sh;

    long long line_n = 0;
    if (model->is_line_.defined() && model->is_line_.numel() > 0)
        line_n = model->is_line_.to(torch::kCPU).sum().item<int64_t>();
    std::cerr << "[OfflineCkpt] model=" << argv[2]
              << " n=" << model->xyz_.size(0) << " line=" << line_n
              << " active_sh=" << model->active_sh_degree_
              << " max_sh=" << model->max_sh_degree_ << "\n";

    const auto cams = readCameras(argv[4]);
    const auto mask_ids = readMaskIds(argv[6]);
    std::vector<torch::Tensor> masks;
    try {
        torch::load(masks, argv[5]);
    } catch (const std::exception& e) {
        std::cerr << "[OfflineCkpt] mask load failed: " << e.what() << "\n";
        return 4;
    }
    if (masks.empty() || mask_ids.empty()) {
        std::cerr << "[OfflineCkpt] no masks\n";
        return 4;
    }

    cv::FileStorage ocfg(argv[7], cv::FileStorage::READ);
    if (!ocfg.isOpened()) {
        std::cerr << "cannot open " << argv[7] << "\n";
        return 2;
    }
    const float old_fx = static_cast<float>(ocfg["Camera1.fx"]);
    const float old_fy = static_cast<float>(ocfg["Camera1.fy"]);
    const float old_cx = static_cast<float>(ocfg["Camera1.cx"]);
    const float old_cy = static_cast<float>(ocfg["Camera1.cy"]);
    const float k1 = ocfg["Camera1.k1"].empty()
                         ? 0.0f : static_cast<float>(ocfg["Camera1.k1"]);
    const float k2 = ocfg["Camera1.k2"].empty()
                         ? 0.0f : static_cast<float>(ocfg["Camera1.k2"]);
    const float p1 = ocfg["Camera1.p1"].empty()
                         ? 0.0f : static_cast<float>(ocfg["Camera1.p1"]);
    const float p2 = ocfg["Camera1.p2"].empty()
                         ? 0.0f : static_cast<float>(ocfg["Camera1.p2"]);

    // The checkpoint stores undistorted intrinsics.  Use the same old K and
    // distortion coefficients as the mapper to produce the GT image.
    auto makeCamera = [&](const CamRow& r, int camera_id) {
        Camera c(static_cast<camera_id_t>(camera_id), r.width, r.height,
                 {r.fx, r.fy, r.cx, r.cy}, Camera::CameraModelType::PINHOLE,
                 true);
        c.dist_coeff_ = (cv::Mat_<float>(1, 4) << k1, k2, p1, p2);
        cv::Mat K_old = (cv::Mat_<float>(3, 3) << old_fx, 0.f, old_cx,
                         0.f, old_fy, old_cy, 0.f, 0.f, 1.f);
        cv::Mat K_new = (cv::Mat_<float>(3, 3) << r.fx, 0.f, r.cx,
                         0.f, r.fy, r.cy, 0.f, 0.f, 1.f);
        c.initUndistortRectifyMapAndMask(K_old, cv::Size(r.width, r.height),
                                         K_new, false);
        return c;
    };

    GaussianPipelineParamsLine pipe(true, true);
    torch::Tensor bg = mp.white_background_
                           ? torch::ones({3}, torch::TensorOptions()
                                                  .dtype(torch::kFloat32)
                                                  .device(device))
                           : torch::zeros({3}, torch::TensorOptions()
                                                   .dtype(torch::kFloat32)
                                                   .device(device));
    torch::Tensor empty_color = torch::empty(
        0, torch::TensorOptions().device(device));

    const auto mask_id = mask_ids.front();
    torch::Tensor mask = masks.front().to(device).to(torch::kFloat32);
    // The saved mask is [3,H,W] for the RGB-D camera.
    if (mask.dim() == 2) mask = mask.unsqueeze(0);
    if (mask.size(0) == 1) mask = mask.repeat({3, 1, 1});

    const auto zero_indices = readZeroIndices(
        std::getenv("PHOTO_SLAM_ZERO_INDICES"));
    torch::Tensor zero_idx;
    if (!zero_indices.empty()) {
        std::vector<int64_t> zi(zero_indices.begin(), zero_indices.end());
        zero_idx = torch::from_blob(zi.data(),
                                    {static_cast<long>(zi.size())},
                                    torch::TensorOptions().dtype(torch::kLong))
                       .clone()
                       .to(device);
    }
    const char* diff_env = std::getenv("PHOTO_SLAM_DIFF_DIR");
    std::filesystem::path diff_dir;
    if (diff_env && diff_env[0]) {
        diff_dir = diff_env;
        std::filesystem::create_directories(diff_dir);
    }
    torch::Tensor saved_opacity = model->opacity_.detach().clone();
    std::cerr << "[OfflineCkpt] zero_opacity_rows=" << zero_indices.size()
              << " diff_dir=" << diff_dir << "\n";

    std::ofstream csv(out / "offline_psnr.csv");
    csv << "fid,psnr_gs,psnr,orig_psnr_gs,orig_psnr,width,height,"
           "render_seconds\n";
    double sum_gs = 0.0, sum_psnr = 0.0;
    double sum_orig_gs = 0.0, sum_orig_p = 0.0;
    std::size_t n = 0;
    for (const auto& r : cams) {
        cv::Mat bgr = cv::imread(r.image, cv::IMREAD_UNCHANGED);
        if (bgr.empty()) {
            std::cerr << "[OfflineCkpt] missing image " << r.image << "\n";
            continue;
        }
        cv::Mat rgb;
        if (bgr.channels() == 3)
            cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        else
            cv::cvtColor(bgr, rgb, cv::COLOR_GRAY2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);

        Camera cam = makeCamera(r, mask_id);
        cv::Mat gt_undist;
        cam.undistortImage(rgb, gt_undist);
        torch::Tensor gt =
            tensor_utils::cvMat2TorchTensor_Float32(gt_undist, device);

        auto kf = std::make_shared<GaussianKeyframeLine>(r.fid, 0);
        kf->setCameraParams(cam);
        kf->setPose(r.qw, r.qx, r.qy, r.qz, r.tx, r.ty, r.tz);
        kf->computeTransformTensors();

        const auto t0 = std::chrono::steady_clock::now();
        torch::NoGradGuard ng;
        // Original render first; the variant is rendered from the exact same
        // model/camera/SH/background/mask with only the selected opacity rows
        // zeroed (logit -100).
        model->opacity_.copy_(saved_opacity);
        auto pkg_orig = GaussianRendererWithLine::renderWithLine(
            kf, r.height, r.width, model, pipe, bg, empty_color);
        torch::Tensor orig = std::get<0>(pkg_orig) * mask;
        torch::Tensor rendered = orig;
        if (zero_idx.defined() && zero_idx.numel() > 0) {
            model->opacity_.index_put_({zero_idx}, -100.0f);
            auto pkg_var = GaussianRendererWithLine::renderWithLine(
                kf, r.height, r.width, model, pipe, bg, empty_color);
            rendered = std::get<0>(pkg_var) * mask;
            model->opacity_.copy_(saved_opacity);
            if (!diff_dir.empty()) {
                torch::Tensor diff = (0.5f + 2.0f * (orig - rendered))
                                         .clamp(0.0f, 1.0f);
                char name[32];
                std::snprintf(name, sizeof(name), "%06zu.png",
                              static_cast<std::size_t>(r.fid));
                saveDiffImage(diff, diff_dir / name);
            }
        }
        auto t1 = std::chrono::steady_clock::now();

        torch::Tensor gt_t = gt;
        float p_gs = loss_utils::psnr_gaussian_splatting(rendered, gt_t)
                         .item().toFloat();
        float p = loss_utils::psnr(rendered, gt_t).item().toFloat();
        torch::Tensor orig_t = orig;
        float orig_gs = loss_utils::psnr_gaussian_splatting(orig_t, gt_t)
                            .item().toFloat();
        float orig_p = loss_utils::psnr(orig_t, gt_t).item().toFloat();
        const double sec =
            std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0)
                .count() / 1e6;
        csv << r.fid << ',' << std::setprecision(10) << p_gs << ',' << p << ','
            << orig_gs << ',' << orig_p << ',' << r.width << ',' << r.height
            << ',' << sec << '\n';
        sum_gs += p_gs;
        sum_psnr += p;
        sum_orig_gs += orig_gs;
        sum_orig_p += orig_p;
        ++n;
    }
    csv.close();

    std::ofstream summary(out / "offline_summary.txt");
    summary << "snapshot=" << argv[2] << "\n"
            << "frames=" << n << "\n"
            << "mean_psnr_gs=" << std::setprecision(10)
            << (n ? sum_gs / n : 0.0) << "\n"
            << "mean_psnr=" << (n ? sum_psnr / n : 0.0) << "\n"
            << "original_mean_psnr_gs="
            << (n ? sum_orig_gs / n : 0.0) << "\n"
            << "original_mean_psnr=" << (n ? sum_orig_p / n : 0.0) << "\n"
            << "zero_opacity_rows=" << zero_indices.size() << "\n"
            << "engine=native_cpp_psnr_gaussian_splatting\n";
    std::cerr << "[OfflineCkpt] frames=" << n << " mean_gs="
              << (n ? sum_gs / n : 0.0) << " -> " << out << "\n";
    return n ? 0 : 5;
}
