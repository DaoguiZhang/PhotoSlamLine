/**
 * This file is part of Photo-SLAM
 *
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

#include "include/gaussian_scene_line.h"

GaussianSceneLine::GaussianSceneLine(
    GaussianModelParamsLine& args,
    int load_iteration,
    bool shuffle,
    std::vector<float> resolution_scales)
{
    if (load_iteration)
    {
        this->loaded_iter_ = load_iteration;
        std::cout << "Loading trained model at iteration " << load_iteration << std::endl;
    }
    global_sample_counter_ = 500000000;
}

void GaussianSceneLine::addCamera(Camera& camera)
{
    this->cameras_.emplace(camera.camera_id_, camera);
}

Camera& GaussianSceneLine::getCamera(camera_id_t cameraId)
{
    return this->cameras_[cameraId];
}

void GaussianSceneLine::addKeyframe(std::shared_ptr<GaussianKeyframeLine> new_kf, bool* shuffled)
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    this->keyframes_.emplace(new_kf->fid_, new_kf);
    *shuffled = false;
}

std::shared_ptr<GaussianKeyframeLine>
GaussianSceneLine::getKeyframe(std::size_t fid)
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    if (this->keyframes_.find(fid) != this->keyframes_.end())
        return this->keyframes_[fid];
    else
        return nullptr;
}

std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>>&
GaussianSceneLine::keyframes()
{
    return this->keyframes_;
}

std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>>
GaussianSceneLine::getAllKeyframes()
{
    std::unique_lock<std::mutex> lock_kfs(this->mutex_kfs_);
    return this->keyframes_;
}

void GaussianSceneLine::cachePoint3D(point3D_id_t point3D_id, Point3D& point3d)
{
    this->cached_point_cloud_[point3D_id] = point3d;
}

void GaussianSceneLine::cacheLine3D(line3D_id_t line3D_id, Line3D& line3d)
{
    this->cached_line3D_cloud_[line3D_id] = line3d;
}

void GaussianSceneLine::cacheLineSampledPnts3D(line3D_id_t line_id, Point3D& sample_pnt)
{
    // 1. 生成唯一的采样点 ID (建议：线 ID + 局部计数，或全局原子计数)
    point3D_id_t new_pnt_id = global_sample_counter_++;

    // 2. 标记属性
    sample_pnt.source_ = PointSourceType::LINE_SAMPLED;
    
    // 3. 存入全局点云（这是 GaussianModel 初始化唯一读取的地方）
    this->cached_point_cloud_[new_pnt_id] = sample_pnt;

    // 4. 建立索引关系（可选，用于后期的 Line-Coherence Loss 快速查找）
    this->line_to_sample_ids_.emplace(line_id, new_pnt_id);
}

Point3D& GaussianSceneLine::getPoint3D(point3D_id_t point3DId)
{
    if (this->cached_point_cloud_.find(point3DId) == this->cached_point_cloud_.end())
        std::cout << "GaussianSceneLine::getPoint3D(" << point3DId << ") invalid point Id, creating new point." << std::endl;

    return this->cached_point_cloud_[point3DId];
}

Line3D& GaussianSceneLine::getLine3D(line3D_id_t line3DId)
{
    if (this->cached_line3D_cloud_.find(line3DId) == this->cached_line3D_cloud_.end())
        std::cout << "GaussianSceneLine::getLine3D(" << line3DId << ") invalid line Id, creating new line." << std::endl;

    return this->cached_line3D_cloud_[line3DId];
}

void GaussianSceneLine::clearCachedPoint3D()
{
    this->cached_point_cloud_.clear();
}

void GaussianSceneLine::clearCachedLine3DToPoint3D()
{
    this->line_to_sample_ids_.clear();
}

void GaussianSceneLine::clearCachedLine3D()
{
    this->cached_line3D_cloud_.clear();
}

void GaussianSceneLine::applyScaledTransformation(
    const float s,
    const Sophus::SE3f T)
{
    // Apply the scaled transformation on gaussian keyframes
    for (auto& kfit : keyframes_) {
        std::shared_ptr<GaussianKeyframeLine> pkf = kfit.second;
        Sophus::SE3f Twc = pkf->getPosef().inverse();
        Twc.translation() *= s;
        Sophus::SE3f Tyc = T * Twc;
        Sophus::SE3f Tcy = Tyc.inverse();
        pkf->setPose(Tcy.unit_quaternion().cast<double>(), Tcy.translation().cast<double>());
        pkf->computeTransformTensors();
    }
}

/**
 * @brief 
 * 
 * @return std::tuple<Eigen::Vector3f, float> first=translate, second=radius
 */
std::tuple<Eigen::Vector3f, float>
GaussianSceneLine::getNerfppNorm()
{
    std::vector<Eigen::Matrix<float, 3, 1>> cam_centers;
    auto kfs = this->getAllKeyframes();
    std::size_t n_cams = kfs.size();
    cam_centers.reserve(n_cams);
    for (auto& kfit : kfs) {
        auto pkf = kfit.second;
        auto W2C = pkf->getWorld2View2();
        auto C2W = W2C.inverse();
        auto cam_center = C2W.block<3, 1>(0, 3);
        cam_centers.emplace_back(cam_center);
    }

    // get_center_and_diag(cam_centers)
    Eigen::Vector3f avg_cam_center;
    avg_cam_center.setZero();
    for (const auto& cam_center : cam_centers) {
        avg_cam_center.x() += cam_center.x();
        avg_cam_center.y() += cam_center.y();
        avg_cam_center.z() += cam_center.z();
    }
    avg_cam_center.x() /= n_cams;
    avg_cam_center.y() /= n_cams;
    avg_cam_center.z() /= n_cams;

    float max_dist = 0.0f; // diagonal
    for (std::size_t cam_idx = 0; cam_idx < n_cams; ++cam_idx) {
        float dist = (cam_centers[cam_idx] - avg_cam_center).norm();
        if (dist > max_dist)
            max_dist = dist;
    }

    float radius = max_dist * 1.1;

    Eigen::Vector3f translate = -avg_cam_center;

    return std::make_tuple(translate, radius);
}

bool GaussianSceneLine::isVoxelOccupied(const Eigen::Vector3f& pos, float voxel_size) {
    VoxelKeyGaussian key;
    key.x = static_cast<int>(std::floor(pos.x() / voxel_size));
    key.y = static_cast<int>(std::floor(pos.y() / voxel_size));
    key.z = static_cast<int>(std::floor(pos.z() / voxel_size));

    return line_voxel_grid_.find(key) != line_voxel_grid_.end();
}

void GaussianSceneLine::addPointToVoxel(const Eigen::Vector3f& pos, float voxel_size) {
    VoxelKeyGaussian key;
    key.x = static_cast<int>(std::floor(pos.x() / voxel_size));
    key.y = static_cast<int>(std::floor(pos.y() / voxel_size));
    key.z = static_cast<int>(std::floor(pos.z() / voxel_size));
    
    line_voxel_grid_.insert(key);
}

void GaussianSceneLine::saveDebugSceneToObj(const std::filesystem::path& filename)
{
    std::cout << "[GaussianScene] Saving debug OBJ to " << filename << std::endl;
    
    std::ofstream obj_file(filename);
    if (!obj_file.is_open()) {
        std::cerr << "[GaussianScene] Error: Cannot open file " << filename << std::endl;
        return;
    }

    int vertex_count = 1; // OBJ 索引从 1 开始

    obj_file << "# Debug Scene Export\n";
    obj_file << "# Red   = Original SLAM MapPoints\n";
    obj_file << "# Blue  = Line Sampled Points\n";
    obj_file << "# Green = Original MapLines Skeleton\n";

    // 1. 导出点云 (cached_point_cloud_)
    // 这里包含了 "普通点" 和 "线采样点"
    int num_slam_points = 0;
    int num_line_samples = 0;

    for (const auto& pair : cached_point_cloud_) {
        const auto& p = pair.second;
        
        // 区分颜色
        float r = 0.0f, g = 0.0f, b = 0.0f;
        
        if (p.source_ == PointSourceType::LINE_SAMPLED) {
            // 蓝色：线采样点
            r = 0.0f; g = 0.0f; b = 1.0f;
            num_line_samples++;
        } else {
            // 红色：普通 SLAM 点
            r = 1.0f; g = 0.0f; b = 0.0f;
            num_slam_points++;
        }

        // 写入顶点: v x y z r g b
        obj_file << "v " << p.xyz_(0) << " " << p.xyz_(1) << " " << p.xyz_(2) 
                 << " " << r << " " << g << " " << b << "\n";
        
        // 写入点图元
        obj_file << "p " << vertex_count++ << "\n";
    }

    // 2. 导出线段骨架 (cached_line3D_cloud_)
    // 这有助于你验证采样点是否真的位于这些绿线上
    int num_lines = 0;
    for (const auto& pair : cached_line3D_cloud_) {
        const auto& l = pair.second;

        // 线段端点使用绿色
        // Endpoint 1
        obj_file << "v " << l.p1_(0) << " " << l.p1_(1) << " " << l.p1_(2) << " 0.0 1.0 0.0\n";
        int idx1 = vertex_count++;

        // Endpoint 2
        obj_file << "v " << l.p2_(0) << " " << l.p2_(1) << " " << l.p2_(2) << " 0.0 1.0 0.0\n";
        int idx2 = vertex_count++;

        // 建立连线
        obj_file << "l " << idx1 << " " << idx2 << "\n";
        num_lines++;
    }

    obj_file.close();

    std::cout << "[GaussianScene] Saved successfully.\n"
              << "  - SLAM Points (Red): " << num_slam_points << "\n"
              << "  - Line Samples (Blue): " << num_line_samples << "\n"
              << "  - Line Segments (Green): " << num_lines << "\n";
}