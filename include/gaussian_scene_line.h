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

#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>    //added by zdg
#include <memory>
#include <mutex>
#include <tuple>
#include <filesystem>

#include "types.h"
#include "camera.h"
#include "point3d.h"
#include "point2d.h"
#include "line_2d.h"
#include "line_3d.h"
#include "gaussian_parameters_line.h"
#include "gaussian_model_line.h"
#include "gaussian_keyframe_line.h"

//added by zdg
// 定义 Voxel 的 Key (哈希用)
struct VoxelKeyGaussian {
    int x, y, z;

    bool operator==(const VoxelKeyGaussian& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

// Voxel 坐标的哈希函数
struct VoxelHasherGaussian {
    size_t operator()(const VoxelKeyGaussian& k) const {
        // 使用质数进行哈希混合，减少碰撞
        return ((std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1)) >> 1) ^ (std::hash<int>()(k.z) << 1);
    }
};

class GaussianSceneLine
{
public:
    GaussianSceneLine(
        GaussianModelParamsLine& args,
        int load_iteration = 0,
        bool shuffle = true,
        std::vector<float> resolution_scales = {1.0f});

public:
    void addCamera(Camera& camera);
    Camera& getCamera(camera_id_t cameraId);

    void addKeyframe(std::shared_ptr<GaussianKeyframeLine> new_kf, bool* shuffled);
    std::shared_ptr<GaussianKeyframeLine> getKeyframe(std::size_t fid);
    std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>>& keyframes();
    std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>> getAllKeyframes();

    //void addKeyframeWithLine(std::shared_ptr<GaussianKeyframeLine> new_kf, bool* shuffled);
    //std::shared_ptr<GaussianKeyframeLine> getKeyframeWithLine(std::size_t fid);
    //std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>>& keyframesWithLine();
    //std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>> getAllKeyframesWithLine();

    void cachePoint3D(point3D_id_t point3D_id, Point3D& point3d);
    void cacheLine3D(line3D_id_t line3D_id, Line3D& line3d);    //Set line3d into cache
    void cacheLineSampledPnts3D(line3D_id_t line3D_id, Point3D& sample_pnt);   //
    Point3D& getPoint3D(point3D_id_t point3DId);
    Line3D& getLine3D(line3D_id_t line3DId);    //Get line3D from cache
    void clearCachedPoint3D();
    void clearCachedLine3D();   //Clear line3D cache
    void clearCachedLine3DToPoint3D();  //Clear line3D to point3D cache

    // 检查并更新 Voxel Grid
    bool isVoxelOccupied(const Eigen::Vector3f& pos, float voxel_size = 0.03f);
    void addPointToVoxel(const Eigen::Vector3f& pos, float voxel_size = 0.03f);
    void clearVoxelGrid() { line_voxel_grid_.clear(); }

    int getVoxelCount() const { return line_voxel_grid_.size(); }

    //Convert line3D to point3D(TO DO NEXT)
    void convertLines3DToPoints3D();  //sample the 3D line and store into Point3D
    void SamplePointsAlongLines3D();    //sample points along 3D lines (first sample 2D lines in images then get point 2d image, then project into 3D)

    void applyScaledTransformation(
        const float s = 1.0,
        const Sophus::SE3f T = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero()));

    std::tuple<Eigen::Vector3f, float> getNerfppNorm();

    std::tuple<std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>>,
               std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>>>
        splitTrainAndTestKeyframes(const float test_ratio);

    // 下面添加debug函数
    void saveDebugSceneToObj(const std::filesystem::path& filename);
    

public:
    float cameras_extent_; ///< scene_info.nerf_normalization["radius"]

    int loaded_iter_;
    point3D_id_t global_sample_counter_ = 0; // Global counter for sampled points along lines
    std::map<camera_id_t, Camera> cameras_;
    std::map<std::size_t, std::shared_ptr<GaussianKeyframeLine>> keyframes_;
    // 3. 所有的点（普通点 + 线采样点）都统一存在这里，方便 Gaussians->createFromPcd 调用
    std::map<point3D_id_t, Point3D> cached_point_cloud_;
    
    // 1. 原始 3D 线段本体（由 ORB-SLAM3 传入，仅存端点）
    std::map<line3D_id_t, Line3D> cached_line3D_cloud_;

    // 2. 核心改进：线到采样点的映射
    // 使用 multimap 或者 vector，允许一个 Line ID 对应多个采样点 ID
    std::unordered_multimap<line3D_id_t, point3D_id_t> line_to_sample_ids_;


protected:

    // 存储已占用的网格
    std::unordered_set<VoxelKeyGaussian, VoxelHasherGaussian> line_voxel_grid_;
    std::mutex mutex_kfs_;
};
