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

#include <Eigen/Core>

enum class PointSourceType {
    MAP_POINT = 0,
    LINE_SAMPLED = 1
};

class Point3D
{
public:
    Point3D()
        : xyz_(0.0, 0.0, 0.0),
          color_(0.0f, 0.0f, 0.0f),
          color256_(0, 0, 0),
          error_(-1.0)
    {
        source_ = PointSourceType::MAP_POINT;   //默认MAP_POINT
    }

    // 在 Point3D 类中
    void setAsLineSample(const Line3D& parent, double alpha, float step) {
        this->source_ = PointSourceType::LINE_SAMPLED;
        this->sample_step_ = step;
    
        // 几何插值
        Eigen::Vector3d dir = parent.p2_ - parent.p1_;
        this->xyz_ = parent.p1_ + alpha * dir;
        this->line_dir_ = dir.normalized().cast<float>();
    
        // 颜色插值 (LERP)
        this->color_ = (1.0f - (float)alpha) * parent.color1_ + (float)alpha * parent.color2_;
    }

public:
    Eigen::Vector3d xyz_;

    Eigen::Matrix<uint8_t, 3, 1> color256_; // not needed if we get color_ directly
    Eigen::Matrix<float, 3, 1> color_;

    PointSourceType source_;   // 新增 added by zdg
    float line_sample_step_;   // 仅对 line 点有效（可选）(to  delete next)
    Eigen::Vector3f line_dir_ = Eigen::Vector3f::UnitX(); // 线方向(单位向量)

    // 用于“严格对齐 CUDA my_radius 门槛”的初始化（推荐填写）
    float ref_depth_z_ = -1.0f;                // 该点在参考视角下的深度 z (camera/view space 的 z)
    float ref_focal_   = -1.0f;                // 参考视角的 focal（建议 (fx+fy)/2）

    float sample_step_ = -1.0f;                // 线采样步长（世界坐标）

    double error_;
};
