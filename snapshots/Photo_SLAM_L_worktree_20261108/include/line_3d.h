/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) 2025-2026 Added by ZDG Zhejiang University.
 *
 * Photo-SLAM_Line is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM_Line is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM_Line.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <Eigen/Core>

class Line3D
{
public:
    Line3D()
        : p1_(0.0, 0.0, 0.0),
          p2_(0.0, 0.0, 0.0),
          color1_(0.0f, 0.0f, 0.0f),
          color2_(0.0f, 0.0f, 0.0f),
          color256_1_(0, 0, 0),
          color256_2_(0, 0, 0),
          error_(-1.0)
    {}

public:
    Eigen::Vector3d p1_;
    Eigen::Vector3d p2_;

    Eigen::Matrix<uint8_t, 3, 1> color256_1_; // not needed if we get color_ directly(start point)
    Eigen::Matrix<uint8_t, 3, 1> color256_2_; // not needed if we get color_ directly(end point)
    Eigen::Matrix<float, 3, 1> color1_;
    Eigen::Matrix<float, 3, 1> color2_;

    double error_;
};
