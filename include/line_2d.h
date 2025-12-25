/**
 * This file is part of Photo-SLAM
 *
 * Copyright (C) added by Daogui Zhang, Zhejiang university.
 *
 * Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
 * License as published by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Photo-SLAM_LINE is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along with Photo-SLAM_LINE.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include <Eigen/Core>

#include "types.h"
#include "point2d.h"

class Line2D
{
public:
    Line2D()
    {
        mline3D_id_ = std::numeric_limits<std::uint64_t>::max();
        p1_.xy_(0.0, 0.0);
        p2_.xy_(0.0, 0.0);
    }

public:
    
    Point2D p1_;
    Point2D p2_;
    std::uint64_t mline3D_id_;
};
    