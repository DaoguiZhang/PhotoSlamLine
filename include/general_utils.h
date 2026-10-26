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

#include <torch/torch.h>

namespace general_utils
{

inline torch::Tensor inverse_sigmoid(const torch::Tensor &x)
{
    return torch::log(x / (1 - x));
}

inline torch::Tensor build_rotation(torch::Tensor &r)
{
    // Autograd-safe functional build (the previous in-place copy_ into views of
    // a zero tensor corrupted the backward graph -> CUDA illegal memory access
    // when the result fed a loss). Quaternion layout: (w, x, y, z).
    // Clamp sum-of-squares BEFORE sqrt so sqrt backward never sees 0.
    using namespace torch::indexing;
    auto norm = torch::sqrt(torch::sum(r * r, /*dim=*/1, /*keepdim=*/true).clamp_min(1e-16f));
    auto q = r / norm;
    auto w = q.index({Slice(), 0});
    auto x = q.index({Slice(), 1});
    auto y = q.index({Slice(), 2});
    auto z = q.index({Slice(), 3});

    auto R = torch::stack({
        torch::stack({1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)}, 1),
        torch::stack({2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)}, 1),
        torch::stack({2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}, 1),
    }, 1);  // [N, 3, 3]
    return R;
}

}
