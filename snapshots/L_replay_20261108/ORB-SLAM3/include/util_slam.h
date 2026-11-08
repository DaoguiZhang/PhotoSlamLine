#ifndef UTIL_SLAM_H
#define UTIL_SLAM_H

#include<map>
#include<unordered_map>
#include <vector>
#include "KeyFrame.h"
#include "Frame.h"
#include "MapPoint.h"
#include "Map.h"
#include "MapLine.h"
#include <unordered_set>

namespace ORB_SLAM3
{

class KeyFrame;
class MapLine;
class Frame;
class MapPoint;


class UtilSlam
{
public:
    static std::size_t MakeDepthKey(const MapLine* ml, const KeyFrame* kf, int idx);

    static inline double ClampD_ZDG(double x, double lo, double hi)
    {
        return std::max(lo, std::min(hi, x));
    }
    // 安全归一化：返回 false 表示退化（不能用）
    static inline bool SafeNormalize_ZDG(const Eigen::Vector3d& s,
                          Eigen::Vector3d& d,
                          double& l)
    {
        l = s.norm();
        if (!std::isfinite(l) || l < 1e-12)
            return false;
        d = s / l;
        return d.allFinite();
    }
};

    
    
}


#endif