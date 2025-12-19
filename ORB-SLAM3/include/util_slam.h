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
};

    
    
}


#endif