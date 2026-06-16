#include "util_slam.h"

namespace ORB_SLAM3
{
    std::size_t UtilSlam::MakeDepthKey(const MapLine* ml, const KeyFrame* kf, int idx)
    {
        // convert pointers to integers
        uintptr_t a = reinterpret_cast<uintptr_t>(ml);
        uintptr_t b = reinterpret_cast<uintptr_t>(kf);

        // mix hash values
        std::hash<uintptr_t> hptr;
        std::hash<int>       hint;

        size_t h1 = hptr(a);
        size_t h2 = hptr(b);
        size_t h3 = hint(idx);

        // simple XOR mixing
        return (h1) ^ (h2 << 1) ^ (h3 << 2);
    }
}