#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace loop_event_key
{
inline std::uint64_t fingerprint(
    unsigned long payload_map_gen,
    unsigned long current_map_gen,
    float scale,
    std::vector<std::pair<unsigned long, Eigen::Matrix4f>> kfs)
{
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&](std::uint64_t v) {
        h ^= v;
        h *= 1099511628211ULL;
    };
    auto mixf = [&](float v) {
        std::uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        mix(static_cast<std::uint64_t>(u));
    };
    mix(static_cast<std::uint64_t>(payload_map_gen));
    mix(static_cast<std::uint64_t>(current_map_gen));
    mixf(scale);
    mix(static_cast<std::uint64_t>(kfs.size()));
    std::sort(kfs.begin(), kfs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    for (const auto& kf : kfs)
    {
        mix(static_cast<std::uint64_t>(kf.first));
        for (int r = 0; r < 4; ++r)
            for (int c = 0; c < 4; ++c)
                mixf(kf.second(r, c));
    }
    return h;
}

inline bool map_matches(unsigned long payload_map_gen,
                        unsigned long current_map_gen)
{
    return payload_map_gen == 0 || payload_map_gen == current_map_gen;
}

} // namespace loop_event_key
