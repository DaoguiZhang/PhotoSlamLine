/**
 * Debug helpers for the Stereo + Line pipeline.
 *
 * Per-frame statistics are gated by the environment variable
 * `PHOTO_SLAM_DEBUG_STEREO_LINE=1` and are OFF by default so that normal runs
 * do not spam the terminal.
 */

#ifndef STEREOLINEDEBUG_H
#define STEREOLINEDEBUG_H

#include <cstdlib>
#include <string>

namespace ORB_SLAM3
{

inline bool IsStereoLineDebugEnabled()
{
    static const bool enabled = []() {
        const char* v = std::getenv("PHOTO_SLAM_DEBUG_STEREO_LINE");
        return v != nullptr && std::string(v) == "1";
    }();
    return enabled;
}

} // namespace ORB_SLAM3

#endif // STEREOLINEDEBUG_H
