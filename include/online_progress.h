#pragma once
#include <atomic>
#include <cstddef>

// Diagnostic-only input progress for the online checkpoint logger. The mapper
// cannot otherwise distinguish "input frame reached" from "mapper iteration".
namespace photo_slam_diag
{
inline std::atomic<long long>& inputFrame()
{
    static std::atomic<long long> v{-1};
    return v;
}
inline std::atomic<long long>& totalFrames()
{
    static std::atomic<long long> v{-1};
    return v;
}
inline void setInputFrame(long long i) { inputFrame().store(i, std::memory_order_relaxed); }
inline void setTotalFrames(long long n) { totalFrames().store(n, std::memory_order_relaxed); }
inline long long getInputFrame() { return inputFrame().load(std::memory_order_relaxed); }
inline long long getTotalFrames() { return totalFrames().load(std::memory_order_relaxed); }
} // namespace photo_slam_diag
