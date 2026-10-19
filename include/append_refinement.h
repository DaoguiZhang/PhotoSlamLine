/**
 * Append-phase refinement helpers (PHOTO_SLAM_APPEND_STEPS).
 *
 * This header intentionally contains only CONTROL and AUDIT code shared by
 * GaussianMapper (point build) and GaussianMapperLine (line build):
 *   - environment parsing,
 *   - pre-generated, recordable append view sequence,
 *   - camera / keyframe stability snapshots,
 *   - optimizer parameter identity + Adam-step snapshots,
 *   - production mapping-operation queue settling.
 *
 * It does NOT contain a training algorithm. Both mappers reuse their existing
 * trainForOneIteration*() bodies; the only extra work in those bodies is the
 * append gating (densify/clone/split/prune/opacity reset off) and counters.
 */
#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace append_refinement
{

struct Config
{
    int steps = 0;                  // PHOTO_SLAM_APPEND_STEPS (0 = disabled)
    int seed = 12345;               // PHOTO_SLAM_APPEND_SEED
    int max_no_progress = 50;       // PHOTO_SLAM_APPEND_MAX_NO_PROGRESS
    int queue_timeout_ms = 10000;   // PHOTO_SLAM_APPEND_QUEUE_TIMEOUT_MS
    int queue_quiet_ms = 500;       // PHOTO_SLAM_APPEND_QUEUE_QUIET_MS

    bool enabled() const { return steps > 0; }
};

inline Config readConfig()
{
    Config c;
    if (const char* e = std::getenv("PHOTO_SLAM_APPEND_STEPS"))
        c.steps = std::atoi(e);
    if (const char* e = std::getenv("PHOTO_SLAM_APPEND_SEED"))
        c.seed = std::atoi(e);
    if (const char* e = std::getenv("PHOTO_SLAM_APPEND_MAX_NO_PROGRESS"))
        c.max_no_progress = std::atoi(e);
    if (const char* e = std::getenv("PHOTO_SLAM_APPEND_QUEUE_TIMEOUT_MS"))
        c.queue_timeout_ms = std::atoi(e);
    if (const char* e = std::getenv("PHOTO_SLAM_APPEND_QUEUE_QUIET_MS"))
        c.queue_quiet_ms = std::atoi(e);
    if (c.steps < 0)
        c.steps = 0;
    if (c.max_no_progress < 1)
        c.max_no_progress = 1;
    if (c.queue_timeout_ms < 0)
        c.queue_timeout_ms = 0;
    if (c.queue_quiet_ms < 0)
        c.queue_quiet_ms = 0;
    return c;
}

inline std::string nowStamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    std::ostringstream os;
    os << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return os.str();
}

inline void appendLine(const std::filesystem::path& path, const std::string& line)
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::out | std::ios::app);
    out << line << '\n';
}

inline void writeKv(const std::filesystem::path& path, const std::string& key, const std::string& value)
{
    appendLine(path, key + "=" + value);
}

/**
 * Deterministic append-phase view order.
 *
 * The sequence is built ONLY from the keyframes already present in the scene;
 * it does not look at any metric/evaluation result. Each cycle reshuffles all
 * keyframes with a fixed seed, so every keyframe is used as evenly as possible
 * while the exact order is reproducible and can be written to disk.
 */
inline std::vector<std::size_t> buildViewSequence(std::size_t num_keyframes, int steps, int seed)
{
    std::vector<std::size_t> seq;
    if (num_keyframes == 0 || steps <= 0)
        return seq;

    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<std::size_t> order(num_keyframes);
    std::iota(order.begin(), order.end(), static_cast<std::size_t>(0));
    seq.reserve(static_cast<std::size_t>(steps));

    while (seq.size() < static_cast<std::size_t>(steps))
    {
        std::shuffle(order.begin(), order.end(), rng);
        for (std::size_t idx : order)
        {
            if (seq.size() >= static_cast<std::size_t>(steps))
                break;
            seq.push_back(idx);
        }
    }
    return seq;
}

struct KeyframeState
{
    unsigned long fid = 0;
    int camera_id = 0;
    double qw = 0.0, qx = 0.0, qy = 0.0, qz = 0.0;
    double tx = 0.0, ty = 0.0, tz = 0.0;
    int width = 0;
    int height = 0;
    std::vector<float> intrinsics;
};

template <typename SceneT>
inline std::vector<unsigned long> keyframeFids(SceneT& scene)
{
    std::vector<unsigned long> fids;
    fids.reserve(scene.keyframes().size());
    for (auto& kv : scene.keyframes())
        fids.push_back(static_cast<unsigned long>(kv.second->fid_));
    return fids;
}

template <typename SceneT>
inline std::vector<KeyframeState> snapshotKeyframeStates(SceneT& scene)
{
    std::vector<KeyframeState> out;
    out.reserve(scene.keyframes().size());
    for (auto& kv : scene.keyframes())
    {
        auto& kf = kv.second;
        const auto pose = kf->getPose();
        const auto q = pose.unit_quaternion();
        const auto t = pose.translation();
        KeyframeState s;
        s.fid = static_cast<unsigned long>(kf->fid_);
        s.camera_id = static_cast<int>(kf->camera_id_);
        s.qw = q.w();
        s.qx = q.x();
        s.qy = q.y();
        s.qz = q.z();
        s.tx = t.x();
        s.ty = t.y();
        s.tz = t.z();
        s.width = kf->image_width_;
        s.height = kf->image_height_;
        s.intrinsics = kf->intr_;
        out.push_back(std::move(s));
    }
    return out;
}

// Returns {max pose delta, max intrinsics delta, changed_keyframe_count}.
inline std::array<double, 3> compareKeyframeStates(const std::vector<KeyframeState>& a,
                                                   const std::vector<KeyframeState>& b)
{
    std::array<double, 3> res = {0.0, 0.0, 0.0};
    if (a.size() != b.size())
    {
        res[0] = std::numeric_limits<double>::infinity();
        res[2] = 1.0;
        return res;
    }
    long long changed = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        const auto& x = a[i];
        const auto& y = b[i];
        bool ch = (x.fid != y.fid) || (x.camera_id != y.camera_id) ||
                  (x.width != y.width) || (x.height != y.height);
        const double d = std::max({
            std::abs(x.qw - y.qw), std::abs(x.qx - y.qx),
            std::abs(x.qy - y.qy), std::abs(x.qz - y.qz),
            std::abs(x.tx - y.tx), std::abs(x.ty - y.ty),
            std::abs(x.tz - y.tz)});
        res[0] = std::max(res[0], d);
        if (x.intrinsics.size() != y.intrinsics.size())
        {
            res[1] = std::numeric_limits<double>::infinity();
            ch = true;
        }
        else
        {
            for (std::size_t k = 0; k < x.intrinsics.size(); ++k)
                res[1] = std::max(res[1], std::abs(static_cast<double>(x.intrinsics[k]) -
                                                   static_cast<double>(y.intrinsics[k])));
        }
        if (ch)
            ++changed;
    }
    res[2] = static_cast<double>(changed);
    return res;
}

template <typename SceneT>
inline std::uint64_t cameraFingerprint(SceneT& scene)
{
    std::uint64_t h = 1469598103934665603ULL;
    auto mix = [&h](const void* data, std::size_t n) {
        const unsigned char* p = static_cast<const unsigned char*>(data);
        for (std::size_t i = 0; i < n; ++i)
        {
            h ^= static_cast<std::uint64_t>(p[i]);
            h *= 1099511628211ULL;
        }
    };
    for (auto& kv : scene.cameras_)
    {
        const int id = static_cast<int>(kv.first);
        mix(&id, sizeof(id));
        auto& c = kv.second;
        const int w = static_cast<int>(c.width_);
        const int hgt = static_cast<int>(c.height_);
        const int model = c.model_id_;
        mix(&w, sizeof(w));
        mix(&hgt, sizeof(hgt));
        mix(&model, sizeof(model));
        for (double v : c.params_)
            mix(&v, sizeof(v));
    }
    return h;
}

struct ParamSnapshot
{
    std::vector<torch::Tensor> values;          // CPU clones
    std::vector<std::uintptr_t> impl_ptrs;
    std::vector<std::uintptr_t> data_ptrs;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<long long> counts;
    std::vector<std::size_t> group_index;
    std::vector<std::size_t> param_index;
};

inline ParamSnapshot captureParamSnapshot(torch::optim::Adam& opt)
{
    ParamSnapshot s;
    for (std::size_t gi = 0; gi < opt.param_groups().size(); ++gi)
    {
        auto& group = opt.param_groups()[gi];
        for (std::size_t pi = 0; pi < group.params().size(); ++pi)
        {
            auto& p = group.params()[pi];
            s.impl_ptrs.push_back(reinterpret_cast<std::uintptr_t>(p.unsafeGetTensorImpl()));
            s.data_ptrs.push_back(reinterpret_cast<std::uintptr_t>(p.data_ptr()));
            s.shapes.push_back(p.sizes().vec());
            s.counts.push_back(static_cast<long long>(p.numel()));
            s.group_index.push_back(gi);
            s.param_index.push_back(pi);
            s.values.push_back(p.detach().to(torch::kCPU).clone());
        }
    }
    return s;
}

struct ParamDiff
{
    long long total_elements = 0;
    long long changed_elements = 0;
    long long nan_elements = 0;
    long long inf_elements = 0;
    double max_abs_delta = 0.0;
    double l2_delta = 0.0;
    bool identity_preserved = true;
    bool shape_preserved = true;
    long long changed_param_tensors = 0;
    long long total_param_tensors = 0;
};

struct ParamTensorDiff
{
    std::size_t group_index = 0;
    std::size_t param_index = 0;
    long long count = 0;
    long long changed = 0;
    double max_abs_delta = 0.0;
    double l2_delta = 0.0;
    long long nan_elements = 0;
    long long inf_elements = 0;
    bool identity_ok = true;
    bool shape_ok = true;
};

inline std::vector<ParamTensorDiff> diffParamSnapshotDetailed(const ParamSnapshot& before,
                                                              torch::optim::Adam& opt)
{
    std::vector<ParamTensorDiff> out;
    std::size_t idx = 0;
    for (std::size_t gi = 0; gi < opt.param_groups().size(); ++gi)
    {
        auto& group = opt.param_groups()[gi];
        for (std::size_t pi = 0; pi < group.params().size(); ++pi, ++idx)
        {
            auto& p = group.params()[pi];
            ParamTensorDiff d;
            d.group_index = gi;
            d.param_index = pi;
            if (idx >= before.values.size())
            {
                d.identity_ok = false;
                out.push_back(d);
                continue;
            }
            d.identity_ok =
                reinterpret_cast<std::uintptr_t>(p.unsafeGetTensorImpl()) == before.impl_ptrs[idx] &&
                reinterpret_cast<std::uintptr_t>(p.data_ptr()) == before.data_ptrs[idx];
            d.shape_ok = (p.sizes().vec() == before.shapes[idx]);
            d.count = static_cast<long long>(p.numel());
            if (!d.shape_ok)
            {
                out.push_back(d);
                continue;
            }
            auto cur = p.detach().to(torch::kCPU);
            auto delta = (cur - before.values[idx]).abs();
            d.changed = static_cast<long long>((delta > 0).sum().item<int64_t>());
            if (d.changed > 0)
                d.max_abs_delta = delta.max().item<double>();
            d.l2_delta = std::sqrt(delta.pow(2).sum().item<double>());
            d.nan_elements = static_cast<long long>(torch::isnan(cur).sum().item<int64_t>());
            d.inf_elements = static_cast<long long>(torch::isinf(cur).sum().item<int64_t>());
            out.push_back(d);
        }
    }
    return out;
}

inline ParamDiff diffParamSnapshot(const ParamSnapshot& before, torch::optim::Adam& opt)
{
    ParamDiff d;
    std::size_t idx = 0;
    for (std::size_t gi = 0; gi < opt.param_groups().size(); ++gi)
    {
        auto& group = opt.param_groups()[gi];
        for (std::size_t pi = 0; pi < group.params().size(); ++pi, ++idx)
        {
            auto& p = group.params()[pi];
            ++d.total_param_tensors;
            if (idx >= before.values.size())
            {
                d.identity_preserved = false;
                continue;
            }
            const auto impl = reinterpret_cast<std::uintptr_t>(p.unsafeGetTensorImpl());
            const auto data = reinterpret_cast<std::uintptr_t>(p.data_ptr());
            if (impl != before.impl_ptrs[idx] || data != before.data_ptrs[idx])
                d.identity_preserved = false;
            if (p.sizes().vec() != before.shapes[idx])
            {
                d.shape_preserved = false;
                continue;
            }

            auto cur = p.detach().to(torch::kCPU);
            const auto& ref = before.values[idx];
            const long long n = static_cast<long long>(cur.numel());
            d.total_elements += n;
            if (n == 0)
                continue;

            auto delta = (cur - ref).abs();
            const long long changed = static_cast<long long>((delta > 0).sum().item<int64_t>());
            d.changed_elements += changed;
            if (changed > 0)
                ++d.changed_param_tensors;
            d.max_abs_delta = std::max(d.max_abs_delta, delta.max().item<double>());
            d.l2_delta += delta.pow(2).sum().item<double>();
            d.nan_elements += static_cast<long long>(torch::isnan(cur).sum().item<int64_t>());
            d.inf_elements += static_cast<long long>(torch::isinf(cur).sum().item<int64_t>());
        }
    }
    d.l2_delta = std::sqrt(d.l2_delta);
    if (idx != before.values.size())
        d.identity_preserved = false;
    return d;
}

struct GroupGradInfo
{
    std::size_t group_index = 0;
    bool has_param = false;
    bool grad_defined = false;
    double grad_l2 = -1.0;
    long long adam_step = -1;
    std::uintptr_t impl_ptr = 0;
    std::uintptr_t data_ptr = 0;
};

inline std::vector<GroupGradInfo> captureGroupGradInfo(torch::optim::Adam& opt)
{
    std::vector<GroupGradInfo> out;
    auto& state = opt.state();
    for (std::size_t gi = 0; gi < opt.param_groups().size(); ++gi)
    {
        auto& group = opt.param_groups()[gi];
        GroupGradInfo info;
        info.group_index = gi;
        if (!group.params().empty())
        {
            auto& p = group.params()[0];
            info.has_param = true;
            info.impl_ptr = reinterpret_cast<std::uintptr_t>(p.unsafeGetTensorImpl());
            info.data_ptr = reinterpret_cast<std::uintptr_t>(p.data_ptr());
            auto g = p.grad();
            info.grad_defined = g.defined();
            if (info.grad_defined)
                info.grad_l2 = g.norm().item<double>();
            auto it = state.find(p.unsafeGetTensorImpl());
            if (it != state.end())
                info.adam_step = static_cast<long long>(
                    static_cast<torch::optim::AdamParamState&>(*it->second).step());
        }
        out.push_back(info);
    }
    return out;
}

inline std::vector<long long> captureAdamSteps(torch::optim::Adam& opt)
{
    std::vector<long long> out;
    auto& state = opt.state();
    for (auto& group : opt.param_groups())
    {
        for (auto& p : group.params())
        {
            long long step = -1;
            auto it = state.find(p.unsafeGetTensorImpl());
            if (it != state.end())
                step = static_cast<long long>(
                    static_cast<torch::optim::AdamParamState&>(*it->second).step());
            out.push_back(step);
        }
    }
    return out;
}

/**
 * Wait until the SLAM mapping-operation queue is empty AND stays empty for
 * `quiet_ms`. Any pending operation is processed through the caller's normal
 * production path (so it is not silently dropped). Returns false with a reason
 * if the queue cannot be settled before the timeout.
 */
inline bool drainAndSettleQueue(const std::function<bool()>& has_ops,
                               const std::function<void()>& drain_one_batch,
                               int timeout_ms,
                               int quiet_ms,
                               int& drained_batches,
                               std::string& reason)
{
    drained_batches = 0;
    const auto start = std::chrono::steady_clock::now();
    auto last_progress = start;
    while (true)
    {
        bool did_work = false;
        while (has_ops())
        {
            drain_one_batch();
            ++drained_batches;
            did_work = true;
            last_progress = std::chrono::steady_clock::now();
            if (drained_batches > 100000)
            {
                reason = "queue_drain_runaway";
                return false;
            }
        }
        const auto now = std::chrono::steady_clock::now();
        const auto since_progress =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress).count();
        const auto since_start =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
        if (!has_ops() && (did_work || since_progress >= quiet_ms))
            return true;
        if (since_start >= timeout_ms)
        {
            reason = has_ops() ? "queue_not_empty_timeout" : "queue_settle_timeout";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

inline std::string boolStr(bool b) { return b ? "1" : "0"; }

} // namespace append_refinement
