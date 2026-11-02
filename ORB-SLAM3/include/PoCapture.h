/**
* Minimal, immutable state pack captured at the pose-optimisation ENTRY.
*
* GATE: PHOTO_SLAM_PO_CAPTURE=<dir>  and  PHOTO_SLAM_PO_CAPTURE_TS=<lo>:<hi>
*       (timestamps of the frames to capture; no window => nothing captured)
*
* WHY: the only way to answer "does the outlier re-evaluation change THIS
* frame's decision" without another blind A/B run is to freeze the real input
* of the real failure frame and replay it offline through both library
* variants. Checksums are not enough: the replay needs geometry, observations,
* octave/information, association and the incoming outlier flags.
*
* READ-ONLY ON THE LIVE RUN. Nothing here is written back into the Frame or the
* MapPoints; the captured file is consumed by a separate process
* (tests/replay_po_state.cpp), so the replay's mutations cannot pollute a real
* run. No extra matching, no extra optimisation, no extra allocation on the
* hot path when the gate is closed.
*/

#ifndef ORB_SLAM3_PO_CAPTURE_H
#define ORB_SLAM3_PO_CAPTURE_H

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace ORB_SLAM3
{
namespace po_capture
{

// ---------------------------------------------------------------------------
// Path validation. A capture request with an unusable path must be REJECTED
// BEFORE the run, never discovered afterwards: a silent os.good()==false that
// still lets the harness report "capture done" is exactly the defect this
// replaces.
//
// Returns 0 when the directory is usable, or a negative code:
//   -1 not set, -2 empty, -3 does not exist, -4 not a directory,
//   -5 not writable, -6 cannot create a probe file.
// ---------------------------------------------------------------------------
inline int ValidateDir(const char* dir, std::string* why = nullptr)
{
    auto set = [&](int code, const char* msg) {
        if (why) *why = msg;
        return code;
    };
    if (dir == nullptr) return set(-1, "PHOTO_SLAM_PO_CAPTURE not set");
    if (dir[0] == '\0') return set(-2, "PHOTO_SLAM_PO_CAPTURE empty");

    struct stat st;
    if (::stat(dir, &st) != 0) return set(-3, "directory does not exist");
    if (!S_ISDIR(st.st_mode)) return set(-4, "path is not a directory");
    if (::access(dir, W_OK | X_OK) != 0) return set(-5, "directory not writable");

    // Probe: a directory can pass access() yet fail to create files (read-only
    // mount, quota). Prove writability with a real create+unlink.
    std::string probe = std::string(dir) + "/.po_capture_probe";
    FILE* f = std::fopen(probe.c_str(), "wb");
    if (f == nullptr) return set(-6, "cannot create a probe file");
    std::fclose(f);
    ::unlink(probe.c_str());
    return 0;
}

inline const char* ValidateName(int code)
{
    switch (code)
    {
        case 0:  return "OK";
        case -1: return "NOT_SET";
        case -2: return "EMPTY";
        case -3: return "MISSING";
        case -4: return "NOT_A_DIR";
        case -5: return "NOT_WRITABLE";
        case -6: return "CANNOT_CREATE";
        default: return "UNKNOWN";
    }
}

struct Gate
{
    bool   on   = false;
    int    validateCode = -1;      // 0 == armed; negative == rejected
    long long writeFailures = 0;   // any failure makes the capture INVALID
    long long written = 0;
    long long duplicates = 0;
    double lo   = 0.0;
    double hi   = 0.0;
    std::string dir;
    bool   wroteAny = false;
};

inline Gate& G()
{
    static Gate* g = []() {
        Gate* n = new Gate();
        const char* d = std::getenv("PHOTO_SLAM_PO_CAPTURE");
        const char* w = std::getenv("PHOTO_SLAM_PO_CAPTURE_TS");
        if (d != nullptr && w != nullptr)
        {
            double a = 0.0, b = 0.0;
            if (std::sscanf(w, "%lf:%lf", &a, &b) == 2 && b >= a)
            {
                std::string why;
                const int code = ValidateDir(d, &why);
                n->validateCode = code;
                if (code == 0)
                {
                    n->on = true;
                    n->dir = d;
                    n->lo = a;
                    n->hi = b;
                }
                else
                {
                    // Loud and immediate: an unusable capture path is a hard
                    // error, never a silent no-op.
                    std::fprintf(stderr, "[PO-CAPTURE-ERROR] path rejected: dir=%s code=%d(%s) %s\n",
                                 d, code, ValidateName(code), why.c_str());
                }
            }
        }
        return n;
    }();
    return *g;
}

inline std::mutex& Mtx() { static std::mutex* m = new std::mutex(); return *m; }

inline bool Wants(double ts)
{
    const Gate& g = G();
    return g.on && ts >= g.lo && ts <= g.hi;
}

inline std::string PathFor(const char* tag, double ts)
{
    std::ostringstream o;
    o.setf(std::ios::fixed);
    o.precision(6);
    o << G().dir << "/po_" << tag << "_" << ts << ".txt";
    return o.str();
}

} // namespace po_capture
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_PO_CAPTURE_H
