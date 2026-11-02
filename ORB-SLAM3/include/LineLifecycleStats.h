/**
* Minimal lifecycle counters for the line map (diagnostic only).
*
* Scope: exactly the stages needed to locate where the line pipeline falls to
* zero. Deliberately NOT a logging platform.
*
* GATE: PHOTO_SLAM_LINE_LIFECYCLE_CSV=<path>   (unset => zero overhead, no I/O)
*
* Levels are kept strictly separate and must never be summed together:
*   - kf_*        : per keyframe / per invocation of CreateNewMapLines()
*   - neigh_*     : neighbour keyframes examined
*   - pair_*      : candidate line pairs returned by the matcher
*   - tri_*       : triangulation attempts
*   - ml_*        : UNIQUE MapLine objects actually constructed (deduped by id)
*   - atlas_*     : census of lines that survived (global / local)
*
* Every rejection is attributed to exactly one earliest stage, so the funnel
* kf -> neigh -> pair -> tri -> ml -> atlas cannot double count.
*/

#ifndef ORB_SLAM3_LINE_LIFECYCLE_STATS_H
#define ORB_SLAM3_LINE_LIFECYCLE_STATS_H

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace ORB_SLAM3
{
namespace line_lifecycle
{

struct Counters
{
    // --- calls ---------------------------------------------------------
    long long kf_calls          = 0;   // CreateNewMapLines() entered
    long long kf_early_newkf    = 0;   // returned because CheckNewKeyFrames()
    long long kf_no_neighbour   = 0;   // zero covisibility neighbours

    // --- neighbour level ----------------------------------------------
    long long neigh_examined    = 0;
    long long rej_base_lt_mb    = 0;   // stereo: baseline < pKF2->mb
    long long rej_median_nopts  = 0;   // mono: ComputeSceneMedianDepth<=0
                                       //       (== neighbour has no MapPoint)
    long long rej_base_over_med = 0;   // mono: baseline/med < 0.01

    // --- pair level ----------------------------------------------------
    long long pair_matched      = 0;   // sum of vMatchedIdx.size()
    long long pair_empty        = 0;   // neighbours returning 0 pairs
    long long rej_pair_short    = 0;   // lineLength < minLinePixels
    long long rej_pair_lenratio = 0;
    long long rej_pair_angle    = 0;

    // --- triangulation -------------------------------------------------
    long long tri_attempt       = 0;   // mono fallback entered
    long long tri_success       = 0;
    long long tri_fail          = 0;   // Triangulate() returned false
    long long rej_reproj_new    = 0;   // fixed check rejected (either KF)
    long long rej_front         = 0;   // z<=0 after creation
    long long rej_ratio_dist    = 0;
    long long rej_far           = 0;

    // --- unique MapLine objects ---------------------------------------
    long long ml_created        = 0;
    long long ml_added_atlas    = 0;
    long long ml_obs_links      = 0;

    // --- shadow (old vs new decision, same pose snapshot) --------------
    long long shadow_pairs      = 0;
    long long shadow_agree      = 0;
    long long shadow_old_pass_new_fail = 0;
    long long shadow_old_fail_new_pass = 0;
    long long shadow_first_diff_kf = -1;
    int       shadow_first_diff_code = 0;

    // --- culling -------------------------------------------------------
    long long cull_erased       = 0;
    long long cull_set_bad      = 0;

    // --- LSD detection (real entry only) -------------------------------
    long long lsd_calls         = 0;
    long long lsd_raw_lines     = 0;
    long long lsd_out_lines     = 0;

    // --- census at last keyframe ---------------------------------------
    long long atlas_lines       = 0;
    long long local_lines       = 0;
    long long cur_kf_lines      = 0;
};

inline Counters& C()
{
    // Intentionally leaked: avoids static destruction order issues when the
    // counters are touched from an atexit handler.
    static Counters* s = new Counters();
    return *s;
}

inline const char* CsvPath()
{
    static const char* p = std::getenv("PHOTO_SLAM_LINE_LIFECYCLE_CSV");
    return p;
}

inline bool Enabled()
{
    const char* p = CsvPath();
    return p != nullptr && p[0] != '\0';
}

// Shadow comparison gate, independent of the CSV so the CSV can be produced
// without paying for the extra (pure) evaluation.
inline bool ShadowEnabled()
{
    if (!Enabled()) return false;
    const char* p = std::getenv("PHOTO_SLAM_LINE_LIFECYCLE_SHADOW");
    return p != nullptr && p[0] == '1';
}

inline std::ofstream& Sink()
{
    static std::ofstream* s = nullptr;
    if (s == nullptr)
    {
        s = new std::ofstream(CsvPath(), std::ios::out | std::ios::trunc);
        if (s->good())
        {
            (*s) << "# Photo-SLAM-L line lifecycle counters (cumulative, one row"
                    " per processed keyframe)\n";
            (*s) << "# levels are NOT additive across kf/neigh/pair/tri/ml\n";
            (*s) << "event,kf_id,"
                    "kf_calls,kf_early_newkf,kf_no_neighbour,"
                    "neigh_examined,rej_base_lt_mb,rej_median_nopts,"
                    "rej_base_over_med,"
                    "pair_matched,pair_empty,rej_pair_short,rej_pair_lenratio,"
                    "rej_pair_angle,"
                    "tri_attempt,tri_success,tri_fail,rej_reproj_new,"
                    "rej_front,rej_ratio_dist,rej_far,"
                    "ml_created,ml_added_atlas,ml_obs_links,"
                    "shadow_pairs,shadow_agree,shadow_old_pass_new_fail,"
                    "shadow_old_fail_new_pass,shadow_first_diff_kf,"
                    "shadow_first_diff_code,"
                    "cull_erased,cull_set_bad,"
                    "lsd_calls,lsd_raw_lines,lsd_out_lines,"
                    "atlas_lines,local_lines,cur_kf_lines\n";
            s->flush();
        }
    }
    return *s;
}

inline void Emit(const char* event, long long kfId)
{
    if (!Enabled()) return;
    std::ofstream& o = Sink();
    if (!o.good()) return;
    const Counters& c = C();
    o << event << ',' << kfId << ','
      << c.kf_calls << ',' << c.kf_early_newkf << ',' << c.kf_no_neighbour << ','
      << c.neigh_examined << ',' << c.rej_base_lt_mb << ','
      << c.rej_median_nopts << ',' << c.rej_base_over_med << ','
      << c.pair_matched << ',' << c.pair_empty << ','
      << c.rej_pair_short << ',' << c.rej_pair_lenratio << ','
      << c.rej_pair_angle << ','
      << c.tri_attempt << ',' << c.tri_success << ',' << c.tri_fail << ','
      << c.rej_reproj_new << ',' << c.rej_front << ',' << c.rej_ratio_dist
      << ',' << c.rej_far << ','
      << c.ml_created << ',' << c.ml_added_atlas << ',' << c.ml_obs_links << ','
      << c.shadow_pairs << ',' << c.shadow_agree << ','
      << c.shadow_old_pass_new_fail << ',' << c.shadow_old_fail_new_pass << ','
      << c.shadow_first_diff_kf << ',' << c.shadow_first_diff_code << ','
      << c.cull_erased << ',' << c.cull_set_bad << ','
      << c.lsd_calls << ',' << c.lsd_raw_lines << ',' << c.lsd_out_lines << ','
      << c.atlas_lines << ',' << c.local_lines << ',' << c.cur_kf_lines << '\n';
    o.flush();
}

// One-line human summary on stderr at exit, same shape as [LSD-COUNT] so the
// existing log scrapers keep working.
inline void RegisterExitSummary()
{
    static bool sRegistered = false;
    if (sRegistered || !Enabled()) return;
    sRegistered = true;
    std::atexit([]() {
        const Counters& c = C();
        std::cerr << "[LINE-LIFE] kf_calls=" << c.kf_calls
                  << " early_newkf=" << c.kf_early_newkf
                  << " no_neigh=" << c.kf_no_neighbour
                  << " neigh=" << c.neigh_examined
                  << " rej_base=" << c.rej_base_lt_mb
                  << " rej_med=" << c.rej_median_nopts
                  << " rej_base_med=" << c.rej_base_over_med
                  << " pairs=" << c.pair_matched
                  << " pair_empty=" << c.pair_empty
                  << " rej_short=" << c.rej_pair_short
                  << " rej_lenratio=" << c.rej_pair_lenratio
                  << " rej_angle=" << c.rej_pair_angle
                  << " tri=" << c.tri_attempt
                  << " tri_ok=" << c.tri_success
                  << " rej_reproj=" << c.rej_reproj_new
                  << " rej_front=" << c.rej_front
                  << " rej_ratio=" << c.rej_ratio_dist
                  << " ml_created=" << c.ml_created
                  << " ml_atlas=" << c.ml_added_atlas
                  << " shadow_diff=" << c.shadow_old_pass_new_fail
                  << "/" << c.shadow_old_fail_new_pass
                  << " cull_erased=" << c.cull_erased
                  << " atlas_lines=" << c.atlas_lines
                  << " local_lines=" << c.local_lines
                  << " lsd_calls=" << c.lsd_calls
                  << " lsd_raw=" << c.lsd_raw_lines
                  << " lsd_out=" << c.lsd_out_lines << std::endl;
    });
}

} // namespace line_lifecycle
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_LINE_LIFECYCLE_STATS_H
