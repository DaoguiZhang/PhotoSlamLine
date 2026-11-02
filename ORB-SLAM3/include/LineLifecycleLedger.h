/**
* MapLine lifecycle ledger (diagnostic only).
*
* GATE: PHOTO_SLAM_LINE_LEDGER_CSV=<path>     (unset => no allocation, no I/O)
*
* WHY A SECOND COUNTER FILE
* -------------------------
* LineLifecycleStats.h counts STAGES inside CreateNewMapLines (candidate pairs,
* triangulation attempts, reprojection rejects). It answers "where do candidates
* die". It cannot answer "what happened to THIS MapLine object", because a line
* can die long after creation, in a different thread, via observation erasure or
* fusion. Inferring deletions from (created - alive) is not acceptable: the same
* object can be removed more than once and different causes can interleave.
*
* This ledger keys every event by the unique identity (mapId, lineId) and keeps
* a state machine per object, so:
*   - call counts    = how many times an API was invoked
*   - unique counts  = how many DISTINCT objects reached that state
*   - double events  = repeated removal of an already-removed object
* and every map gets a reconciliation row.
*
* States: 0 = created, 1 = in map (added), 2 = bad (SetBadFlag / Replace),
*         3 = erased from map. Only a 0->1, 1->2, 2->3 progression is counted as
* a unique transition; repeats are counted separately as `dup_*`.
*/

#ifndef ORB_SLAM3_LINE_LIFECYCLE_LEDGER_H
#define ORB_SLAM3_LINE_LIFECYCLE_LEDGER_H

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <string>
#include <utility>

namespace ORB_SLAM3
{
namespace line_ledger
{

enum State { ST_CREATED = 0, ST_IN_MAP = 1, ST_BAD = 2, ST_ERASED = 3 };

struct MapTally
{
    long long created        = 0;   // unique objects constructed in this map
    long long added          = 0;   // unique objects entering the map set
    long long transferred_in = 0;
    long long transferred_out= 0;
    long long setbad         = 0;   // unique objects flagged bad in this map
    long long erased         = 0;   // unique objects erased from this map set
    long long final_size     = 0;   // atlas/map set size at shutdown
};

struct Ledger
{
    std::mutex mu;
    std::map<std::pair<long long,long long>, int> state;  // (mapId, lineId) -> State
    std::map<long long, MapTally> perMap;

    // --- call-level (every invocation) ---------------------------------
    long long create_calls     = 0;
    long long atlas_add_calls  = 0;
    long long obs_add_calls    = 0;
    long long obs_erase_calls  = 0;
    long long obs_erase_to_bad = 0;   // erasures that pushed nObs <= 2
    long long setbad_calls     = 0;
    long long erase_calls      = 0;
    long long replace_calls    = 0;
    long long updatemap_calls  = 0;

    // --- unique-object (distinct identities) ---------------------------
    long long create_unique    = 0;
    long long atlas_add_unique = 0;
    long long setbad_unique    = 0;
    long long erase_unique     = 0;
    long long replace_unique   = 0;

    // --- anomalies ------------------------------------------------------
    long long dup_create       = 0;   // same id constructed twice (id reuse)
    long long dup_setbad       = 0;   // SetBadFlag on an already-bad object
    long long dup_erase        = 0;   // erased again after already erased
    long long erase_unknown    = 0;   // erased without a ledger entry
    long long bad_never_added  = 0;   // went bad without ever entering a map
};

inline bool Enabled()
{
    static const char* p = std::getenv("PHOTO_SLAM_LINE_LEDGER_CSV");
    return p != nullptr && p[0] != '\0';
}

inline Ledger& L()
{
    static Ledger* s = new Ledger();   // leaked on purpose (atexit use)
    return *s;
}

inline std::ofstream& Sink()
{
    static std::ofstream* s = nullptr;
    if (s == nullptr)
    {
        s = new std::ofstream(std::getenv("PHOTO_SLAM_LINE_LEDGER_CSV"),
                              std::ios::out | std::ios::trunc);
        if (s->good())
        {
            (*s) << "# MapLine lifecycle ledger. One row per state transition.\n";
            (*s) << "# dup_* rows are repeated events on an already-transitioned"
                    " object: they are NOT new deletions.\n";
            (*s) << "event,map_id,line_id,extra,state_before,state_after\n";
            s->flush();
        }
    }
    return *s;
}

inline void Row(const char* ev, long long mapId, long long lineId,
                long long extra, int before, int after)
{
    if (!Enabled()) return;
    std::ofstream& o = Sink();
    if (!o.good()) return;
    o << ev << ',' << mapId << ',' << lineId << ',' << extra << ','
      << before << ',' << after << '\n';
}

// ---- hooks ------------------------------------------------------------

inline void OnCreate(long long mapId, long long lineId)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.create_calls;
    const auto key = std::make_pair(mapId, lineId);
    auto it = l.state.find(key);
    if (it != l.state.end())
    {
        ++l.dup_create;
        Row("DUP_CREATE", mapId, lineId, 0, it->second, it->second);
        return;
    }
    l.state[key] = ST_CREATED;
    ++l.create_unique;
    ++l.perMap[mapId].created;
    Row("CREATE", mapId, lineId, 0, -1, ST_CREATED);
}

inline void OnAtlasAdd(long long mapId, long long lineId)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.atlas_add_calls;
    const auto key = std::make_pair(mapId, lineId);
    auto it = l.state.find(key);
    const int before = (it == l.state.end()) ? -1 : it->second;
    if (before == ST_IN_MAP) { Row("ADD_AGAIN", mapId, lineId, 0, before, before); return; }
    if (it == l.state.end()) l.state[key] = ST_CREATED;   // added without CREATE hook
    l.state[key] = ST_IN_MAP;
    ++l.atlas_add_unique;
    ++l.perMap[mapId].added;
    Row("ATLAS_ADD", mapId, lineId, 0, before, ST_IN_MAP);
}

inline void OnObsAdd(long long mapId, long long lineId)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.obs_add_calls;
    Row("OBS_ADD", mapId, lineId, 0, -1, -1);
}

inline void OnObsErase(long long mapId, long long lineId, int nObsLeft, bool pushedBad)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.obs_erase_calls;
    if (pushedBad) ++l.obs_erase_to_bad;
    Row("OBS_ERASE", mapId, lineId, nObsLeft, -1, -1);
}

inline void OnSetBad(long long mapId, long long lineId)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.setbad_calls;
    const auto key = std::make_pair(mapId, lineId);
    auto it = l.state.find(key);
    const int before = (it == l.state.end()) ? -1 : it->second;
    if (before == ST_BAD || before == ST_ERASED)
    {
        ++l.dup_setbad;
        Row("DUP_SETBAD", mapId, lineId, 0, before, before);
        return;
    }
    if (before == ST_CREATED) ++l.bad_never_added;
    l.state[key] = ST_BAD;
    ++l.setbad_unique;
    ++l.perMap[mapId].setbad;
    Row("SET_BAD", mapId, lineId, 0, before, ST_BAD);
}

inline void OnErase(long long mapId, long long lineId)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.erase_calls;
    const auto key = std::make_pair(mapId, lineId);
    auto it = l.state.find(key);
    if (it == l.state.end())
    {
        ++l.erase_unknown;
        Row("ERASE_UNKNOWN", mapId, lineId, 0, -1, ST_ERASED);
        return;
    }
    const int before = it->second;
    if (before == ST_ERASED)
    {
        ++l.dup_erase;
        Row("DUP_ERASE", mapId, lineId, 0, before, before);
        return;
    }
    it->second = ST_ERASED;
    ++l.erase_unique;
    ++l.perMap[mapId].erased;
    Row("ERASE", mapId, lineId, 0, before, ST_ERASED);
}

inline void OnReplace(long long fromMap, long long fromId,
                      long long toMap, long long toId)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.replace_calls;
    const auto key = std::make_pair(fromMap, fromId);
    auto it = l.state.find(key);
    const int before = (it == l.state.end()) ? -1 : it->second;
    if (before == ST_BAD || before == ST_ERASED)
    {
        Row("DUP_REPLACE", fromMap, fromId, toId, before, before);
        return;
    }
    l.state[key] = ST_BAD;
    ++l.replace_unique;
    ++l.perMap[fromMap].setbad;
    Row("REPLACE", fromMap, fromId, toId, before, ST_BAD);
}

// Map transfer. Not a deletion: the object keeps living in another map.
inline void OnUpdateMap(long long oldMap, long long newMap, long long lineId)
{
    if (!Enabled()) return;
    Ledger& l = L();
    std::lock_guard<std::mutex> g(l.mu);
    ++l.updatemap_calls;
    if (oldMap == newMap) return;
    const auto oldKey = std::make_pair(oldMap, lineId);
    auto it = l.state.find(oldKey);
    if (it == l.state.end()) return;
    const int st = it->second;
    it->second = ST_ERASED;                 // leaves the old map's set
    l.state[std::make_pair(newMap, lineId)] = (st == ST_ERASED) ? ST_CREATED : st;
    ++l.perMap[oldMap].transferred_out;
    ++l.perMap[newMap].transferred_in;
    Row("MAP_TRANSFER", oldMap, lineId, newMap, st, st);
}

// ---- shutdown reconciliation -----------------------------------------

inline void RegisterExitSummary()
{
    static bool sReg = false;
    if (sReg || !Enabled()) return;
    sReg = true;
    std::atexit([]() {
        Ledger& l = L();
        std::lock_guard<std::mutex> g(l.mu);
        std::ofstream sum(std::string(std::getenv("PHOTO_SLAM_LINE_LEDGER_CSV"))
                          + ".maps.csv", std::ios::out | std::ios::trunc);
        sum << "# per-map reconciliation (unique objects)\n";
        sum << "map_id,created,added,transferred_in,transferred_out,setbad,erased,"
               "surviving_unique\n";
        long long totCreate = 0, totErased = 0;
        for (auto& kv : l.perMap)
        {
            const MapTally& t = kv.second;
            // unique objects created in this map that have not been erased
            long long surviving = t.created + t.transferred_in - t.erased
                                  - t.transferred_out;
            sum << kv.first << ',' << t.created << ',' << t.added << ','
                << t.transferred_in << ',' << t.transferred_out << ','
                << t.setbad << ',' << t.erased << ',' << surviving << '\n';
            totCreate += t.created;
            totErased += t.erased;
        }
        sum.flush();

        std::cerr << "[LINE-LEDGER]"
                  << " create_calls=" << l.create_calls
                  << " create_unique=" << l.create_unique
                  << " atlas_add_calls=" << l.atlas_add_calls
                  << " atlas_add_unique=" << l.atlas_add_unique
                  << " obs_add=" << l.obs_add_calls
                  << " obs_erase=" << l.obs_erase_calls
                  << " obs_erase_to_bad=" << l.obs_erase_to_bad
                  << " setbad_calls=" << l.setbad_calls
                  << " setbad_unique=" << l.setbad_unique
                  << " erase_calls=" << l.erase_calls
                  << " erase_unique=" << l.erase_unique
                  << " replace_calls=" << l.replace_calls
                  << " replace_unique=" << l.replace_unique
                  << " updatemap=" << l.updatemap_calls
                  << " dup_setbad=" << l.dup_setbad
                  << " dup_erase=" << l.dup_erase
                  << " erase_unknown=" << l.erase_unknown
                  << " bad_never_added=" << l.bad_never_added
                  << " total_created=" << totCreate
                  << " total_erased=" << totErased << std::endl;
    });
}

inline void Touch()
{
    if (Enabled()) RegisterExitSummary();
}

} // namespace line_ledger
} // namespace ORB_SLAM3

#endif // ORB_SLAM3_LINE_LIFECYCLE_LEDGER_H
