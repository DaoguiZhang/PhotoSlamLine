/**
 * Vertex-id allocation shared by every reachable bundle adjuster.
 *
 * Why this exists
 * ---------------
 * The line endpoints of a MapLine are added to a g2o graph as extra vertices.
 * Their ids used to be derived from CONTAINER SIZES:
 *
 *     int nextVertexId = (int)maxKFid + vpMPs.size() + 2;
 *
 * MapPoint ids are sparse (mnId keeps increasing while size() counts the
 * survivors), so that expression can land on an id that is already registered.
 * addVertex then fails, the code takes the "registration failed" early-exit
 * branch, and the endpoint vertices are cleaned up. Deriving the id from the
 * largest id the graph actually holds removes the collision, and therefore also
 * removes that whole failure path in the common case.
 *
 * It lives in a header so the regression test exercises the SAME code the
 * optimizers use, instead of a copy that could drift.
 *
 * Nothing here changes residuals, weights, thresholds, iteration counts or the
 * map-merge policy.
 */
#ifndef ORB_SLAM3_OPTIMIZER_VERTEX_IDS_H
#define ORB_SLAM3_OPTIMIZER_VERTEX_IDS_H

#include <climits>
#include <iostream>

#include "g2o/core/optimizable_graph.h"

namespace ORB_SLAM3
{
namespace optids
{

// Largest id currently registered in the graph; -1 when the graph is empty.
inline long long MaxRegisteredVertexId(const g2o::OptimizableGraph &opt)
{
    long long mx = -1;
    const g2o::HyperGraph::VertexIDMap &vs = opt.vertices();
    for (g2o::HyperGraph::VertexIDMap::const_iterator it = vs.begin();
         it != vs.end(); ++it)
    {
        const long long id = (long long)it->first;
        if (id > mx) mx = id;
    }
    return mx;
}

// First id that is free for a block of `reserve` consecutive ids.
// All arithmetic is done in long long so that neither the size conversion nor
// the addition can overflow before the range check.
// Returns false (and does not touch out_first) if the block would not fit into
// the int id space used by g2o.
inline bool NextFreeVertexId(const g2o::OptimizableGraph &opt, long long reserve,
                             int &out_first)
{
    if (reserve < 0) return false;
    const long long first = MaxRegisteredVertexId(opt) + 1;
    if (first + reserve > (long long)INT_MAX) return false;
    out_first = (int)first;
    return true;
}

/* Ownership contract of the block that adds line endpoints.
 *
 *   bool ok = graph.addVertex(v);
 *
 *   ok == true   -> the graph owns v. It must be released through the graph
 *                   (removeVertex, which DELETES it) and must never be deleted
 *                   by the caller afterwards.
 *   ok == false  -> the graph did not take it. The caller still owns v and must
 *                   delete it exactly once.
 *
 * The helper below is the single place that implements that rule so the two
 * failure combinations cannot drift apart again.
 */
inline void ReleaseEndpointVertices(g2o::OptimizableGraph &graph,
                                    g2o::HyperGraph::Vertex *v1, bool ok1,
                                    g2o::HyperGraph::Vertex *v2, bool ok2)
{
    if (v1)
    {
        if (ok1) graph.removeVertex(v1);  // graph deletes it
        else delete v1;                   // never registered: caller owns it
    }
    if (v2)
    {
        if (ok2) graph.removeVertex(v2);
        else delete v2;
    }
}

}  // namespace optids
}  // namespace ORB_SLAM3

#endif  // ORB_SLAM3_OPTIMIZER_VERTEX_IDS_H
