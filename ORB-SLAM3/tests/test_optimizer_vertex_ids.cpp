// Regression test for the id allocation and endpoint ownership used by the
// bundle adjusters. It calls the SHARED header implementation
// (ORB_SLAM3/include/OptimizerVertexIds.h), i.e. the same code the optimizers
// compile against, not a test-local copy.
#include <cstdio>
#include <vector>

#include "OptimizerVertexIds.h"
#include "g2o/core/optimizable_graph.h"
#include "g2o/core/block_solver.h"
#include "g2o/solvers/linear_solver_dense.h"
#include "g2o/types/types_sba.h"

using namespace ORB_SLAM3;

static int fails = 0;
static void check(const char *name, bool cond, const char *detail = "")
{
    std::printf("  %s  %s  %s\n", cond ? "PASS" : "FAIL", name, detail);
    if (!cond) fails++;
}

static g2o::VertexSBAPointXYZ *mk(int id)
{
    g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
    v->setId(id);
    v->setEstimate(Eigen::Vector3d::Zero());
    return v;
}

int main()
{
    // ---- 1. empty graph -> first free id is 0
    {
        g2o::OptimizableGraph g;
        int first = -999;
        bool ok = optids::NextFreeVertexId(g, 8, first);
        check("empty graph: NextFreeVertexId ok", ok);
        check("empty graph: first id == 0", first == 0);
    }

    // ---- 2. SPARSE MapPoint ids (the old size-based formula was wrong here)
    {
        g2o::OptimizableGraph g;
        const int ids[3] = {1, 100, 5000};
        for (int i = 0; i < 3; ++i) g.addVertex(mk(ids[i]));
        int first = -999;
        optids::NextFreeVertexId(g, 8, first);
        check("sparse ids: first id == 5001 (not size-based 3)", first == 5001);
        char d[64];
        std::snprintf(d, sizeof d, "first=%d (size-based would be %d)", first, 3);
        check("sparse ids: differs from maxKFid+size", first != 3, d);
        g.clear();
    }

    // ---- 3. reserve overflow is refused, not truncated
    {
        g2o::OptimizableGraph g;
        g.addVertex(mk(2147483000));
        int first = -999;
        bool ok = optids::NextFreeVertexId(g, 100000, first);
        check("overflow: refused", !ok);
        g.clear();
    }

    // ---- 4. addVertex on a taken id fails, on a free id succeeds
    {
        g2o::OptimizableGraph g;
        check("addVertex(0) succeeds", g.addVertex(mk(7)));
        g2o::VertexSBAPointXYZ *dup = mk(7);
        bool ok = g.addVertex(dup);
        check("addVertex(duplicate 7) fails", !ok);
        if (!ok) delete dup;               // caller owns it when registration fails
        int first = -999;
        optids::NextFreeVertexId(g, 4, first);
        check("after failure: next free is still 8", first == 8);
        g.clear();
    }

    // ---- 5. first endpoint fails, second succeeds
    {
        g2o::OptimizableGraph g;
        g.addVertex(mk(0));
        g2o::VertexSBAPointXYZ *a = mk(0);   // collides -> fails
        g2o::VertexSBAPointXYZ *b = mk(1);   // ok
        bool ok1 = g.addVertex(a), ok2 = g.addVertex(b);
        check("case first-fail: ok1 == false", !ok1);
        check("case first-fail: ok2 == true", ok2);
        optids::ReleaseEndpointVertices(g, a, ok1, b, ok2);
        // a failed registration is released by us; the registered one is
        // released through the graph, so only the pre-existing vertex remains
        check("case first-fail: graph back to 1 vertex", g.vertices().size() == 1);
        g.clear();
    }

    // ---- 6. both endpoints fail
    {
        g2o::OptimizableGraph g;
        g.addVertex(mk(0));
        g2o::VertexSBAPointXYZ *a = mk(0), *b = mk(0);
        bool ok1 = g.addVertex(a), ok2 = g.addVertex(b);
        check("case both-fail: neither registered", !ok1 && !ok2);
        optids::ReleaseEndpointVertices(g, a, ok1, b, ok2);
        check("case both-fail: graph still holds 1 vertex", g.vertices().size() == 1);
        g.clear();
    }

    // ---- 7. normal registration, explicit early exit, then destructor
    {
        g2o::OptimizableGraph g;
        g.addVertex(mk(0));
        g2o::VertexSBAPointXYZ *a = mk(1), *b = mk(2);
        bool ok1 = g.addVertex(a), ok2 = g.addVertex(b);
        check("case ok: both registered", ok1 && ok2);
        optids::ReleaseEndpointVertices(g, a, ok1, b, ok2);   // early exit path
        check("case ok: graph back to 1 vertex", g.vertices().size() == 1);
        int first = -999;
        optids::NextFreeVertexId(g, 4, first);
        check("case ok: ids 1 and 2 are free again", first == 1);
        g.clear();     // destructor path; ASan would flag any double free
        check("case ok: clear() survived", true);
    }

    std::printf("\n%s\n", fails ? "SOME CHECKS FAILED" : "ALL CHECKS PASSED");
    return fails ? 1 : 0;
}
