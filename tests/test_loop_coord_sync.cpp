// Standalone runner for the CPU rigid/sim(3) correction and LOOP payload-key
// contracts. No dataset, CUDA context, or SLAM startup needed.
#include <cstdio>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "include/loop_event_key.h"

bool runRigidCorrectionContractTest();

static bool runLoopKeyContractTest()
{
    Eigen::Matrix4f A = Eigen::Matrix4f::Identity();
    A(0, 3) = 0.1f;
    Eigen::Matrix4f B = Eigen::Matrix4f::Identity();
    B(1, 3) = 0.2f;

    std::vector<std::pair<unsigned long, Eigen::Matrix4f>> k1{
        {10u, A}, {20u, B}, {30u, A}};
    std::vector<std::pair<unsigned long, Eigen::Matrix4f>> k2{
        {30u, A}, {10u, A}, {20u, B}};

    const std::uint64_t a = loop_event_key::fingerprint(0u, 0u, 1.0f, k1);
    const std::uint64_t b = loop_event_key::fingerprint(0u, 0u, 1.0f, k2);
    const std::uint64_t c = loop_event_key::fingerprint(0u, 1u, 1.0f, k1);
    const bool order_ok = (a == b);
    const bool map_ok = (a != c);
    const bool match_ok = loop_event_key::map_matches(0u, 7u) &&
                          !loop_event_key::map_matches(1u, 7u) &&
                          loop_event_key::map_matches(7u, 7u);
    std::printf("[LoopKeyContract] order=%d map=%d match=%d : %s\n",
                order_ok ? 1 : 0, map_ok ? 1 : 0, match_ok ? 1 : 0,
                (order_ok && map_ok && match_ok) ? "PASS" : "FAIL");
    return order_ok && map_ok && match_ok;
}

int main()
{
    const bool pass = runRigidCorrectionContractTest() &&
                      runLoopKeyContractTest();
    std::printf("[test_loop_coord_sync] %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
