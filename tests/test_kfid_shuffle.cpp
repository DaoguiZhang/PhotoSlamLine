// Standalone verification A: keyframe-order shuffle determinism.
// Mirrors the exact algorithm in GaussianMapperLine::generateKfidRandomShuffle():
//   std::iota(0..N-1) then std::shuffle with std::mt19937.
// Explicit-seed path: engine seeded ONCE, reused -> deterministic.
// Default path: fresh engine seeded from random_device per call -> non-deterministic.
// Checksums use the same FNV-1a hash as the [KfShuffleDiag] runtime print so the
// two are directly comparable.
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

static std::uint64_t fnv(std::uint64_t h, std::uint64_t v) {
    h ^= v;
    h *= 0x100000001b3ULL;
    return h;
}

static std::uint64_t hash_of(const std::vector<std::size_t>& v) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t x : v) h = fnv(h, static_cast<std::uint64_t>(x));
    return h;
}

int main() {
    const std::size_t N = 100;
    std::vector<std::size_t> a(N), b(N);

    // --- Explicit-seed path (engine seeded once, reused) ---
    {
        std::mt19937 rng0(static_cast<std::mt19937::result_type>(0));
        std::iota(a.begin(), a.end(), 0);
        std::shuffle(a.begin(), a.end(), rng0);   // 1st shuffle, seed 0

        std::mt19937 rng0b(static_cast<std::mt19937::result_type>(0));
        std::iota(b.begin(), b.end(), 0);
        std::shuffle(b.begin(), b.end(), rng0b);  // independent run, seed 0

        bool same = (a == b);
        std::cout << "[seed=0] same-seed identical = " << (same ? "PASS" : "FAIL")
                  << "  hash=" << std::hex << hash_of(a) << std::dec << std::endl;

        std::mt19937 rng1(static_cast<std::mt19937::result_type>(1));
        std::iota(b.begin(), b.end(), 0);
        std::shuffle(b.begin(), b.end(), rng1);   // seed 1

        bool diff = (a != b);
        std::cout << "[seed=0 vs 1] differ = " << (diff ? "PASS" : "FAIL")
                  << "  hash(1)=" << std::hex << hash_of(b) << std::dec << std::endl;

        // persistent engine: two consecutive shuffles with the SAME engine advance state
        std::mt19937 rng_persist(static_cast<std::mt19937::result_type>(0));
        std::iota(a.begin(), a.end(), 0);
        std::shuffle(a.begin(), a.end(), rng_persist);
        std::iota(b.begin(), b.end(), 0);
        std::shuffle(b.begin(), b.end(), rng_persist);
        std::cout << "[persistent] 2nd shuffle hash=" << std::hex << hash_of(b)
                  << " (differs from 1st=" << hash_of(a) << ")" << std::dec << std::endl;

        return (same && diff) ? 0 : 1;
    }
}
