// Host-side acceptance for the H stage rule.  No GPU launch is performed here;
// the full H online run checks actual backward counts.  This test checks that
// the rule is resolved from the CPU SH value and that no static A/C cache
// prevents the switch.
#include <cstdlib>
#include <iostream>

void SetRasterBackend(int b);
void SetRasterSh(int sh);
void RasterBackendInitFromEnv();
int RasterBackendEffective();
int RasterBackendGet();
void ResetRasterHStats();
void GetRasterHStats(long long& a, long long& c, long long& switches,
                     int& last_b, int& last_sh);

int main()
{
    setenv("PHOTO_SLAM_RASTER_BACKEND", "H", 1);
    RasterBackendInitFromEnv();
    ResetRasterHStats();

    bool ok = (RasterBackendGet() == 3);
    auto check = [&](int sh, int expected) {
        SetRasterSh(sh);
        const int got = RasterBackendEffective();
        std::cout << "[RasterHDispatch] sh=" << sh << " backend="
                  << (got == 0 ? "A" : (got == 2 ? "C" : "?"))
                  << " expected=" << (expected == 0 ? "A" : "C") << '\n';
        if (got != expected) ok = false;
    };
    check(0, 2);
    check(1, 2);
    check(2, 2);
    check(3, 0);

    // Explicit A/C pins must remain exact (default production paths).
    SetRasterBackend(0);
    SetRasterSh(3);
    if (RasterBackendEffective() != 0) ok = false;
    SetRasterBackend(2);
    SetRasterSh(0);
    if (RasterBackendEffective() != 2) ok = false;

    long long a = -1, c = -1, sw = -1;
    int lb = -1, lsh = -1;
    GetRasterHStats(a, c, sw, lb, lsh);
    if (a != 0 || c != 0) ok = false;

    std::cout << "[RasterHDispatch] " << (ok ? "PASS" : "FAIL") << '\n';
    return ok ? 0 : 1;
}
