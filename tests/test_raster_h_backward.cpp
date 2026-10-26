// Small GPU acceptance: H at SH<3 must produce the same backward outputs as an
// explicit C pin, and H at SH=3 the same as an explicit A pin.
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>

#include "cuda_rasterizer/backward.h"

void SetRasterBackend(int b);
void SetRasterSh(int sh);
void RasterBackendInitFromEnv();
void ResetRasterHStats();
void GetRasterHStats(long long& a, long long& c, long long& switches,
                     int& last_b, int& last_sh);

struct Buffers
{
    uint2* ranges;
    uint32_t* points;
    float* bg;
    float2* means;
    float4* conic;
    float* colors;
    float* final_t;
    uint32_t* ncontrib;
    float* dL_pix;
    float3* dmean;
    float4* dconic;
    float* dop;
    float* dcol;
};

static void alloc(Buffers& b)
{
    cudaMalloc(&b.ranges, sizeof(uint2));
    cudaMalloc(&b.points, sizeof(uint32_t));
    cudaMalloc(&b.bg, 3 * sizeof(float));
    cudaMalloc(&b.means, sizeof(float2));
    cudaMalloc(&b.conic, sizeof(float4));
    cudaMalloc(&b.colors, 3 * sizeof(float));
    cudaMalloc(&b.final_t, sizeof(float));
    cudaMalloc(&b.ncontrib, sizeof(uint32_t));
    cudaMalloc(&b.dL_pix, 3 * sizeof(float));
    cudaMalloc(&b.dmean, sizeof(float3));
    cudaMalloc(&b.dconic, sizeof(float4));
    cudaMalloc(&b.dop, sizeof(float));
    cudaMalloc(&b.dcol, 3 * sizeof(float));
}

static void init(Buffers& b)
{
    uint2 h_r{0, 1};
    uint32_t h_p = 0;
    float h_bg[3] = {0.f, 0.f, 0.f};
    float2 h_m{8.f, 8.f};
    float4 h_c{0.02f, 0.f, 0.02f, 0.8f};
    float h_col[3] = {1.f, 0.5f, 0.25f};
    float h_t = 0.2f;
    uint32_t h_n = 1;
    float h_dL[3] = {1.f, -0.5f, 0.25f};
    cudaMemcpy(b.ranges, &h_r, sizeof(h_r), cudaMemcpyHostToDevice);
    cudaMemcpy(b.points, &h_p, sizeof(h_p), cudaMemcpyHostToDevice);
    cudaMemcpy(b.bg, h_bg, sizeof(h_bg), cudaMemcpyHostToDevice);
    cudaMemcpy(b.means, &h_m, sizeof(h_m), cudaMemcpyHostToDevice);
    cudaMemcpy(b.conic, &h_c, sizeof(h_c), cudaMemcpyHostToDevice);
    cudaMemcpy(b.colors, h_col, sizeof(h_col), cudaMemcpyHostToDevice);
    cudaMemcpy(b.final_t, &h_t, sizeof(h_t), cudaMemcpyHostToDevice);
    cudaMemcpy(b.ncontrib, &h_n, sizeof(h_n), cudaMemcpyHostToDevice);
    cudaMemcpy(b.dL_pix, h_dL, sizeof(h_dL), cudaMemcpyHostToDevice);
}

static void clearOut(Buffers& b)
{
    cudaMemset(b.dmean, 0, sizeof(float3));
    cudaMemset(b.dconic, 0, sizeof(float4));
    cudaMemset(b.dop, 0, sizeof(float));
    cudaMemset(b.dcol, 0, 3 * sizeof(float));
}

static void call(Buffers& b)
{
    cudaMemset(b.dmean, 0, sizeof(float3));
    cudaMemset(b.dconic, 0, sizeof(float4));
    cudaMemset(b.dop, 0, sizeof(float));
    cudaMemset(b.dcol, 0, 3 * sizeof(float));
    BACKWARD::render(dim3(1, 1, 1), dim3(16, 16, 1), b.ranges, b.points, 1,
                     16, 16, b.bg, b.means, b.conic, b.colors, b.final_t,
                     b.ncontrib, b.dL_pix, b.dmean, b.dconic, b.dop, b.dcol);
    cudaDeviceSynchronize();
}

static bool same(Buffers& a, Buffers& b)
{
    float3 ma, mb;
    float4 ca, cb;
    float oa, ob;
    float coa[3], cob[3];
    cudaMemcpy(&ma, a.dmean, sizeof(ma), cudaMemcpyDeviceToHost);
    cudaMemcpy(&mb, b.dmean, sizeof(mb), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ca, a.dconic, sizeof(ca), cudaMemcpyDeviceToHost);
    cudaMemcpy(&cb, b.dconic, sizeof(cb), cudaMemcpyDeviceToHost);
    cudaMemcpy(&oa, a.dop, sizeof(oa), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ob, b.dop, sizeof(ob), cudaMemcpyDeviceToHost);
    cudaMemcpy(coa, a.dcol, sizeof(coa), cudaMemcpyDeviceToHost);
    cudaMemcpy(cob, b.dcol, sizeof(cob), cudaMemcpyDeviceToHost);
    return std::memcmp(&ma, &mb, sizeof(ma)) == 0 &&
           std::memcmp(&ca, &cb, sizeof(ca)) == 0 &&
           std::memcmp(&oa, &ob, sizeof(oa)) == 0 &&
           std::memcmp(coa, cob, sizeof(coa)) == 0;
}

int main()
{
    setenv("PHOTO_SLAM_RASTER_BACKEND", "H", 1);
    RasterBackendInitFromEnv();
    ResetRasterHStats();
    Buffers h, c, a;
    alloc(h); alloc(c); alloc(a);
    init(h); init(c); init(a);
    bool ok = true;

    SetRasterBackend(3);
    SetRasterSh(0);
    call(h);
    SetRasterBackend(2);
    call(c);
    if (!same(h, c)) { std::cout << "[RasterHBackward] SH0 H != C\n"; ok = false; }

    SetRasterBackend(3);
    SetRasterSh(3);
    call(h);
    SetRasterBackend(0);
    call(a);
    if (!same(h, a)) { std::cout << "[RasterHBackward] SH3 H != A\n"; ok = false; }

    long long ha = 0, hc = 0, sw = 0;
    int lb = -1, ls = -1;
    GetRasterHStats(ha, hc, sw, lb, ls);
    if (ha != 1 || hc != 1) ok = false;
    std::cout << "[RasterHBackward] H A=" << ha << " C=" << hc
              << " switches=" << sw << " " << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
