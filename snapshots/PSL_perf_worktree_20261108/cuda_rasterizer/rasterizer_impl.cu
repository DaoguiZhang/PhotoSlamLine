/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "rasterizer_impl.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Helper function to find the next-highest bit of the MSB
// on the CPU.
uint32_t getHigherMsb(uint32_t n)
{
	uint32_t msb = sizeof(n) * 4;
	uint32_t step = msb;
	while (step > 1)
	{
		step /= 2;
		if (n >> msb)
			msb += step;
		else
			msb -= step;
	}
	if (n >> msb)
		msb++;
	return msb;
}

// Wrapper method to call auxiliary coarse frustum containment test.
// Mark all Gaussians that pass it.
__global__ void checkFrustum(int P,
	const float* orig_points,
	const float* viewmatrix,
	const float* projmatrix,
	bool* present)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	float3 p_view;
	present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Generates one key/value pair for all Gaussian / tile overlaps. 
// Run once per Gaussian (1:N mapping).
__global__ void duplicateWithKeys(
	int P,
	const float2* points_xy,
	const float* depths,
	const uint32_t* offsets,
	uint64_t* gaussian_keys_unsorted,
	uint32_t* gaussian_values_unsorted,
	int* radii,
	dim3 grid)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Generate no key/value pair for invisible Gaussians
	if (radii[idx] > 0)
	{
		// Find this Gaussian's offset in buffer for writing keys/values.
		uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
		uint2 rect_min, rect_max;

		getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);

		// For each tile that the bounding rect overlaps, emit a 
		// key/value pair. The key is |  tile ID  |      depth      |,
		// and the value is the ID of the Gaussian. Sorting the values 
		// with this key yields Gaussian IDs in a list, such that they
		// are first sorted by tile and then by depth. 
		for (int y = rect_min.y; y < rect_max.y; y++)
		{
			for (int x = rect_min.x; x < rect_max.x; x++)
			{
				uint64_t key = y * grid.x + x;
				key <<= 32;
				key |= *((uint32_t*)&depths[idx]);
				gaussian_keys_unsorted[off] = key;
				gaussian_values_unsorted[off] = idx;
				off++;
			}
		}
	}
}

// Check keys to see if it is at the start/end of one tile's range in 
// the full sorted list. If yes, write start/end of this tile. 
// Run once per instanced (duplicated) Gaussian ID.
__global__ void identifyTileRanges(int L, uint64_t* point_list_keys, uint2* ranges)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= L)
		return;

	// Read tile ID from key. Update start/end of tile range if at limit.
	uint64_t key = point_list_keys[idx];
	uint32_t currtile = key >> 32;
	if (idx == 0)
		ranges[currtile].x = 0;
	else
	{
		uint32_t prevtile = point_list_keys[idx - 1] >> 32;
		if (currtile != prevtile)
		{
			ranges[prevtile].y = idx;
			ranges[currtile].x = idx;
		}
	}
	if (idx == L - 1)
		ranges[currtile].y = L;
}

// Mark Gaussians as visible/invisible, based on view frustum testing
void CudaRasterizer::Rasterizer::markVisible(
	int P,
	float* means3D,
	float* viewmatrix,
	float* projmatrix,
	bool* present)
{
	checkFrustum << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		viewmatrix, projmatrix,
		present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, size_t P)
{
	GeometryState geom;
	obtain(chunk, geom.depths, P, 128);
	obtain(chunk, geom.clamped, P * 3, 128);
	obtain(chunk, geom.internal_radii, P, 128);
	obtain(chunk, geom.means2D, P, 128);
	obtain(chunk, geom.cov3D, P * 6, 128);
	obtain(chunk, geom.conic_opacity, P, 128);
	obtain(chunk, geom.rgb, P * 3, 128);
	obtain(chunk, geom.tiles_touched, P, 128);
	cub::DeviceScan::InclusiveSum(nullptr, geom.scan_size, geom.tiles_touched, geom.tiles_touched, P);
	obtain(chunk, geom.scanning_space, geom.scan_size, 128);
	obtain(chunk, geom.point_offsets, P, 128);
	return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, size_t N)
{
	ImageState img;
	obtain(chunk, img.accum_alpha, N, 128);
	obtain(chunk, img.n_contrib, N, 128);
	obtain(chunk, img.ranges, N, 128);
	return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, size_t P)
{
	BinningState binning;
	obtain(chunk, binning.point_list, P, 128);
	obtain(chunk, binning.point_list_unsorted, P, 128);
	obtain(chunk, binning.point_list_keys, P, 128);
	obtain(chunk, binning.point_list_keys_unsorted, P, 128);
	cub::DeviceRadixSort::SortPairs(
		nullptr, binning.sorting_size,
		binning.point_list_keys_unsorted, binning.point_list_keys,
		binning.point_list_unsorted, binning.point_list, P);
	obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
	return binning;
}

// Forward rendering procedure for differentiable rasterization
// of Gaussians.
// ---------------------------------------------------------------------------
// raster-perf-v1: stage-level rasteriser profile (PHOTO_SLAM_RASTER_PROF=1).
//
// CUDA events are recorded at the rasteriser's OWN stage boundaries, so each
// stage is measured directly instead of being inferred from a whole
// forward/backward span. Stages are disjoint, so they are not double counted.
// Off by default: no events are created and no arithmetic changes.
// ---------------------------------------------------------------------------
struct RasterProf {
	bool on = false;
	int cycles = 0;
	int min_p = 0;            // only profile at/above this Gaussian count
	bool active = false;      // per-cycle gate
	cudaEvent_t ev[10];
	// cumulative, in stage order: preproc, scan, duplicate, sort, tile_ranges,
	// fwd_render, fwd_total, bwd_render, bwd_preproc
	double cum[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	double prev[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
	int prev_cycles = 0;
	double inst = 0.0;
	long long inst_max = 0, inst_last = 0;
	RasterProf() {
		on = (std::getenv("PHOTO_SLAM_RASTER_PROF") != nullptr);
		if (const char* m = std::getenv("PHOTO_SLAM_RASTER_PROF_MIN_P"))
			min_p = std::atoi(m);
		if (on) for (int i = 0; i < 10; ++i) cudaEventCreate(&ev[i]);
		if (on)
			printf("[RasterProf] enabled, min_P=%d (only cycles at/above this "
			       "Gaussian count are profiled)\n", min_p);
	}
	inline bool gate(int P) { active = on && P >= min_p; return active; }
	inline void mark(int i) { if (active) cudaEventRecord(ev[i]); }
	inline void setInst(int n) { inst_last = n; inst += n; if (n > inst_max) inst_max = n; }
	void finish() {
		if (!active) return;
		cudaEventSynchronize(ev[9]);
		auto span = [&](int a, int b) { float v = 0.0f; cudaEventElapsedTime(&v, ev[a], ev[b]); return (double)v; };
		cum[0] += span(0, 1);   // FORWARD::preprocess (projection + SH + cov + bounding)
		cum[1] += span(1, 2);   // cub scan + the host read of num_rendered
		cum[2] += span(2, 3);   // duplicateWithKeys (tile instances)
		cum[3] += span(3, 4);   // DeviceRadixSort
		cum[4] += span(4, 5);   // identifyTileRanges (+ memset)
		cum[5] += span(5, 6);   // FORWARD::render
		cum[6] += span(0, 6);   // forward total
		cum[7] += span(7, 8);   // BACKWARD::render
		cum[8] += span(8, 9);   // BACKWARD::preprocess
		++cycles;
		if (cycles % 500 == 0) {
			// Incremental (independent block) means since the previous report, so
			// the printed series IS the distribution across blocks rather than a
			// single growing average.
			const double n = (double)(cycles - prev_cycles);
			double b[9];
			for (int i = 0; i < 9; ++i) { b[i] = (cum[i] - prev[i]) / n; prev[i] = cum[i]; }
			prev_cycles = cycles;
			printf("[RasterProf] block cycles=%d..%d instances_mean=%.0f instances_max=%lld\n"
			       "    preprocess        %.4f ms\n"
			       "    scan+host_read    %.4f ms\n"
			       "    duplicateWithKeys %.4f ms\n"
			       "    sort              %.4f ms\n"
			       "    identifyRanges    %.4f ms\n"
			       "    FORWARD::render   %.4f ms\n"
			       "    forward_total     %.4f ms\n"
			       "    BACKWARD::render  %.4f ms\n"
			       "    BACKWARD::preproc %.4f ms\n",
			       prev_cycles - (int)n, prev_cycles, inst / (double)cycles, inst_max,
			       b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]);
			fflush(stdout);
		}
	}
};
static RasterProf g_raster_prof;

// ---------------------------------------------------------------------------
// raster-perf-v6: rasteriser CALL COUNTERS (PHOTO_SLAM_RASTER_CALLTRACE=1).
//
// The open question from v5 is "how many rasteriser calls does ONE optimizer
// update cost", because the 5000-step difference (~2.7 ms/step) is an order of
// magnitude larger than one backward kernel launch at the same scale (~0.2 ms).
// This answers it by counting only.
//
// Deliberately counters and nothing else: no CUDA events, no host
// synchronisation, no allocation. Enabling it cannot change the shape of the
// kernel timeline. With the gate off every site is one not-taken branch.
// ---------------------------------------------------------------------------
struct RasterCallTrace {
	bool on = (std::getenv("PHOTO_SLAM_RASTER_CALLTRACE") != nullptr);
	long long fwd_calls = 0;   // Rasterizer::forward invocations
	long long bwd_calls = 0;   // Rasterizer::backward invocations
	long long fwd_render = 0;  // FORWARD::render invocations
	long long bwd_render = 0;  // BACKWARD::render invocations
	int last_P = 0, last_W = 0, last_H = 0, last_D = 0;  // last rasteriser dims
};
RasterCallTrace g_rast_ct;

// Dims of the most recent rasteriser call, so the trace line can state what the
// call counts refer to (Gaussian count, resolution, SH degree).
void RasterCallTraceDims(int* out) { out[0] = g_rast_ct.last_P; out[1] = g_rast_ct.last_W; out[2] = g_rast_ct.last_H; out[3] = g_rast_ct.last_D; }

// Launch counters are free globals so forward.cu / backward.cu (separate
// translation units) can bump them with a plain extern declaration. This is the
// count that exposes EXTRA launches (self-check / golden), which the entry-point
// counters above would hide.
long long g_rast_ct_fwd_launch = 0;
long long g_rast_ct_bwd_launch = 0;
int g_rast_ct_on = 0;

void RasterCallTraceReset()
{
	g_rast_ct.fwd_calls = g_rast_ct.bwd_calls = 0;
	g_rast_ct.fwd_render = g_rast_ct.bwd_render = 0;
	g_rast_ct_fwd_launch = g_rast_ct_bwd_launch = 0;
}

// out[0..5] = fwd_calls, bwd_calls, fwd_render, bwd_render, fwd_launch, bwd_launch
void RasterCallTraceGet(long long* out)
{
	out[0] = g_rast_ct.fwd_calls;
	out[1] = g_rast_ct.bwd_calls;
	out[2] = g_rast_ct.fwd_render;
	out[3] = g_rast_ct.bwd_render;
	out[4] = g_rast_ct_fwd_launch;
	out[5] = g_rast_ct_bwd_launch;
}

int RasterCallTraceOn() { return g_rast_ct.on ? 1 : 0; }


int CudaRasterizer::Rasterizer::forward(
	std::function<char* (size_t)> geometryBuffer,
	std::function<char* (size_t)> binningBuffer,
	std::function<char* (size_t)> imageBuffer,
	const int P, int D, int M,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* opacities,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* cam_pos,
	const float tan_fovx, float tan_fovy,
	const bool prefiltered,
	float* out_color,
	int* radii)
{
	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	g_raster_prof.gate(P);
	g_raster_prof.mark(0);
	if (g_rast_ct.on) {
		++g_rast_ct.fwd_calls;
		g_rast_ct_on = 1;
		g_rast_ct.last_P = P; g_rast_ct.last_W = width; g_rast_ct.last_H = height;
		g_rast_ct.last_D = D;
	}
	size_t chunk_size = required<GeometryState>(P);
	char* chunkptr = geometryBuffer(chunk_size);
	GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Dynamically resize image-based auxiliary buffers during training
	size_t img_chunk_size = required<ImageState>(width * height);
	char* img_chunkptr = imageBuffer(img_chunk_size);
	ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height);

	if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
	{
		throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");
	}

	// Run preprocessing per-Gaussian (transformation, bounding, conversion of SHs to RGB)
	FORWARD::preprocess(
		P, D, M,
		means3D,
		(glm::vec3*)scales,
		scale_modifier,
		(glm::vec4*)rotations,
		opacities,
		shs,
		geomState.clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, projmatrix,
		(glm::vec3*)cam_pos,
		width, height,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		radii,
		geomState.means2D,
		geomState.depths,
		geomState.cov3D,
		geomState.rgb,
		geomState.conic_opacity,
		tile_grid,
		geomState.tiles_touched,
		prefiltered
	);

	g_raster_prof.mark(1);
	// Compute prefix sum over full list of touched tile counts by Gaussians
	// E.g., [2, 3, 0, 2, 1] -> [2, 5, 5, 7, 8]
	cub::DeviceScan::InclusiveSum(geomState.scanning_space, geomState.scan_size,
		geomState.tiles_touched, geomState.point_offsets, P);

	// Retrieve total number of Gaussian instances to launch and resize aux buffers
	int num_rendered;
	cudaMemcpy(&num_rendered, geomState.point_offsets + P - 1, sizeof(int), cudaMemcpyDeviceToHost);

	size_t binning_chunk_size = required<BinningState>(num_rendered);
	char* binning_chunkptr = binningBuffer(binning_chunk_size);
	BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered);

	g_raster_prof.mark(2);
	// For each instance to be rendered, produce adequate [ tile | depth ] key 
	// and corresponding dublicated Gaussian indices to be sorted
	duplicateWithKeys << <(P + 255) / 256, 256 >> > (
		P,
		geomState.means2D,
		geomState.depths,
		geomState.point_offsets,
		binningState.point_list_keys_unsorted,
		binningState.point_list_unsorted,
		radii,
		tile_grid
		);

	g_raster_prof.mark(3);
	int bit = getHigherMsb(tile_grid.x * tile_grid.y);

	// Sort complete list of (duplicated) Gaussian indices by keys
	cub::DeviceRadixSort::SortPairs(
		binningState.list_sorting_space,
		binningState.sorting_size,
		binningState.point_list_keys_unsorted, binningState.point_list_keys,
		binningState.point_list_unsorted, binningState.point_list,
		num_rendered, 0, 32 + bit);

	g_raster_prof.mark(4);
	cudaMemset(imgState.ranges, 0, tile_grid.x * tile_grid.y * sizeof(uint2));

	// Identify start and end of per-tile workloads in sorted list
	if (num_rendered > 0)
		identifyTileRanges << <(num_rendered + 255) / 256, 256 >> > (
			num_rendered,
			binningState.point_list_keys,
			imgState.ranges
			);

	g_raster_prof.mark(5);
	// Let each tile blend its range of Gaussians independently in parallel
	const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
	if (g_rast_ct.on) ++g_rast_ct.fwd_render;
	FORWARD::render(
		tile_grid, block,
		imgState.ranges,
		binningState.point_list,
		width, height,
		geomState.means2D,
		feature_ptr,
		geomState.conic_opacity,
		imgState.accum_alpha,
		imgState.n_contrib,
		background,
		out_color);

	g_raster_prof.mark(6);
	g_raster_prof.setInst(num_rendered);
	return num_rendered;
}

// Produce necessary gradients for optimization, corresponding
// to forward render pass
void CudaRasterizer::Rasterizer::backward(
	const int P, int D, int M, int R,
	const float* background,
	const int width, int height,
	const float* means3D,
	const float* shs,
	const float* colors_precomp,
	const float* scales,
	const float scale_modifier,
	const float* rotations,
	const float* cov3D_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const float* campos,
	const float tan_fovx, float tan_fovy,
	const int* radii,
	char* geom_buffer,
	char* binning_buffer,
	char* img_buffer,
	const float* dL_dpix,
	float* dL_dmean2D,
	float* dL_dconic,
	float* dL_dopacity,
	float* dL_dcolor,
	float* dL_dmean3D,
	float* dL_dcov3D,
	float* dL_dsh,
	float* dL_dscale,
	float* dL_drot)
{
	g_raster_prof.mark(7);
	if (g_rast_ct.on) ++g_rast_ct.bwd_calls;
	GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);
	BinningState binningState = BinningState::fromChunk(binning_buffer, R);
	ImageState imgState = ImageState::fromChunk(img_buffer, width * height);

	if (radii == nullptr)
	{
		radii = geomState.internal_radii;
	}

	const float focal_y = height / (2.0f * tan_fovy);
	const float focal_x = width / (2.0f * tan_fovx);

	const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
	const dim3 block(BLOCK_X, BLOCK_Y, 1);

	// Compute loss gradients w.r.t. 2D mean position, conic matrix,
	// opacity and RGB of Gaussians from per-pixel loss gradients.
	// If we were given precomputed colors and not SHs, use them.
	const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
	if (g_rast_ct.on) ++g_rast_ct.bwd_render;
	BACKWARD::render(
		tile_grid,
		block,
		imgState.ranges,
		binningState.point_list,
		P,
		width, height,
		background,
		geomState.means2D,
		geomState.conic_opacity,
		color_ptr,
		imgState.accum_alpha,
		imgState.n_contrib,
		dL_dpix,
		(float3*)dL_dmean2D,
		(float4*)dL_dconic,
		dL_dopacity,
		dL_dcolor);

	g_raster_prof.mark(8);
	// Take care of the rest of preprocessing. Was the precomputed covariance
	// given to us or a scales/rot pair? If precomputed, pass that. If not,
	// use the one we computed ourselves.
	const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
	BACKWARD::preprocess(P, D, M,
		(float3*)means3D,
		radii,
		shs,
		geomState.clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		cov3D_ptr,
		viewmatrix,
		projmatrix,
		focal_x, focal_y,
		tan_fovx, tan_fovy,
		(glm::vec3*)campos,
		(float3*)dL_dmean2D,
		dL_dconic,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		(glm::vec3*)dL_dscale,
		(glm::vec4*)dL_drot);

	g_raster_prof.mark(9);
	g_raster_prof.finish();
}