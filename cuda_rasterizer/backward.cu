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

#include "backward.h"
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}

// Backward version of INVERSE 2D covariance matrix computation
// (due to length launched as separate kernel before other 
// backward steps contained in preprocess)
__global__ void computeCov2DCUDA(int P,
	const float3* means,
	const int* radii,
	const float* cov3Ds,
	const float h_x, float h_y,
	const float tan_fovx, float tan_fovy,
	const float* view_matrix,
	const float* dL_dconics,
	float3* dL_dmeans,
	float* dL_dcov)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	// Reading location of 3D covariance for this Gaussian
	const float* cov3D = cov3Ds + 6 * idx;

	// Fetch gradients, recompute 2D covariance and relevant 
	// intermediate forward results needed in the backward.
	float3 mean = means[idx];
	float3 dL_dconic = { dL_dconics[4 * idx], dL_dconics[4 * idx + 1], dL_dconics[4 * idx + 3] };
	float3 t = transformPoint4x3(mean, view_matrix);
	
	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;
	
	const float x_grad_mul = txtz < -limx || txtz > limx ? 0 : 1;
	const float y_grad_mul = tytz < -limy || tytz > limy ? 0 : 1;

	glm::mat3 J = glm::mat3(h_x / t.z, 0.0f, -(h_x * t.x) / (t.z * t.z),
		0.0f, h_y / t.z, -(h_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view_matrix[0], view_matrix[4], view_matrix[8],
		view_matrix[1], view_matrix[5], view_matrix[9],
		view_matrix[2], view_matrix[6], view_matrix[10]);

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 T = W * J;

	glm::mat3 cov2D = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Use helper variables for 2D covariance entries. More compact.
	float a = cov2D[0][0] += 0.3f;
	float b = cov2D[0][1];
	float c = cov2D[1][1] += 0.3f;

	float denom = a * c - b * b;
	float dL_da = 0, dL_db = 0, dL_dc = 0;
	float denom2inv = 1.0f / ((denom * denom) + 0.0000001f);

	if (denom2inv != 0)
	{
		// Gradients of loss w.r.t. entries of 2D covariance matrix,
		// given gradients of loss w.r.t. conic matrix (inverse covariance matrix).
		// e.g., dL / da = dL / d_conic_a * d_conic_a / d_a
		dL_da = denom2inv * (-c * c * dL_dconic.x + 2 * b * c * dL_dconic.y + (denom - a * c) * dL_dconic.z);
		dL_dc = denom2inv * (-a * a * dL_dconic.z + 2 * a * b * dL_dconic.y + (denom - a * c) * dL_dconic.x);
		dL_db = denom2inv * 2 * (b * c * dL_dconic.x - (denom + 2 * b * b) * dL_dconic.y + a * b * dL_dconic.z);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (diagonal).
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[1][0] * dL_db + T[1][0] * T[1][0] * dL_dc);
		dL_dcov[6 * idx + 3] = (T[0][1] * T[0][1] * dL_da + T[0][1] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
		dL_dcov[6 * idx + 5] = (T[0][2] * T[0][2] * dL_da + T[0][2] * T[1][2] * dL_db + T[1][2] * T[1][2] * dL_dc);

		// Gradients of loss L w.r.t. each 3D covariance matrix (Vrk) entry, 
		// given gradients w.r.t. 2D covariance matrix (off-diagonal).
		// Off-diagonal elements appear twice --> double the gradient.
		// cov2D = transpose(T) * transpose(Vrk) * T;
		dL_dcov[6 * idx + 1] = 2 * T[0][0] * T[0][1] * dL_da + (T[0][0] * T[1][1] + T[0][1] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][1] * dL_dc;
		dL_dcov[6 * idx + 2] = 2 * T[0][0] * T[0][2] * dL_da + (T[0][0] * T[1][2] + T[0][2] * T[1][0]) * dL_db + 2 * T[1][0] * T[1][2] * dL_dc;
		dL_dcov[6 * idx + 4] = 2 * T[0][2] * T[0][1] * dL_da + (T[0][1] * T[1][2] + T[0][2] * T[1][1]) * dL_db + 2 * T[1][1] * T[1][2] * dL_dc;
	}
	else
	{
		for (int i = 0; i < 6; i++)
			dL_dcov[6 * idx + i] = 0;
	}

	// Gradients of loss w.r.t. upper 2x3 portion of intermediate matrix T
	// cov2D = transpose(T) * transpose(Vrk) * T;
	float dL_dT00 = 2 * (T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_da +
		(T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_db;
	float dL_dT01 = 2 * (T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_da +
		(T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_db;
	float dL_dT02 = 2 * (T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_da +
		(T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_db;
	float dL_dT10 = 2 * (T[1][0] * Vrk[0][0] + T[1][1] * Vrk[0][1] + T[1][2] * Vrk[0][2]) * dL_dc +
		(T[0][0] * Vrk[0][0] + T[0][1] * Vrk[0][1] + T[0][2] * Vrk[0][2]) * dL_db;
	float dL_dT11 = 2 * (T[1][0] * Vrk[1][0] + T[1][1] * Vrk[1][1] + T[1][2] * Vrk[1][2]) * dL_dc +
		(T[0][0] * Vrk[1][0] + T[0][1] * Vrk[1][1] + T[0][2] * Vrk[1][2]) * dL_db;
	float dL_dT12 = 2 * (T[1][0] * Vrk[2][0] + T[1][1] * Vrk[2][1] + T[1][2] * Vrk[2][2]) * dL_dc +
		(T[0][0] * Vrk[2][0] + T[0][1] * Vrk[2][1] + T[0][2] * Vrk[2][2]) * dL_db;

	// Gradients of loss w.r.t. upper 3x2 non-zero entries of Jacobian matrix
	// T = W * J
	float dL_dJ00 = W[0][0] * dL_dT00 + W[0][1] * dL_dT01 + W[0][2] * dL_dT02;
	float dL_dJ02 = W[2][0] * dL_dT00 + W[2][1] * dL_dT01 + W[2][2] * dL_dT02;
	float dL_dJ11 = W[1][0] * dL_dT10 + W[1][1] * dL_dT11 + W[1][2] * dL_dT12;
	float dL_dJ12 = W[2][0] * dL_dT10 + W[2][1] * dL_dT11 + W[2][2] * dL_dT12;

	float tz = 1.f / t.z;
	float tz2 = tz * tz;
	float tz3 = tz2 * tz;

	// Gradients of loss w.r.t. transformed Gaussian mean t
	float dL_dtx = x_grad_mul * -h_x * tz2 * dL_dJ02;
	float dL_dty = y_grad_mul * -h_y * tz2 * dL_dJ12;
	float dL_dtz = -h_x * tz2 * dL_dJ00 - h_y * tz2 * dL_dJ11 + (2 * h_x * t.x) * tz3 * dL_dJ02 + (2 * h_y * t.y) * tz3 * dL_dJ12;

	// Account for transformation of mean to t
	// t = transformPoint4x3(mean, view_matrix);
	float3 dL_dmean = transformVec4x3Transpose({ dL_dtx, dL_dty, dL_dtz }, view_matrix);

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the covariance matrix.
	// Additional mean gradient is accumulated in BACKWARD::preprocess.
	dL_dmeans[idx] = dL_dmean;
}

// Backward pass for the conversion of scale and rotation to a 
// 3D covariance matrix for each Gaussian. 
__device__ void computeCov3D(int idx, const glm::vec3 scale, float mod, const glm::vec4 rot, const float* dL_dcov3Ds, glm::vec3* dL_dscales, glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 S = glm::mat3(1.0f);

	glm::vec3 s = mod * scale;
	S[0][0] = s.x;
	S[1][1] = s.y;
	S[2][2] = s.z;

	glm::mat3 M = S * R;

	const float* dL_dcov3D = dL_dcov3Ds + 6 * idx;

	glm::vec3 dunc(dL_dcov3D[0], dL_dcov3D[3], dL_dcov3D[5]);
	glm::vec3 ounc = 0.5f * glm::vec3(dL_dcov3D[1], dL_dcov3D[2], dL_dcov3D[4]);

	// Convert per-element covariance loss gradients to matrix form
	glm::mat3 dL_dSigma = glm::mat3(
		dL_dcov3D[0], 0.5f * dL_dcov3D[1], 0.5f * dL_dcov3D[2],
		0.5f * dL_dcov3D[1], dL_dcov3D[3], 0.5f * dL_dcov3D[4],
		0.5f * dL_dcov3D[2], 0.5f * dL_dcov3D[4], dL_dcov3D[5]
	);

	// Compute loss gradient w.r.t. matrix M
	// dSigma_dM = 2 * M
	glm::mat3 dL_dM = 2.0f * M * dL_dSigma;

	glm::mat3 Rt = glm::transpose(R);
	glm::mat3 dL_dMt = glm::transpose(dL_dM);

	// Gradients of loss w.r.t. scale
	glm::vec3* dL_dscale = dL_dscales + idx;
	dL_dscale->x = glm::dot(Rt[0], dL_dMt[0]);
	dL_dscale->y = glm::dot(Rt[1], dL_dMt[1]);
	dL_dscale->z = glm::dot(Rt[2], dL_dMt[2]);

	dL_dMt[0] *= s.x;
	dL_dMt[1] *= s.y;
	dL_dMt[2] *= s.z;

	// Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	*dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };//dnormvdv(float4{ rot.x, rot.y, rot.z, rot.w }, float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w });
}

// Backward pass of the preprocessing steps, except
// for the covariance computation and inversion
// (those are handled by a previous kernel call)
template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* proj,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	glm::vec3* dL_dmeans,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	float3 m = means[idx];

	// Taking care of gradients from the screenspace points
	float4 m_hom = transformPoint4x4(m, proj);
	float m_w = 1.0f / (m_hom.w + 0.0000001f);

	// Compute loss gradient w.r.t. 3D means due to gradients of 2D means
	// from rendering procedure
	glm::vec3 dL_dmean;
	float mul1 = (proj[0] * m.x + proj[4] * m.y + proj[8] * m.z + proj[12]) * m_w * m_w;
	float mul2 = (proj[1] * m.x + proj[5] * m.y + proj[9] * m.z + proj[13]) * m_w * m_w;
	dL_dmean.x = (proj[0] * m_w - proj[3] * mul1) * dL_dmean2D[idx].x + (proj[1] * m_w - proj[3] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.y = (proj[4] * m_w - proj[7] * mul1) * dL_dmean2D[idx].x + (proj[5] * m_w - proj[7] * mul2) * dL_dmean2D[idx].y;
	dL_dmean.z = (proj[8] * m_w - proj[11] * mul1) * dL_dmean2D[idx].x + (proj[9] * m_w - proj[11] * mul2) * dL_dmean2D[idx].y;

	// That's the second part of the mean gradient. Previous computation
	// of cov2D and following SH conversion also affects it.
	dL_dmeans[idx] += dL_dmean;

	// Compute gradient updates due to computing colors from SHs
	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means, *campos, shs, clamped, (glm::vec3*)dL_dcolor, (glm::vec3*)dL_dmeans, (glm::vec3*)dL_dsh);

	// Compute gradient updates due to computing covariance from scale/rotation
	if (scales)
		computeCov3D(idx, scales[idx], scale_modifier, rotations[idx], dL_dcov3D, dL_dscale, dL_drot);
}

// ---------------------------------------------------------------------------
// raster-perf-v3 item 一: standalone test of __fns, with NO Gaussian rendering
// involved. Settles the documented PTX semantics for the cases that matter here:
//   * offset = 0
//   * masks that do / do not contain lane 0
//   * the highest bit, a single lane, alternating bits, irregular masks
//   * the invalid return value
// and compares the ORIGINAL enumeration call __fns(mask,0,i+1) against the
// modified one __fns(mask,0,i), so the report's explanation is corrected by
// measurement instead of by reading.
//
// __fns(mask, base, offset) returns the bit position of the (offset)-th set bit
// with position STRICTLY GREATER THAN base; i.e. the search starts after `base`
// and offset is 0-based from there. It returns 0xFFFFFFFF when no such bit
// exists. Only one lane of the warp evaluates every case, so the printed table
// is a per-case answer rather than a per-lane answer.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned long long fns_tbl_col(const unsigned mask, const int offset)
{
	// Encode "invalid" distinctly from a valid lane id.
	const unsigned long long r = static_cast<unsigned long long>(__fns(mask, 0u, offset));
	return (r == 0xFFFFFFFFull) ? 0xFFFFFFFFFFFFFFFFull : r;
}

__global__ void fnsSelfTestKernel(unsigned long long* out /* [6][4] */)
{
	if (threadIdx.x != 0) return;
	const unsigned cases[6] = {
		0xFFFFFFFFu,  // full warp, contains lane 0
		0xFFFFFFFEu,  // lane 0 cleared; lowest set bit is lane 1
		0x80000000u,  // only the highest bit
		0x00000001u,  // only lane 0
		0xAAAAAAAAu,  // alternating, starts at lane 1
		0x00050005u,  // irregular: lanes 0, 2, 16, 18
	};
	for (int c = 0; c < 6; ++c)
	{
		out[c * 4 + 0] = static_cast<unsigned long long>(cases[c]);
		out[c * 4 + 1] = fns_tbl_col(cases[c], 0);   // offset 0
		out[c * 4 + 2] = fns_tbl_col(cases[c], 1);   // offset 1 (old i+1 at i=0)
		out[c * 4 + 3] = fns_tbl_col(cases[c], 2);   // offset 2
	}
}

// ---------------------------------------------------------------------------
// raster-perf-v3 item 二: full-warp 5-stage tree reduction.
//
// Contract: EVERY thread of the physical warp executes this, and threads that
// must not contribute pass 0.0f. That is why the loop body below was rewritten
// so that no `continue` can leave a lane outside the converged region: a sparse
// mask must never be applied to a plain shuffle tree, and __activemask() is not
// a convergence point. The result is only valid in lane 0.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_tree_sum(float v)
{
#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		v += __shfl_down_sync(0xffffffffu, v, off);
	return v;
}

// Old per-lane enumeration kept for reference only (archived candidate).
__device__ __forceinline__ void warp_agg_add(float* __restrict__ addr, const float val)
{
	const unsigned act = __activemask();
	if (act == 0) return;
	const int n = __popc(act);
	float total = 0.0f;
	for (int i = 0; i < n; ++i)
	{
		// MEASURED semantics (see [FnsTest]): __fns(mask, 0, k) with k >= 1 returns
		// the (k-1)-th set bit at or after bit 0, and k == 0 only tests bit 0.
		// The return type is unsigned, so the sentinel must be compared explicitly:
		// `lane < 0` on an unsigned value is always false.
		const unsigned lane = __fns(act, 0, i + 1);
		if (lane == 0xFFFFFFFFu || lane >= 32u)
			break;                      // never shuffle an invalid lane
		total += __shfl_sync(act, val, static_cast<int>(lane));
	}
	// The lane id must be the HARDWARE lane, i.e. the linear thread index within
	// the warp. The block is 16x16, so a warp spans two tile rows and
	// (threadIdx.x & 31) COLLIDES between them: two threads would be treated as
	// the leader and the aggregated sum would be added twice. That was the real
	// integration bug of the archived candidate.
	const unsigned laneid = (threadIdx.y * BLOCK_X + threadIdx.x) & 31u;
	if (laneid == (unsigned)(__ffs(act) - 1))
		atomicAdd(addr, total);
}

// raster-perf-v4 note: the old per-contribution dispatcher macro is gone. The
// reference kernel above uses plain atomicAdd unconditionally, and the candidate
// backends are separate kernel instances selected on the HOST.

// Backward version of the rendering procedure.
// A = REFERENCE. This kernel is intentionally left as the upstream algorithm:
// no backend template parameter and no extra device-side branch, so selecting a
// candidate backend cannot change the reference's cost or code generation.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// Compute blending values, as before.
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}
			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;


			// Helpful reusable temporary variables
			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

			// Update gradients w.r.t. 2D mean position of the Gaussian
			atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx * ddelx_dx);
			atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely * ddely_dy);

			// Update gradients w.r.t. 2D covariance (2x2 matrix, symmetric)
			atomicAdd(&dL_dconic2D[global_id].x, -0.5f * gdx * d.x * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].y, -0.5f * gdx * d.y * dL_dG);
			atomicAdd(&dL_dconic2D[global_id].w, -0.5f * gdy * d.y * dL_dG);

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), G * dL_dalpha);
		}
	}
}

void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* cov3Ds,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	const glm::vec3* campos,
	const float3* dL_dmean2D,
	const float* dL_dconic,
	glm::vec3* dL_dmean3D,
	float* dL_dcolor,
	float* dL_dcov3D,
	float* dL_dsh,
	glm::vec3* dL_dscale,
	glm::vec4* dL_drot)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	computeCov2DCUDA << <(P + 255) / 256, 256 >> > (
		P,
		means3D,
		radii,
		cov3Ds,
		focal_x,
		focal_y,
		tan_fovx,
		tan_fovy,
		viewmatrix,
		dL_dconic,
		(float3*)dL_dmean3D,
		dL_dcov3D);

	// Propagate gradients for remaining steps: finish 3D mean gradients,
	// propagate color gradients to SH (if desireD), propagate 3D covariance
	// matrix gradients to scale and rotation.
	preprocessCUDA<NUM_CHANNELS> << < (P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		projmatrix,
		campos,
		(float3*)dL_dmean2D,
		(glm::vec3*)dL_dmean3D,
		dL_dcolor,
		dL_dcov3D,
		dL_dsh,
		dL_dscale,
		dL_drot);
}

// ---------------------------------------------------------------------------
// raster-perf-v4: backend B (converged control flow, plain atomicAdd) and
// backend C (same, plus a 5-stage full-warp tree reduction written by one lane).
//
// Contract, identical for B and C:
//  * Every one of the 32 physical lanes of the warp reaches the common
//    write-back point in EVERY Gaussian iteration. The original `continue`s and
//    the `!done` loop exit are replaced by a per-iteration `valid` flag, and the
//    per-lane contribution registers are re-zeroed at the TOP of every iteration
//    so no value can leak from the previous one.
//  * `valid` therefore decides only whether a contribution is COMPUTED; it never
//    decides whether a lane takes part in the reduction.
//  * The hardware lane id is the block-linear index
//    (threadIdx.y * BLOCK_X + threadIdx.x) & 31, never threadIdx.x: the 16x16
//    block makes a warp span two tile rows of the SAME tile, and both rows must
//    use their true lane ids.
//  * All 32 lanes act on the same Gaussian and the same component, because
//    `collected_id[j]` is shared memory indexed by the loop variable `j`, which
//    is uniform across the block.
//  * For C the sum covers all 32 lanes (invalid lanes contribute exactly 0.0f)
//    and lane 0 writes it, so the leader's own pixel being invalid cannot drop
//    the other lanes' contributions.
//  * Transparency/colour recursion, the alpha threshold, last_contributor and
//    the transmittance semantics are unchanged for valid contributions. Threads
//    outside the image never read the image or the per-pixel buffers (their
//    `valid` is false from the first iteration), so no out-of-range load is
//    introduced.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void tree_write(float* __restrict__ addr, const float v)
{
	float s = v;
#pragma unroll
	for (int off = 16; off > 0; off >>= 1)
		s += __shfl_down_sync(0xFFFFFFFFu, s, off);
	if (((threadIdx.y * BLOCK_X + threadIdx.x) & 31u) == 0u)
		atomicAdd(addr, s);
}

template <uint32_t C, bool TREE>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_conv(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	float3* __restrict__ dL_dmean2D,
	float4* __restrict__ dL_dconic2D,
	float* __restrict__ dL_opacity,
	float* __restrict__ dL_dcolors)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];

	float last_alpha = 0;
	float last_color[C] = { 0 };

	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// `done` was `!inside` in the reference and is never assigned after that, so
	// it is kept as an explicit flag rather than as a loop-exit condition.
	const bool outside = !inside;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int k = 0; k < C; k++)
				collected_colors[k * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + k];
		}
		block.sync();

		// NOTE: `!done` is NOT part of this condition any more. Every lane
		// iterates the same number of times so the reduction below is executed
		// by the whole warp.
		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			bool valid = !outside && (contributor < last_contributor);

			// Contribution registers for THIS iteration, re-zeroed every time.
			float wr_col[C];
			for (int k = 0; k < C; k++) wr_col[k] = 0.0f;
			float wr_mx = 0.0f, wr_my = 0.0f;
			float wr_cx = 0.0f, wr_cy = 0.0f, wr_cw = 0.0f;
			float wr_op = 0.0f;

			if (valid)
			{
				const float2 xy = collected_xy[j];
				const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
				const float4 con_o = collected_conic_opacity[j];
				const float power =
					-0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
				if (power <= 0.0f)
				{
					const float G = expf(power);
					const float alpha = min(0.99f, con_o.w * G);
					if (alpha >= 1.0f / 255.0f)
					{
						T = T / (1.f - alpha);
						const float dchannel_dcolor = alpha * T;

						float dL_dalpha = 0.0f;
						for (int k = 0; k < C; k++)
						{
							const float c = collected_colors[k * BLOCK_SIZE + j];
							accum_rec[k] = last_alpha * last_color[k] + (1.f - last_alpha) * accum_rec[k];
							last_color[k] = c;
							const float dL_dchannel = dL_dpixel[k];
							dL_dalpha += (c - accum_rec[k]) * dL_dchannel;
							wr_col[k] = dchannel_dcolor * dL_dchannel;
						}
						dL_dalpha *= T;
						last_alpha = alpha;

						float bg_dot_dpixel = 0;
						for (int k = 0; k < C; k++)
							bg_dot_dpixel += bg_color[k] * dL_dpixel[k];
						dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

						const float dL_dG = con_o.w * dL_dalpha;
						const float gdx = G * d.x;
						const float gdy = G * d.y;
						const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
						const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;

						wr_mx = dL_dG * dG_ddelx * ddelx_dx;
						wr_my = dL_dG * dG_ddely * ddely_dy;
						wr_cx = -0.5f * gdx * d.x * dL_dG;
						wr_cy = -0.5f * gdx * d.y * dL_dG;
						wr_cw = -0.5f * gdy * d.y * dL_dG;
						wr_op = G * dL_dalpha;
					}
				}
			}

			// -------- common write-back point, executed by the whole warp -------
			const int gid = collected_id[j];
			if (TREE)
			{
				// The preceding `if (valid)` block diverges, so converge the warp
				// explicitly before the full-mask shuffles of the tree.
				__syncwarp();
				for (int k = 0; k < C; k++)
					tree_write(&dL_dcolors[gid * C + k], wr_col[k]);
				tree_write(&dL_dmean2D[gid].x, wr_mx);
				tree_write(&dL_dmean2D[gid].y, wr_my);
				tree_write(&dL_dconic2D[gid].x, wr_cx);
				tree_write(&dL_dconic2D[gid].y, wr_cy);
				tree_write(&dL_dconic2D[gid].w, wr_cw);
				tree_write(&dL_opacity[gid], wr_op);
			}
			else
			{
				if (valid)
				{
					for (int k = 0; k < C; k++)
						atomicAdd(&dL_dcolors[gid * C + k], wr_col[k]);
					atomicAdd(&dL_dmean2D[gid].x, wr_mx);
					atomicAdd(&dL_dmean2D[gid].y, wr_my);
					atomicAdd(&dL_dconic2D[gid].x, wr_cx);
					atomicAdd(&dL_dconic2D[gid].y, wr_cy);
					atomicAdd(&dL_dconic2D[gid].w, wr_cw);
					atomicAdd(&dL_opacity[gid], wr_op);
				}
			}
		}
	}
}

// ---------------------------------------------------------------------------
// raster-perf-v5: GOLDEN reference for the reduction, not a performance
// candidate. It computes the same per-contribution values as backend B but
// accumulates them into DOUBLE precision, where the error of summing a few
// thousand terms of comparable magnitude is ~1e-13 relative - far below fp32's
// ~1e-7. That makes it an independent high-precision reference for the question
// "does backend C add up the same set of contributions?", which the reference
// itself cannot answer because its own atomic order varies between runs.
//
// Its contribution formulas are a textual copy of the converged kernel, so it
// shares any error in the FORMULAS with B/C; it is independent only of the
// REDUCTION, which is exactly the property under test here.
// ---------------------------------------------------------------------------
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_golden(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	double* __restrict__ g_mx, double* __restrict__ g_my,
	double* __restrict__ g_cx, double* __restrict__ g_cy, double* __restrict__ g_cw,
	double* __restrict__ g_op, double* __restrict__ g_col)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];

	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C] = { 0 };
	if (inside)
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	float last_alpha = 0;
	float last_color[C] = { 0 };
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;
	const bool outside = !inside;

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			for (int k = 0; k < C; k++)
				collected_colors[k * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + k];
		}
		block.sync();

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor--;
			const bool valid = !outside && (contributor < last_contributor);
			if (!valid) continue;   // a plain skip is fine here: the target is double atomics
			const float2 xy = collected_xy[j];
			const float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			const float4 con_o = collected_conic_opacity[j];
			const float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f) continue;
			const float G = expf(power);
			const float alpha = min(0.99f, con_o.w * G);
			if (alpha < 1.0f / 255.0f) continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;
			float dL_dalpha = 0.0f;
			const int gid = collected_id[j];
			for (int k = 0; k < C; k++)
			{
				const float c = collected_colors[k * BLOCK_SIZE + j];
				accum_rec[k] = last_alpha * last_color[k] + (1.f - last_alpha) * accum_rec[k];
				last_color[k] = c;
				const float dL_dchannel = dL_dpixel[k];
				dL_dalpha += (c - accum_rec[k]) * dL_dchannel;
				atomicAdd(&g_col[gid * C + k], (double)(dchannel_dcolor * dL_dchannel));
			}
			dL_dalpha *= T;
			last_alpha = alpha;
			float bg_dot_dpixel = 0;
			for (int k = 0; k < C; k++) bg_dot_dpixel += bg_color[k] * dL_dpixel[k];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			const float dL_dG = con_o.w * dL_dalpha;
			const float gdx = G * d.x;
			const float gdy = G * d.y;
			const float dG_ddelx = -gdx * con_o.x - gdy * con_o.y;
			const float dG_ddely = -gdy * con_o.z - gdx * con_o.y;
			atomicAdd(&g_mx[gid], (double)(dL_dG * dG_ddelx * ddelx_dx));
			atomicAdd(&g_my[gid], (double)(dL_dG * dG_ddely * ddely_dy));
			atomicAdd(&g_cx[gid], (double)(-0.5f * gdx * d.x * dL_dG));
			atomicAdd(&g_cy[gid], (double)(-0.5f * gdx * d.y * dL_dG));
			atomicAdd(&g_cw[gid], (double)(-0.5f * gdy * d.y * dL_dG));
			atomicAdd(&g_op[gid], (double)(G * dL_dalpha));
		}
	}
}

// raster-perf-v5: the backend can be pinned per condition by the mapper, so one
// frozen process can compare A and C on the SAME scene, init and schedule. -1 =
// not pinned -> derive once from PHOTO_SLAM_RASTER_BACKEND (default A).
int g_raster_backend = -1;
// Experimental H mode uses the CPU-side active SH degree.  The mapper sets
// this before every training render; no GPU readback is involved.
int g_raster_sh = 0;
long long g_raster_h_a = 0, g_raster_h_c = 0, g_raster_h_switch = 0;
int g_raster_h_last_b = -1, g_raster_h_last_sh = -1;
// raster-perf-v6: launch counter, defined in rasterizer_impl.cu.
extern long long g_rast_ct_bwd_launch;
extern int g_rast_ct_on;
void SetRasterBackend(int b) {	g_raster_backend = b;
	printf("[RasterBackendPin] condition pinned to %s\n",
	       b == 0 ? "A" : (b == 1 ? "B" : (b == 2 ? "C" : "H")));
	fflush(stdout);
}
void SetRasterSh(int sh) { g_raster_sh = sh; }
void ResetRasterHStats()
{
	g_raster_h_a = 0;
	g_raster_h_c = 0;
	g_raster_h_switch = 0;
	g_raster_h_last_b = -1;
	g_raster_h_last_sh = -1;
}
void GetRasterHStats(long long& a, long long& c, long long& switches,
                     int& last_b, int& last_sh)
{
	a = g_raster_h_a;
	c = g_raster_h_c;
	switches = g_raster_h_switch;
	last_b = g_raster_h_last_b;
	last_sh = g_raster_h_last_sh;
}
void RasterBackendInitFromEnv()
{
	if (g_raster_backend >= 0) return;
	const char* e = std::getenv("PHOTO_SLAM_RASTER_BACKEND");
	int b = 0;
	if (e != nullptr)
	{
		if (e[0] == 'B' || e[0] == 'b') b = 1;
		else if (e[0] == 'C' || e[0] == 'c') b = 2;
		else if (e[0] == 'H' || e[0] == 'h') b = 3;
	}
	printf("[RasterBackend] mode = %s\n",
	       b == 0 ? "A reference" : (b == 1 ? "B converged" :
	       (b == 2 ? "C converged+warp" : "H SH-stage protection")));
	fflush(stdout);
	g_raster_backend = b;
}
int RasterBackendEffective()
{
	RasterBackendInitFromEnv();
	if (g_raster_backend == 3) return g_raster_sh >= 3 ? 0 : 2;
	return g_raster_backend < 0 ? 0 : g_raster_backend;
}
// raster-perf-v6: read the backend actually in force (pin or env), for the
// per-optimizer-update call trace.
int RasterBackendGet() { RasterBackendInitFromEnv(); return g_raster_backend; }

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	const int n_gaussians,
	int W, int H,
	const float* bg_color,
	const float2* means2D,
	const float4* conic_opacity,
	const float* colors,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float* dL_dpixels,
	float3* dL_dmean2D,
	float4* dL_dconic2D,
	float* dL_dopacity,
	float* dL_dcolors)
{
	// ---------------------------------------------------------------------
	// raster-perf-v4: HOST-side dispatch to three independent kernel instances.
	//   A = reference              renderCUDA<NUM_CHANNELS>
	//   B = converged control flow renderCUDA_conv<NUM_CHANNELS,false>
	//   C = B + 5-stage warp tree  renderCUDA_conv<NUM_CHANNELS,true>
	// Each backend is its own instantiation, so selecting B or C adds no
	// device-side branch and no extra register pressure to the reference.
	// ---------------------------------------------------------------------
	RasterBackendInitFromEnv();
	int backend = g_raster_backend;
	if (g_raster_backend == 3)
	{
		// H: fixed room1 stage rule.  active_sh < 3 -> C; active_sh >= 3 -> A.
		// This is a host-side kernel choice; the selected kernel is one of the
		// existing A/C instances and no CUDA formula is changed.
		backend = (g_raster_sh >= 3) ? 0 : 2;
		if (backend == 0) ++g_raster_h_a;
		else ++g_raster_h_c;
		if (backend != g_raster_h_last_b || g_raster_sh != g_raster_h_last_sh)
		{
			++g_raster_h_switch;
			g_raster_h_last_b = backend;
			g_raster_h_last_sh = g_raster_sh;
			printf("[RasterH] switch sh=%d backend=%s a=%lld c=%lld\n",
			       g_raster_sh, backend == 0 ? "A" : "C",
			       g_raster_h_a, g_raster_h_c);
			fflush(stdout);
		}
	}

	auto launch = [&](int b) {
		if (g_rast_ct_on) ++g_rast_ct_bwd_launch;
		if (b == 1)
			renderCUDA_conv<NUM_CHANNELS, false> << <grid, block >> >(
				ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors,
				final_Ts, n_contrib, dL_dpixels, dL_dmean2D, dL_dconic2D,
				dL_dopacity, dL_dcolors);
		else if (b == 2)
			renderCUDA_conv<NUM_CHANNELS, true> << <grid, block >> >(
				ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors,
				final_Ts, n_contrib, dL_dpixels, dL_dmean2D, dL_dconic2D,
				dL_dopacity, dL_dcolors);
		else
			renderCUDA<NUM_CHANNELS> << <grid, block >> >(
				ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors,
				final_Ts, n_contrib, dL_dpixels, dL_dmean2D, dL_dconic2D,
				dL_dopacity, dL_dcolors);
	};

	static const int selfcheck = []() {
		const char* e = std::getenv("PHOTO_SLAM_RASTER_AGG_SELFCHECK");
		return e ? std::atoi(e) : 0;
	}();
	// item 一 of the previous round: the __fns primitive table, run exactly once.
	if (selfcheck)
	{
		static bool fns_done = false;
		if (!fns_done)
		{
			fns_done = true;
			unsigned long long* d = nullptr;
			if (cudaMalloc(&d, 6 * 4 * sizeof(unsigned long long)) == cudaSuccess)
			{
				fnsSelfTestKernel << <1, 32 >> > (d);
				cudaDeviceSynchronize();
				unsigned long long h[24] = {0};
				cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
				printf("[FnsTest] __fns(mask, base=0, offset)  (0xffffffffffffffff = invalid)\n");
				const char* nm[6] = {"FFFFFFFF", "FFFFFFFE", "80000000",
				                     "00000001", "AAAAAAAA", "00050005"};
				for (int c = 0; c < 6; ++c)
					printf("    mask=%s  off0=%s  off1=%s  off2=%s\n", nm[c],
					       h[c*4+1] == 0xFFFFFFFFFFFFFFFFull ? "invalid" : std::to_string(h[c*4+1]).c_str(),
					       h[c*4+2] == 0xFFFFFFFFFFFFFFFFull ? "invalid" : std::to_string(h[c*4+2]).c_str(),
					       h[c*4+3] == 0xFFFFFFFFFFFFFFFFull ? "invalid" : std::to_string(h[c*4+3]).c_str());
				fflush(stdout);
				cudaFree(d);
			}
		}
	}

	// ---------------------------------------------------------------------
	// A / A-repeat / B / C on IDENTICAL inputs, into separate buffers.
	// Reported as three pairs: A-A (noise floor), A-B (control-flow refactor),
	// A-C and B-C (the tree). The selected backend is then launched ONE more time
	// into the caller's outputs, so a candidate training run genuinely returns the
	// candidate's gradients and each backward executes exactly one backend.
	//
	// Pre-stated acceptance: a pair passes when its max_abs is no worse than
	// 10x the A-A max_abs (or 1e-7, whichever is larger) AND its global relative
	// L2 is <= 1e-6. Near-zero entries are never judged by their own relative
	// error; the global L2 is used instead.
	// ---------------------------------------------------------------------
	if (selfcheck && n_gaussians > 0)
	{
		// Run for the first calls (small tiles / boundary cases) and then once per
		// new size bucket, so the check covers small AND real-scale inputs without
		// paying the comparison cost on every iteration.
		static int sc_calls = 0;
		static int sc_bucket = -1;
		const int bucket = n_gaussians / 40000;
		const bool do_check = (sc_calls < 2) || (bucket != sc_bucket);
		++sc_calls;
		sc_bucket = bucket;

		if (do_check)
		{
			const size_t P = (size_t)n_gaussians;
			const size_t n_mean = P * 3, n_conic = P * 4;
			const size_t n_op = P, n_col = P * NUM_CHANNELS;
			const size_t sz = (n_mean + n_conic + n_op + n_col) * sizeof(float);
			float *bA = nullptr, *bN = nullptr, *bB = nullptr, *bC = nullptr;
			cudaMalloc(&bA, sz); cudaMalloc(&bN, sz); cudaMalloc(&bB, sz); cudaMalloc(&bC, sz);
			if (bA && bN && bB && bC)
			{
				cudaMemset(bA, 0, sz); cudaMemset(bN, 0, sz);
				cudaMemset(bB, 0, sz); cudaMemset(bC, 0, sz);
				float* A_mean = bA; float* A_conic = A_mean + n_mean;
				float* A_op = A_conic + n_conic; float* A_col = A_op + n_op;
				float* N_mean = bN; float* N_conic = N_mean + n_mean;
				float* N_op = N_conic + n_conic; float* N_col = N_op + n_op;
				float* B_mean = bB; float* B_conic = B_mean + n_mean;
				float* B_op = B_conic + n_conic; float* B_col = B_op + n_op;
				float* C_mean = bC; float* C_conic = C_mean + n_mean;
				float* C_op = C_conic + n_conic; float* C_col = C_op + n_op;

				cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
				float msA = 0, msN = 0, msB = 0, msC = 0;
				#define TIME_LAUNCH(BK, MEAN, CONIC, OP, COL, MS) \
					cudaEventRecord(e0); \
					if ((BK) == 1) renderCUDA_conv<NUM_CHANNELS,false> << <grid, block >> >(ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors, final_Ts, n_contrib, dL_dpixels, (float3*)(MEAN), (float4*)(CONIC), (OP), (COL)); \
					else if ((BK) == 2) renderCUDA_conv<NUM_CHANNELS,true> << <grid, block >> >(ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors, final_Ts, n_contrib, dL_dpixels, (float3*)(MEAN), (float4*)(CONIC), (OP), (COL)); \
					else renderCUDA<NUM_CHANNELS> << <grid, block >> >(ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors, final_Ts, n_contrib, dL_dpixels, (float3*)(MEAN), (float4*)(CONIC), (OP), (COL)); \
					cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&(MS), e0, e1);
				TIME_LAUNCH(0, A_mean, A_conic, A_op, A_col, msA);
				TIME_LAUNCH(0, N_mean, N_conic, N_op, N_col, msN);
				TIME_LAUNCH(1, B_mean, B_conic, B_op, B_col, msB);
				TIME_LAUNCH(2, C_mean, C_conic, C_op, C_col, msC);
				#undef TIME_LAUNCH

				// ---- FP64 golden reference, independent of the reduction ----
				double *g_mx = nullptr, *g_my = nullptr, *g_cx = nullptr,
				       *g_cy = nullptr, *g_cw = nullptr, *g_op = nullptr, *g_col = nullptr;
				cudaMalloc(&g_mx, P * sizeof(double)); cudaMalloc(&g_my, P * sizeof(double));
				cudaMalloc(&g_cx, P * sizeof(double)); cudaMalloc(&g_cy, P * sizeof(double));
				cudaMalloc(&g_cw, P * sizeof(double)); cudaMalloc(&g_op, P * sizeof(double));
				cudaMalloc(&g_col, P * NUM_CHANNELS * sizeof(double));
				const bool have_gold = g_mx && g_my && g_cx && g_cy && g_cw && g_op && g_col;
				if (have_gold)
				{
					cudaMemset(g_mx, 0, P * sizeof(double)); cudaMemset(g_my, 0, P * sizeof(double));
					cudaMemset(g_cx, 0, P * sizeof(double)); cudaMemset(g_cy, 0, P * sizeof(double));
					cudaMemset(g_cw, 0, P * sizeof(double)); cudaMemset(g_op, 0, P * sizeof(double));
					cudaMemset(g_col, 0, P * NUM_CHANNELS * sizeof(double));
					renderCUDA_golden<NUM_CHANNELS> << <grid, block >> >(
						ranges, point_list, W, H, bg_color, means2D, conic_opacity, colors,
						final_Ts, n_contrib, dL_dpixels,
						g_mx, g_my, g_cx, g_cy, g_cw, g_op, g_col);
					cudaDeviceSynchronize();
				}

				// ---- per-group noise floor, measured BEFORE any judgement ----
				// The reference is NOT deterministic: its atomicAdd order varies
				// between launches, so A vs A already differs. Each group must
				// therefore be judged against ITS OWN reproducibility, never
				// against a fixed constant and never by the relative error of
				// near-zero entries.
				auto maxabs = [&](const float* x, const float* y, size_t n) {
					std::vector<float> hx(n), hy(n);
					cudaMemcpy(hx.data(), x, n * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(hy.data(), y, n * sizeof(float), cudaMemcpyDeviceToHost);
					double ma = 0.0, num = 0.0, den = 0.0;
					for (size_t i = 0; i < n; ++i) {
						const double d = std::fabs((double)hx[i] - (double)hy[i]);
						if (d > ma) ma = d;
						num += d * d;
						den += (double)hy[i] * (double)hy[i];
					}
					return std::make_pair(ma, std::sqrt(num) / (std::sqrt(den) + 1e-30));
				};
				const size_t ns[4] = { n_mean, n_conic, n_op, n_col };
				const char* gn[4] = { "mean2D", "conic2D", "opacity", "colors" };
				const float* Ag[4] = { A_mean, A_conic, A_op, A_col };
				const float* Ng[4] = { N_mean, N_conic, N_op, N_col };
				const float* Bg[4] = { B_mean, B_conic, B_op, B_col };
				const float* Cg[4] = { C_mean, C_conic, C_op, C_col };
				double noise[4] = {0, 0, 0, 0};
				printf("[AggSelfCheck] P=%zu  A=%.4f ms  A2=%.4f ms  B=%.4f ms  C=%.4f ms\n",
				       P, msA, msN, msB, msC);
				printf("  noise floor (A vs A repeat) -- the scale every judge uses:\n");
				for (int g = 0; g < 4; ++g) {
					const auto r = maxabs(Ng[g], Ag[g], ns[g]);
					noise[g] = r.first;
					printf("    %-8s max_abs=%.4e relL2=%.4e\n", gn[g], r.first, r.second);
				}
				// Pre-stated margin: a group passes when its max_abs is within 10x
				// the SAME group's A-A max_abs (or 1e-7), AND the global relative L2
				// is within 10x the same group's A-A relative L2 (or 1e-6).
				auto judge = [&](int g, const char* tag, const float* X) {
					const auto r = maxabs(X, Ag[g], ns[g]);
					const auto rn = maxabs(Ng[g], Ag[g], ns[g]);
					const double thr = std::max(10.0 * noise[g], 1e-7);
					const double thrl2 = std::max(10.0 * rn.second, 1e-6);
					printf("    %-3s %-8s max_abs=%.4e (thr %.3e, %.1fx noise)  relL2=%.4e  %s\n",
					       tag, gn[g], r.first, thr, noise[g] > 0 ? r.first / noise[g] : 0.0,
					       r.second, (r.first <= thr && r.second <= thrl2) ? "PASS" : "FAIL");
				};
				printf("  B vs A  (control-flow refactor only):\n");
				for (int g = 0; g < 4; ++g) judge(g, "B-A", Bg[g]);
				printf("  C vs A  (converged + tree):\n");
				for (int g = 0; g < 4; ++g) judge(g, "C-A", Cg[g]);
				printf("  C vs B  (tree only, same control flow):\n");
				for (int g = 0; g < 4; ++g) judge(g, "C-B", Cg[g]);

				// ---- independent FP64 reference: A, B and C against golden ----
				// This is the criterion that does NOT depend on the reference's own
				// scatter. The fp64 sum of a few thousand comparable terms has a
				// relative error ~1e-13, so a candidate whose deviation from golden is
				// at fp32 rounding level is adding up the same contributions.
				if (have_gold)
				{
					const size_t tot = n_mean + n_conic + n_op + n_col;
					std::vector<float> hA(tot), hB(tot), hC(tot);
					cudaMemcpy(hA.data(), bA, tot * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(hB.data(), bB, tot * sizeof(float), cudaMemcpyDeviceToHost);
					cudaMemcpy(hC.data(), bC, tot * sizeof(float), cudaMemcpyDeviceToHost);
					std::vector<double> dMx(P), dMy(P), dCx(P), dCy(P), dCw(P), dOp(P), dCol(P * NUM_CHANNELS);
					cudaMemcpy(dMx.data(), g_mx, P * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(dMy.data(), g_my, P * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(dCx.data(), g_cx, P * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(dCy.data(), g_cy, P * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(dCw.data(), g_cw, P * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(dOp.data(), g_op, P * sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(dCol.data(), g_col, P * NUM_CHANNELS * sizeof(double), cudaMemcpyDeviceToHost);

					const size_t off_mean = 0, off_conic = n_mean,
					             off_op = n_mean + n_conic, off_col = off_op + n_op;
					printf("  vs FP64 golden (fp32 rounding of a comparable-magnitude sum is ~1e-7):\n");
					auto report = [&](const char* bk, const std::vector<float>& h) {
						double ma, num, den;
						auto acc = [&](size_t off, int ncomp, int comp, const double* g, size_t n) {
							double m = 0, nn = 0, dd = 0;
							for (size_t i = 0; i < n; ++i) {
								const double d = std::fabs((double)h[off + i * ncomp + comp] - g[i]);
								if (d > m) m = d;
								nn += d * d; dd += g[i] * g[i];
							}
							ma = std::max(ma, m); num += nn; den += dd;
						};
						ma = 0; num = 0; den = 0;
						acc(off_mean, 3, 0, dMx.data(), P); acc(off_mean, 3, 1, dMy.data(), P);
						printf("    %s mean2D  max_abs=%.4e relL2=%.4e\n", bk, ma,
						       std::sqrt(num) / (std::sqrt(den) + 1e-30));
						ma = 0; num = 0; den = 0;
						acc(off_conic, 4, 0, dCx.data(), P); acc(off_conic, 4, 1, dCy.data(), P);
						acc(off_conic, 4, 3, dCw.data(), P);
						printf("    %s conic2D max_abs=%.4e relL2=%.4e\n", bk, ma,
						       std::sqrt(num) / (std::sqrt(den) + 1e-30));
						ma = 0; num = 0; den = 0;
						acc(off_op, 1, 0, dOp.data(), P);
						printf("    %s opacity max_abs=%.4e relL2=%.4e\n", bk, ma,
						       std::sqrt(num) / (std::sqrt(den) + 1e-30));
						ma = 0; num = 0; den = 0;
						acc(off_col, 1, 0, dCol.data(), P * NUM_CHANNELS);
						printf("    %s colors  max_abs=%.4e relL2=%.4e\n", bk, ma,
						       std::sqrt(num) / (std::sqrt(den) + 1e-30));
					};
					report("A", hA); report("B", hB); report("C", hC);
					cudaFree(g_mx); cudaFree(g_my); cudaFree(g_cx); cudaFree(g_cy);
					cudaFree(g_cw); cudaFree(g_op); cudaFree(g_col);
					fflush(stdout);
				}

				// ---- alternating multi-block timing, warm-up + median/dist ----
				const int WARM = 3, BLOCKS = 9;
				float tA[9], tB[9], tC[9];
				for (int w = 0; w < WARM; ++w) { launch(0); launch(1); launch(2); }
				cudaDeviceSynchronize();
				for (int k = 0; k < BLOCKS; ++k) {
					float ms = 0;
					cudaEventRecord(e0); launch(0);
					cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
					tA[k] = ms;
					cudaEventRecord(e0); launch(1);
					cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
					tB[k] = ms;
					cudaEventRecord(e0); launch(2);
					cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
					tC[k] = ms;
				}
				auto stat = [&](const char* nm, float* t) {
					std::vector<float> v(t, t + BLOCKS);
					std::sort(v.begin(), v.end());
					double s = 0; for (float x : v) s += x;
					printf("    %-2s min=%.4f median=%.4f max=%.4f mean=%.4f\n",
					       nm, v.front(), v[BLOCKS / 2], v.back(), s / BLOCKS);
				};
				printf("  alternating blocks (n=%d after %d warm-ups, same inputs):\n",
				       BLOCKS, WARM);
				stat("A", tA); stat("B", tB); stat("C", tC);
				fflush(stdout);
				cudaFree(bA); cudaFree(bN); cudaFree(bB); cudaFree(bC);
			}
		}
	}

	// Exactly ONE backend runs into the caller's buffers: the selected one.
	launch(backend);
}