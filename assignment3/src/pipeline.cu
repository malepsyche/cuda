#include "pipeline.cuh"
#include <cmath>
#include <cstdio>

// Roofline sketch for the report. Gaussian does about twenty-five fused multiply-adds per output pixel once data sits in shared memory. Sobel does about eighteen multiply-adds plus sqrt per pixel from global memory. Histogram does one atomic per pixel. Equalize does a few float ops and one global read per pixel for the CDF table.

__device__ __forceinline__ int dev_clampi(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

// Gaussian weights in constant memory. Sum of weights is one.
__constant__ float c_gauss[5][5] = {
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 7.f/273, 26.f/273, 41.f/273, 26.f/273,  7.f/273 },
    { 4.f/273, 16.f/273, 26.f/273, 16.f/273,  4.f/273 },
    { 1.f/273,  4.f/273,  7.f/273,  4.f/273,  1.f/273 },
};

#define S_MEM_W (TILE_W + 2 * GAUSS_RADIUS)
#define S_MEM_H (TILE_H + 2 * GAUSS_RADIUS)

// Stage 1. Cooperative load of halo tile then 5x5 convolution from shared memory.
__global__ void gaussianBlurKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    __shared__ float s_tile[S_MEM_H][S_MEM_W];

    const int nthreads = blockDim.x * blockDim.y;
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;

    for (int idx = tid; idx < S_MEM_W * S_MEM_H; idx += nthreads) {
        const int sx = idx % S_MEM_W;
        const int sy = idx / S_MEM_W;
        int gx = blockIdx.x * TILE_W + sx - GAUSS_RADIUS;
        int gy = blockIdx.y * TILE_H + sy - GAUSS_RADIUS;
        gx = dev_clampi(gx, 0, width - 1);
        gy = dev_clampi(gy, 0, height - 1);
        s_tile[sy][sx] = static_cast<float>(in[gy * width + gx]);
    }

    __syncthreads();

    const int out_x = blockIdx.x * TILE_W + threadIdx.x;
    const int out_y = blockIdx.y * TILE_H + threadIdx.y;

    if (out_x >= width || out_y >= height)
        return;

    float sum = 0.f;
    for (int ki = 0; ki < 5; ++ki) {
        for (int kj = 0; kj < 5; ++kj) {
            sum += c_gauss[ki][kj] * s_tile[threadIdx.y + ki][threadIdx.x + kj];
        }
    }

    const int v = dev_clampi(static_cast<int>(roundf(sum)), 0, 255);
    out[out_y * width + out_x] = static_cast<uint8_t>(v);
}

__device__ __forceinline__ float sobel_sample(
    const uint8_t* in, int px, int py, int width, int height)
{
    const int cx = dev_clampi(px, 0, width - 1);
    const int cy = dev_clampi(py, 0, height - 1);
    return static_cast<float>(in[cy * width + cx]);
}

// Stage 2. Clamp-to-edge samples then Gx Gy and magnitude with roundf.
__global__ void sobelKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const float p00 = sobel_sample(in, x - 1, y - 1, width, height);
    const float p10 = sobel_sample(in, x,     y - 1, width, height);
    const float p20 = sobel_sample(in, x + 1, y - 1, width, height);
    const float p01 = sobel_sample(in, x - 1, y,     width, height);
    const float p11 = sobel_sample(in, x,     y,     width, height);
    const float p21 = sobel_sample(in, x + 1, y,     width, height);
    const float p02 = sobel_sample(in, x - 1, y + 1, width, height);
    const float p12 = sobel_sample(in, x,     y + 1, width, height);
    const float p22 = sobel_sample(in, x + 1, y + 1, width, height);

    const float gx = -p00 + p20 - 2.f * p01 + 2.f * p21 - p02 + p22;
    const float gy =  p00 + 2.f * p10 + p20 - p02 - 2.f * p12 - p22;

    const float mag = sqrtf(gx * gx + gy * gy);
    // Reference PGM edges use truncation toward zero of sqrt magnitude not roundf.
    const int   outv = dev_clampi(static_cast<int>(mag), 0, 255);
    out[y * width + x] = static_cast<uint8_t>(outv);
}

// Stage 3A Section 3.3. atomicAdd updates one bin per pixel as the brief describes.
__global__ void histogramKernel(
    const uint8_t*  __restrict__ in,
    unsigned int*   hist,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;
    const unsigned int v = in[y * width + x];
    atomicAdd(&hist[v], 1u);
}

// Stage 3C. Map each pixel using exclusive CDF and cdf_min on the host.
__global__ void equalizeKernel(
    const uint8_t* __restrict__ in,
    uint8_t*       __restrict__ out,
    const float*   cdf,
    float          cdf_min,
    int width, int height)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
        return;

    const int   v = in[y * width + x];
    const float c = cdf[v];
    const float n = static_cast<float>(width) * static_cast<float>(height);
    const float denom = n - cdf_min;

    if (denom <= 0.f) {
        out[y * width + x] = static_cast<uint8_t>(v);
        return;
    }

    const float t = (c - cdf_min) / denom * 255.f;
    const int   nv = dev_clampi(static_cast<int>(roundf(t)), 0, 255);
    out[y * width + x] = static_cast<uint8_t>(nv);
}
