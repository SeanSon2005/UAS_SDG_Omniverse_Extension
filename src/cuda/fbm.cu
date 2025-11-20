#include <cuda_runtime.h>
#include <vector_types.h>
#include <stdint.h>
#include <math.h>
#include <cstdlib>

#include "fbm.cuh"

#ifdef __CUDACC__

// Computes the fade curve for Perlin noise interpolation.
static __device__ __forceinline__ float fbm_fade(float t) {
    return t * t * t * (t * (t * 6.f - 15.f) + 10.f);
}

// Computes a hash value for 2D integer coordinates and a seed.
static __device__ __forceinline__ uint32_t fbm_hash2i(int x, int y, uint32_t seed) {
    uint32_t h = (uint32_t)x * 0x27d4eb2dU ^ (uint32_t)y * 0x85ebca6bU ^ seed * 0x9e3779b9U;
    h ^= h >> 16; h *= 0x7feb352dU;
    h ^= h >> 15; h *= 0x846ca68bU;
    h ^= h >> 16;
    return h;
}

// Computes the dot product of a gradient vector (determined by hash) and a distance vector.
static __device__ __forceinline__ float fbm_grad_dot(uint32_t h, float dx, float dy) {
    switch (h & 7U) {
        case 0:  return  dx +  dy;
        case 1:  return  dx -  dy;
        case 2:  return -dx +  dy;
        case 3:  return -dx -  dy;
        case 4:  return  dx;
        case 5:  return -dx;
        case 6:  return  dy;
        default: return -dy;
    }
}

// Computes 2D Perlin noise for a given point and seed.
static __device__ __forceinline__ float perlin2d(float x, float y, int seed, void*) {
    int   xi = (int)floorf(x);
    int   yi = (int)floorf(y);
    float xf = x - (float)xi;
    float yf = y - (float)yi;

    float u = fbm_fade(xf);
    float v = fbm_fade(yf);

    uint32_t h00 = fbm_hash2i(xi,     yi,     (uint32_t)seed);
    uint32_t h10 = fbm_hash2i(xi + 1, yi,     (uint32_t)seed);
    uint32_t h01 = fbm_hash2i(xi,     yi + 1, (uint32_t)seed);
    uint32_t h11 = fbm_hash2i(xi + 1, yi + 1, (uint32_t)seed);

    float g00 = fbm_grad_dot(h00,  xf,       yf);
    float g10 = fbm_grad_dot(h10,  xf - 1.f, yf);
    float g01 = fbm_grad_dot(h01,  xf,       yf - 1.f);
    float g11 = fbm_grad_dot(h11,  xf - 1.f, yf - 1.f);

    float x1 = g00 + u * (g10 - g00);
    float x2 = g01 + u * (g11 - g01);
    float n  = x1 + v * (x2 - x1);

    return n * 0.70710678f;
}

// Computes Fractal Brownian Motion (FBM) noise by summing octaves of Perlin noise.
static __device__ __forceinline__
float fbm2d_device(float2 point, float freq,
                   float lacun, float persist,
                   int init_seed, int octaves) {
    float amplitude = 1.0f;
    point.x *= freq;
    point.y *= freq;

    float result = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        int seed = (init_seed + i) & 0xffffffff;
        float n = perlin2d(point.x, point.y, seed, nullptr);
        result += n * amplitude;
        point.x *= lacun;
        point.y *= lacun;
        amplitude *= persist;
    }
    return result;
}

// CUDA kernel that computes FBM noise for an array of points in parallel.
__global__ void fbm2d_kernel(const float2* __restrict__ pts,
                             float* __restrict__ out,
                             int n,
                             float scale,
                             float freq, float lacun, float persist,
                             int init_seed, int octaves) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float2 p = pts[i];
        p.x *= scale;
        p.y *= scale;
        out[i] = fbm2d_device(p, freq, lacun, persist, init_seed, octaves);
    }
}

#endif // __CUDACC__

// Host wrapper function that manages memory and launches the FBM CUDA kernel.
extern "C" void fbm2d_cuda(float* h_out,
                           const float* h_x, const float* h_y,
                           int n,
                           float scale,
                           float freq, float lacun, float persist,
                           int init_seed, int octaves) {
#ifndef __CUDACC__
    (void)h_out; (void)h_x; (void)h_y; (void)n; (void)scale;
    (void)freq; (void)lacun; (void)persist; (void)init_seed; (void)octaves;
    return;
#else
    if (n <= 0) return;

    float2* d_pts = nullptr;
    float*  d_out = nullptr;
    cudaMalloc(&d_pts, sizeof(float2) * n);
    cudaMalloc(&d_out, sizeof(float)  * n);

    float2* h_pts = (float2*)std::malloc(sizeof(float2) * n);
    for (int i = 0; i < n; ++i) h_pts[i] = make_float2(h_x[i], h_y[i]);
    cudaMemcpy(d_pts, h_pts, sizeof(float2) * n, cudaMemcpyHostToDevice);
    std::free(h_pts);

    int t = 256, b = (n + t - 1) / t;
    fbm2d_kernel<<<b, t>>>(d_pts, d_out, n, scale, freq, lacun, persist, init_seed, octaves);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, sizeof(float) * n, cudaMemcpyDeviceToHost);
    cudaFree(d_pts);
    cudaFree(d_out);
#endif
}
