#include <cuda_runtime.h>
#include <vector_types.h>
#include <math_functions.h>
#include <algorithm>
#include <cstdlib>

#include "mesh_kernels.cuh"

#ifdef __CUDACC__

static __device__ __forceinline__ float3 fbm_cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

static __device__ __forceinline__ float3 fbm_normalize(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-6f) {
        float inv = 1.0f / len;
        return make_float3(v.x * inv, v.y * inv, v.z * inv);
    }
    return make_float3(0.f, 1.f, 0.f);
}

__global__ void build_vertex_normal_kernel(const float* __restrict__ heights,
                                           float* __restrict__ vertices,
                                           float* __restrict__ normals,
                                           int size,
                                           float mesh_scale,
                                           float height_scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = size * size;
    if (idx >= total) return;

    int x = idx % size;
    int y = idx / size;

    float step = (size > 1) ? (mesh_scale / static_cast<float>(size - 1)) : mesh_scale;
    float half = mesh_scale * 0.5f;

    float height = heights[idx] * height_scale;

    float px = (size > 1) ? (x * step - half) : 0.0f;
    float pz = (size > 1) ? (y * step - half) : 0.0f;

    vertices[idx * 3 + 0] = px;
    vertices[idx * 3 + 1] = height;
    vertices[idx * 3 + 2] = pz;

    if (size <= 1) {
        normals[idx * 3 + 0] = 0.0f;
        normals[idx * 3 + 1] = 1.0f;
        normals[idx * 3 + 2] = 0.0f;
        return;
    }

    int xl = (x > 0) ? x - 1 : x;
    int xr = (x + 1 < size) ? x + 1 : x;
    int yd = (y > 0) ? y - 1 : y;
    int yu = (y + 1 < size) ? y + 1 : y;

    float hL = heights[y * size + xl] * height_scale;
    float hR = heights[y * size + xr] * height_scale;
    float hD = heights[yd * size + x] * height_scale;
    float hU = heights[yu * size + x] * height_scale;

    float spanX = (xr - xl) * step;
    if (spanX <= 1e-6f) spanX = step;
    float spanZ = (yu - yd) * step;
    if (spanZ <= 1e-6f) spanZ = step;

    float3 tangentX = make_float3(spanX, hR - hL, 0.0f);
    float3 tangentZ = make_float3(0.0f, hU - hD, spanZ);

    float3 n = fbm_cross(tangentZ, tangentX);
    n = fbm_normalize(n);

    normals[idx * 3 + 0] = n.x;
    normals[idx * 3 + 1] = n.y;
    normals[idx * 3 + 2] = n.z;
}

__global__ void build_topology_kernel(int grid_size,
                                      int quad_dim,
                                      int* counts,
                                      int* indices) {
    int quad = blockIdx.x * blockDim.x + threadIdx.x;
    int quad_total = quad_dim * quad_dim;
    if (quad >= quad_total) return;

    int y = quad / quad_dim;
    int x = quad % quad_dim;
    int face = quad * 2;
    int tri_idx = face * 3;

    int i0 = y * grid_size + x;
    int i1 = y * grid_size + (x + 1);
    int i2 = (y + 1) * grid_size + x;
    int i3 = (y + 1) * grid_size + (x + 1);

    counts[face + 0] = 3;
    counts[face + 1] = 3;

    indices[tri_idx + 0] = i0;
    indices[tri_idx + 1] = i2;
    indices[tri_idx + 2] = i1;
    indices[tri_idx + 3] = i1;
    indices[tri_idx + 4] = i2;
    indices[tri_idx + 5] = i3;
}

#endif // __CUDACC__

extern "C" void build_mesh_geometry_cuda(const float* heights,
                                         int size,
                                         float mesh_scale,
                                         float height_scale,
                                         float* vertices_out,
                                         float* normals_out) {
#ifndef __CUDACC__
    (void)heights;
    (void)size;
    (void)mesh_scale;
    (void)height_scale;
    (void)vertices_out;
    (void)normals_out;
#else
    if (!heights || !vertices_out || !normals_out || size <= 0) {
        return;
    }

    int total = size * size;
    size_t heights_bytes = sizeof(float) * static_cast<size_t>(total);
    size_t vector_bytes = sizeof(float) * static_cast<size_t>(total) * 3;

    float* d_heights = nullptr;
    float* d_vertices = nullptr;
    float* d_normals = nullptr;

    cudaMalloc(&d_heights, heights_bytes);
    cudaMalloc(&d_vertices, vector_bytes);
    cudaMalloc(&d_normals, vector_bytes);

    cudaMemcpy(d_heights, heights, heights_bytes, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    build_vertex_normal_kernel<<<blocks, threads>>>(
        d_heights, d_vertices, d_normals, size, mesh_scale, height_scale);
    cudaDeviceSynchronize();

    cudaMemcpy(vertices_out, d_vertices, vector_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(normals_out, d_normals, vector_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_heights);
    cudaFree(d_vertices);
    cudaFree(d_normals);
#endif
}

extern "C" void build_mesh_topology_cuda(int grid_size,
                                         int quad_dim,
                                         int* counts_out,
                                         int* indices_out) {
#ifndef __CUDACC__
    (void)grid_size;
    (void)quad_dim;
    (void)counts_out;
    (void)indices_out;
#else
    if (!counts_out || !indices_out || quad_dim <= 0 || grid_size <= 1) {
        return;
    }

    int quad_total = quad_dim * quad_dim;
    if (quad_total <= 0) {
        return;
    }
    int face_count = quad_total * 2;
    size_t counts_bytes = sizeof(int) * static_cast<size_t>(face_count);
    size_t indices_bytes = sizeof(int) * static_cast<size_t>(face_count) * 3;

    int* d_counts = nullptr;
    int* d_indices = nullptr;

    cudaMalloc(&d_counts, counts_bytes);
    cudaMalloc(&d_indices, indices_bytes);

    int threads = 256;
    int blocks = (quad_total + threads - 1) / threads;
    build_topology_kernel<<<blocks, threads>>>(
        grid_size, quad_dim, d_counts, d_indices);
    cudaDeviceSynchronize();

    cudaMemcpy(counts_out, d_counts, counts_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(indices_out, d_indices, indices_bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_counts);
    cudaFree(d_indices);
#endif
}
