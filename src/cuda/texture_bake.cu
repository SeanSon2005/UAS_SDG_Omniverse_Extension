#include <cuda_runtime.h>
#include <math_constants.h>
#include <math_functions.h>
#include <algorithm>

#include "texture_bake.cuh"

#ifdef __CUDACC__

static __device__ __forceinline__ float fbm_saturate(float v) {
    return fminf(fmaxf(v, 0.0f), 1.0f);
}

static __device__ __forceinline__ float fbm_smoothstep(float edge0, float edge1, float x) {
    float t = fbm_saturate((x - edge0) / (edge1 - edge0 + 1e-6f));
    return t * t * (3.0f - 2.0f * t);
}

static __device__ __forceinline__ float srgb_to_linear(float value) {
    if (value <= 0.04045f) {
        return value / 12.92f;
    }
    return powf((value + 0.055f) / 1.055f, 2.4f);
}

static __device__ __forceinline__ float linear_to_srgb(float value) {
    value = fbm_saturate(value);
    if (value <= 0.0031308f) {
        return value * 12.92f;
    }
    return 1.055f * powf(value, 1.0f / 2.4f) - 0.055f;
}

static __device__ __forceinline__ int wrap_texel(float coord, int resolution) {
    int x = static_cast<int>(floorf(coord * resolution));
    x %= resolution;
    if (x < 0) {
        x += resolution;
    }
    return x;
}

static __device__ float4 sample_rgba(const float* texels, int width, int height, float u, float v) {
    if (!texels || width <= 0 || height <= 0) {
        return make_float4(0.f, 0.f, 0.f, 0.f);
    }
    u = u - floorf(u);
    v = v - floorf(v);
    int ix = wrap_texel(u, width);
    int iy = wrap_texel(v, height);
    int idx = (iy * width + ix) * 4;
    return make_float4(texels[idx + 0], texels[idx + 1], texels[idx + 2], texels[idx + 3]);
}

static __device__ float sample_scalar(const float* texels, int width, int height, float u, float v) {
    float4 rgba = sample_rgba(texels, width, height, u, v);
    return rgba.x;
}

static __device__ __forceinline__ float3 srgb_to_linear(const float3& c) {
    return make_float3(srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z));
}

static __device__ __forceinline__ float3 linear_to_srgb(const float3& c) {
    return make_float3(linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z));
}

__global__ void blend_layers_kernel(const float* __restrict__ heights,
                                    int grid_size,
                                    float min_height,
                                    float max_height,
                                    const TextureLayerDeviceDesc* __restrict__ layers,
                                    int layer_count,
                                    float* __restrict__ out_base_color,
                                    float* __restrict__ out_normals,
                                    float* __restrict__ out_roughness,
                                    float* __restrict__ out_specular) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = grid_size * grid_size;
    if (idx >= total) {
        return;
    }

    float height_value = heights[idx];
    float span = fmaxf(max_height - min_height, 1e-6f);
    float normalized = fbm_saturate((height_value - min_height) / span);

    int x = idx % grid_size;
    int y = idx / grid_size;
    float u = (grid_size > 1) ? (static_cast<float>(x) / static_cast<float>(grid_size - 1)) : 0.0f;
    float v = (grid_size > 1) ? (static_cast<float>(y) / static_cast<float>(grid_size - 1)) : 0.0f;

    float3 accum_color = make_float3(0.f, 0.f, 0.f);
    float3 accum_normal = make_float3(0.f, 0.f, 0.f);
    float accum_roughness = 0.f;
    float accum_specular = 0.f;
    float weight_sum = 0.f;

    for (int i = 0; i < layer_count; ++i) {
        const TextureLayerDeviceDesc& layer = layers[i];
        float span = fmaxf(layer.height_max - layer.height_min, 1e-4f);
        float half_span = span * 0.5f;
        float center = layer.height_min + half_span;
        float dist = fabsf(normalized - center);
        float w = fbm_saturate(1.0f - dist / (half_span + 1e-6f));
        w *= layer.blend_amount;
        if (w <= 0.0f) {
            continue;
        }

        float4 color_sample = sample_rgba(layer.base_color, layer.base_width, layer.base_height, u, v);
        float3 color_linear = srgb_to_linear(make_float3(color_sample.x, color_sample.y, color_sample.z));
        accum_color.x += color_linear.x * w;
        accum_color.y += color_linear.y * w;
        accum_color.z += color_linear.z * w;

        float4 normal_sample = layer.normal_map
            ? sample_rgba(layer.normal_map, layer.normal_width, layer.normal_height, u, v)
            : make_float4(0.5f, 0.5f, 1.0f, 1.0f);
        float3 normal_vec = make_float3(
            normal_sample.x * 2.0f - 1.0f,
            normal_sample.y * 2.0f - 1.0f,
            normal_sample.z * 2.0f - 1.0f);
        float len = sqrtf(fmaxf(normal_vec.x * normal_vec.x +
                                normal_vec.y * normal_vec.y +
                                normal_vec.z * normal_vec.z,
                                1e-8f));
        normal_vec.x /= len;
        normal_vec.y /= len;
        normal_vec.z /= len;
        accum_normal.x += normal_vec.x * w;
        accum_normal.y += normal_vec.y * w;
        accum_normal.z += normal_vec.z * w;

        float rough_sample = layer.roughness_map
            ? sample_scalar(layer.roughness_map, layer.roughness_width, layer.roughness_height, u, v)
            : layer.roughness_constant;
        accum_roughness += rough_sample * w;

        float spec_sample = layer.specular_map
            ? sample_scalar(layer.specular_map, layer.specular_width, layer.specular_height, u, v)
            : layer.specular_constant;
        accum_specular += spec_sample * w;

        weight_sum += w;
    }

    int base_idx = idx * 3;
    if (weight_sum > 1e-6f) {
        float inv = 1.0f / weight_sum;
        float3 color_lin = make_float3(accum_color.x * inv,
                                       accum_color.y * inv,
                                       accum_color.z * inv);
        float3 color_srgb = linear_to_srgb(color_lin);
        out_base_color[base_idx + 0] = color_srgb.x;
        out_base_color[base_idx + 1] = color_srgb.y;
        out_base_color[base_idx + 2] = color_srgb.z;

        float3 normal_vec = accum_normal;
        float len = sqrtf(fmaxf(normal_vec.x * normal_vec.x +
                                normal_vec.y * normal_vec.y +
                                normal_vec.z * normal_vec.z,
                                1e-8f));
        normal_vec.x /= len;
        normal_vec.y /= len;
        normal_vec.z /= len;
        out_normals[base_idx + 0] = normal_vec.x * 0.5f + 0.5f;
        out_normals[base_idx + 1] = normal_vec.y * 0.5f + 0.5f;
        out_normals[base_idx + 2] = normal_vec.z * 0.5f + 0.5f;

        out_roughness[idx] = fbm_saturate(accum_roughness * inv);
        out_specular[idx] = fbm_saturate(accum_specular * inv);
    } else {
        out_base_color[base_idx + 0] = 0.0f;
        out_base_color[base_idx + 1] = 0.0f;
        out_base_color[base_idx + 2] = 0.0f;
        out_normals[base_idx + 0] = 0.5f;
        out_normals[base_idx + 1] = 0.5f;
        out_normals[base_idx + 2] = 1.0f;
        out_roughness[idx] = 0.5f;
        out_specular[idx] = 0.5f;
    }
}

#endif // __CUDACC__

extern "C" void blend_layers_cuda(const float* d_heights,
                                  int grid_size,
                                  float min_height,
                                  float max_height,
                                  const TextureLayerDeviceDesc* d_layers,
                                  int layer_count,
                                  float* d_out_base_color,
                                  float* d_out_normals,
                                  float* d_out_roughness,
                                  float* d_out_specular) {
#ifdef __CUDACC__
    if (!d_heights || !d_layers || grid_size <= 0 || layer_count <= 0 ||
        !d_out_base_color || !d_out_normals || !d_out_roughness || !d_out_specular) {
        return;
    }

    const int total = grid_size * grid_size;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    blend_layers_kernel<<<blocks, threads>>>(
        d_heights,
        grid_size,
        min_height,
        max_height,
        d_layers,
        layer_count,
        d_out_base_color,
        d_out_normals,
        d_out_roughness,
        d_out_specular);
    cudaDeviceSynchronize();
#else
    (void)d_heights;
    (void)grid_size;
    (void)min_height;
    (void)max_height;
    (void)d_layers;
    (void)layer_count;
    (void)d_out_base_color;
    (void)d_out_normals;
    (void)d_out_roughness;
    (void)d_out_specular;
#endif
}
