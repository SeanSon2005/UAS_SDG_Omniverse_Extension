#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

struct TextureLayerDeviceDesc {
    const float* base_color;
    int base_width;
    int base_height;
    const float* normal_map;
    int normal_width;
    int normal_height;
    const float* roughness_map;
    int roughness_width;
    int roughness_height;
    const float* specular_map;
    int specular_width;
    int specular_height;
    float height_min;
    float height_max;
    float blend_amount;
    float specular_constant;
    float roughness_constant;
};

#ifdef __cplusplus
extern "C" {
#endif

void blend_layers_cuda(const float* d_heights,
                       int grid_size,
                       float min_height,
                       float max_height,
                       const TextureLayerDeviceDesc* d_layers,
                       int layer_count,
                       float* d_out_base_color,
                       float* d_out_normals,
                       float* d_out_roughness,
                       float* d_out_specular);

#ifdef __cplusplus
}
#endif
