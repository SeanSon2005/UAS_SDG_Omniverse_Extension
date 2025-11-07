#pragma once

#include <string>
#include <vector>

#include <pxr/usd/usd/stage.h>

struct TextureLayerParams {
    std::string usd_name;
    std::string diffuse_texture;
    std::string normal_texture;
    std::string roughness_texture;
    std::string specular_texture;
    float height_min = 0.0f;
    float height_max = 1.0f;
    float blend_amount = 1.0f;
    float specular_constant = 0.5f;
    float roughness_constant = 0.5f;
};

struct TextureOutputPaths {
    std::string base_color;
    std::string normal;
    std::string roughness;
    std::string specular;

    bool IsValid() const {
        return !base_color.empty() && !normal.empty() && !roughness.empty() && !specular.empty();
    }
};

struct TextureBakeParams {
    std::vector<TextureLayerParams> layers;
    TextureOutputPaths outputs;

    bool ShouldBake() const {
        return !layers.empty() && outputs.IsValid();
    }
};

struct FbmMeshParams {
    int size = 0;
    float scale = 1.0f;
    float frequency = 1.0f;
    float mesh_scale = 1.0f;
    float lacunarity = 2.0f;
    float persistence = 0.5f;
    int seed = 0;
    int octaves = 1;
    float height_scale = 1.0f;
    std::string prim_path;
    TextureBakeParams texture_bake;
};

void GenerateFbmMesh(const pxr::UsdStageRefPtr& stage, const FbmMeshParams& params);
