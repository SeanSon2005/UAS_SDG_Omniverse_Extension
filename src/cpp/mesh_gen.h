#pragma once

#include <string>
#include <pxr/usd/usd/stage.h>

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
};

void GenerateFbmMesh(const pxr::UsdStageRefPtr& stage, const FbmMeshParams& params);
