#include "mesh_gen.h"

#include "fbm.cuh"
#include "mesh_kernels.cuh"

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/base/gf/range3f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/vt/array.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

pxr::SdfPath ResolvePrimPath(const std::string& prim_path) {
    const std::string default_path = "/World/FBMTerrain";
    pxr::SdfPath path = prim_path.empty() ? pxr::SdfPath(default_path) : pxr::SdfPath(prim_path);
    if (!path.IsAbsolutePath()) {
        throw std::runtime_error("Prim path must be an absolute USD path.");
    }
    return path;
}

} // namespace

void GenerateFbmMesh(const pxr::UsdStageRefPtr& stage, const FbmMeshParams& params) {
    if (!stage) {
        throw std::runtime_error("Invalid USD stage.");
    }
    if (params.size <= 0) {
        throw std::runtime_error("Terrain size must be greater than zero.");
    }
    if (params.octaves <= 0) {
        throw std::runtime_error("Octaves must be greater than zero.");
    }
    if (params.mesh_scale <= 0.0f) {
        throw std::runtime_error("Mesh scale must be greater than zero.");
    }

    int grid_size = params.size;
    const int sample_count = grid_size * grid_size;

    std::vector<float> grid_x(sample_count);
    std::vector<float> grid_y(sample_count);
    for (int y = 0; y < grid_size; ++y) {
        for (int x = 0; x < grid_size; ++x) {
            const int idx = y * grid_size + x;
            grid_x[idx] = static_cast<float>(x) / static_cast<float>(grid_size);
            grid_y[idx] = static_cast<float>(y) / static_cast<float>(grid_size);
        }
    }
    std::vector<float> heights(sample_count, 0.0f);

    fbm2d_cuda(
        heights.data(),
        grid_x.data(),
        grid_y.data(),
        sample_count,
        params.scale,
        params.frequency,
        params.lacunarity,
        params.persistence,
        params.seed,
        params.octaves);

    std::vector<float> vertices(static_cast<size_t>(sample_count) * 3, 0.0f);
    std::vector<float> normals(static_cast<size_t>(sample_count) * 3, 0.0f);

    build_mesh_geometry_cuda(
        heights.data(),
        grid_size,
        params.mesh_scale,
        params.height_scale,
        vertices.data(),
        normals.data());

    pxr::VtArray<pxr::GfVec3f> point_array(sample_count);
    pxr::VtArray<pxr::GfVec3f> normal_array(sample_count);
    pxr::GfRange3f bounds;
    bounds.SetEmpty();

    for (int i = 0; i < sample_count; ++i) {
        pxr::GfVec3f point(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
        pxr::GfVec3f normal(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
        point_array[i] = point;
        normal_array[i] = normal;
        bounds.ExtendBy(point);
    }

    const int quad_dim = std::max(0, grid_size - 1);
    const int quad_count = quad_dim * quad_dim;
    const int face_count = quad_count * 2;

    pxr::VtArray<int> counts(face_count);
    pxr::VtArray<int> indices(face_count * 3);

    for (int f = 0; f < face_count; ++f) {
        counts[f] = 3;
    }

    int face_index = 0;
    for (int y = 0; y < quad_dim; ++y) {
        for (int x = 0; x < quad_dim; ++x) {
            const int i0 = y * grid_size + x;
            const int i1 = y * grid_size + (x + 1);
            const int i2 = (y + 1) * grid_size + x;
            const int i3 = (y + 1) * grid_size + (x + 1);

            indices[face_index * 3 + 0] = i0;
            indices[face_index * 3 + 1] = i2;
            indices[face_index * 3 + 2] = i1;
            ++face_index;

            indices[face_index * 3 + 0] = i1;
            indices[face_index * 3 + 1] = i2;
            indices[face_index * 3 + 2] = i3;
            ++face_index;
        }
    }

    pxr::SdfPath prim_path = ResolvePrimPath(params.prim_path);
    if (stage->GetPrimAtPath(prim_path)) {
        stage->RemovePrim(prim_path);
    }

    pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh::Define(stage, prim_path);
    mesh.CreatePointsAttr().Set(point_array);
    mesh.CreateNormalsAttr().Set(normal_array);
    mesh.SetNormalsInterpolation(pxr::UsdGeomTokens->vertex);
    mesh.CreateFaceVertexCountsAttr().Set(counts);
    mesh.CreateFaceVertexIndicesAttr().Set(indices);

    pxr::VtArray<pxr::GfVec3f> extent(2);
    extent[0] = bounds.GetMin();
    extent[1] = bounds.GetMax();
    mesh.CreateExtentAttr().Set(extent);
}
