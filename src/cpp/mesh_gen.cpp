#include "mesh_gen.h"

#include "fbm.cuh"
#include "mesh_kernels.cuh"
#include "texture_bake.cuh"

#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/output.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/base/gf/range3f.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#define STBI_MSC_SECURE_CRT
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

namespace {

pxr::SdfPath ResolvePrimPath(const std::string& prim_path) {
    const std::string default_path = "/World/FBMTerrain";
    pxr::SdfPath path = prim_path.empty() ? pxr::SdfPath(default_path) : pxr::SdfPath(prim_path);
    if (!path.IsAbsolutePath()) {
        throw std::runtime_error("Prim path must be an absolute USD path.");
    }
    return path;
}

struct LoadedTexture {
    int width = 0;
    int height = 0;
    std::vector<float> pixels; // RGBA floats in [0, 1]

    bool IsValid() const {
        return width > 0 && height > 0 && !pixels.empty();
    }
};

LoadedTexture LoadTexture(const std::string& path, bool required) {
    LoadedTexture tex;
    if (path.empty()) {
        if (required) {
            throw std::runtime_error("Texture path is empty.");
        }
        return tex;
    }
    if (!std::filesystem::exists(path)) {
        if (required) {
            throw std::runtime_error("Texture file not found: " + path);
        }
        return tex;
    }

    int width = 0, height = 0, comp = 0;
    stbi_uc* data = stbi_load(path.c_str(), &width, &height, &comp, 4);
    if (!data) {
        if (required) {
            throw std::runtime_error("Failed to load texture: " + path);
        }
        return tex;
    }

    tex.width = width;
    tex.height = height;
    tex.pixels.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < pixel_count; ++i) {
        tex.pixels[i * 4 + 0] = static_cast<float>(data[i * 4 + 0]) / 255.0f;
        tex.pixels[i * 4 + 1] = static_cast<float>(data[i * 4 + 1]) / 255.0f;
        tex.pixels[i * 4 + 2] = static_cast<float>(data[i * 4 + 2]) / 255.0f;
        tex.pixels[i * 4 + 3] = static_cast<float>(data[i * 4 + 3]) / 255.0f;
    }
    stbi_image_free(data);
    return tex;
}

void EnsureOutputDirectory(const std::filesystem::path& output_path) {
    const auto parent = output_path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::filesystem::create_directories(parent);
    }
}

void SaveTexturePng(const std::string& path,
                    int grid_size,
                    int channels,
                    const std::vector<float>& float_data) {
    if (channels <= 0) {
        throw std::runtime_error("Channel count must be positive for texture output.");
    }
    const size_t expected = static_cast<size_t>(grid_size) * static_cast<size_t>(grid_size) *
        static_cast<size_t>(channels);
    if (float_data.size() != expected) {
        throw std::runtime_error("Unexpected texture buffer size for " + path);
    }

    std::vector<unsigned char> bytes(float_data.size());
    for (size_t i = 0; i < float_data.size(); ++i) {
        float v = std::clamp(float_data[i], 0.0f, 1.0f);
        bytes[i] = static_cast<unsigned char>(v * 255.0f + 0.5f);
    }
    EnsureOutputDirectory(path);
    if (!stbi_write_png(path.c_str(), grid_size, grid_size, channels, bytes.data(), grid_size * channels)) {
        throw std::runtime_error("Failed to write texture: " + path);
    }
}

void CudaCheck(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error (") + expr + "): " + cudaGetErrorString(err));
    }
}

struct DeviceBuffer {
    void* ptr = nullptr;
    size_t bytes = 0;

    DeviceBuffer() = default;
    DeviceBuffer(size_t size_bytes) : bytes(size_bytes) {
        if (size_bytes > 0) {
            CudaCheck(cudaMalloc(&ptr, size_bytes), "cudaMalloc");
        }
    }
    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept {
        ptr = other.ptr;
        bytes = other.bytes;
        other.ptr = nullptr;
        other.bytes = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) {
                cudaFree(ptr);
            }
            ptr = other.ptr;
            bytes = other.bytes;
            other.ptr = nullptr;
            other.bytes = 0;
        }
        return *this;
    }
};

struct LayerDeviceBuffers {
    DeviceBuffer base_color;
    DeviceBuffer normal;
    DeviceBuffer roughness;
    DeviceBuffer specular;
};

void BakeMaterialTextures(const std::vector<float>& heights,
                          int grid_size,
                          const TextureBakeParams& bake_params) {
    if (!bake_params.ShouldBake()) {
        return;
    }
    if (grid_size <= 0) {
        throw std::runtime_error("Grid size must be positive for texture bake.");
    }

    const int total = grid_size * grid_size;
    if (static_cast<int>(heights.size()) < total) {
        throw std::runtime_error("Height buffer size is insufficient for baking.");
    }

    auto [min_it, max_it] = std::minmax_element(heights.begin(), heights.begin() + total);
    const float min_height = (min_it != heights.end()) ? *min_it : 0.0f;
    const float max_height = (max_it != heights.end()) ? *max_it : 1.0f;

    struct HostLayerTextures {
        LoadedTexture base;
        LoadedTexture normal;
        LoadedTexture roughness;
        LoadedTexture specular;
    };

    std::vector<HostLayerTextures> host_layers;
    host_layers.reserve(bake_params.layers.size());
    for (const auto& layer : bake_params.layers) {
        HostLayerTextures host_tex;
        host_tex.base = LoadTexture(layer.diffuse_texture, /*required=*/true);
        host_tex.normal = LoadTexture(layer.normal_texture, /*required=*/false);
        host_tex.roughness = LoadTexture(layer.roughness_texture, /*required=*/false);
        host_tex.specular = LoadTexture(layer.specular_texture, /*required=*/false);
        host_layers.push_back(std::move(host_tex));
    }

    DeviceBuffer d_heights(static_cast<size_t>(total) * sizeof(float));
    CudaCheck(cudaMemcpy(d_heights.ptr,
                         heights.data(),
                         static_cast<size_t>(total) * sizeof(float),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy heights");

    std::vector<LayerDeviceBuffers> d_textures;
    d_textures.reserve(host_layers.size());
    std::vector<TextureLayerDeviceDesc> device_layers;
    device_layers.reserve(bake_params.layers.size());

    for (size_t i = 0; i < bake_params.layers.size(); ++i) {
        const auto& host = host_layers[i];
        LayerDeviceBuffers buffers;
        TextureLayerDeviceDesc desc{};

        if (host.base.IsValid()) {
            const auto tex_bytes = static_cast<size_t>(host.base.width) *
                static_cast<size_t>(host.base.height) * 4 * sizeof(float);
            buffers.base_color = DeviceBuffer(tex_bytes);
            CudaCheck(cudaMemcpy(buffers.base_color.ptr,
                                 host.base.pixels.data(),
                                 tex_bytes,
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy base color");
            desc.base_color = static_cast<float*>(buffers.base_color.ptr);
            desc.base_width = host.base.width;
            desc.base_height = host.base.height;
        }

        if (host.normal.IsValid()) {
            const auto tex_bytes = static_cast<size_t>(host.normal.width) *
                static_cast<size_t>(host.normal.height) * 4 * sizeof(float);
            buffers.normal = DeviceBuffer(tex_bytes);
            CudaCheck(cudaMemcpy(buffers.normal.ptr,
                                 host.normal.pixels.data(),
                                 tex_bytes,
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy normal");
            desc.normal_map = static_cast<float*>(buffers.normal.ptr);
            desc.normal_width = host.normal.width;
            desc.normal_height = host.normal.height;
        }

        if (host.roughness.IsValid()) {
            const auto tex_bytes = static_cast<size_t>(host.roughness.width) *
                static_cast<size_t>(host.roughness.height) * 4 * sizeof(float);
            buffers.roughness = DeviceBuffer(tex_bytes);
            CudaCheck(cudaMemcpy(buffers.roughness.ptr,
                                 host.roughness.pixels.data(),
                                 tex_bytes,
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy roughness");
            desc.roughness_map = static_cast<float*>(buffers.roughness.ptr);
            desc.roughness_width = host.roughness.width;
            desc.roughness_height = host.roughness.height;
        }

        if (host.specular.IsValid()) {
            const auto tex_bytes = static_cast<size_t>(host.specular.width) *
                static_cast<size_t>(host.specular.height) * 4 * sizeof(float);
            buffers.specular = DeviceBuffer(tex_bytes);
            CudaCheck(cudaMemcpy(buffers.specular.ptr,
                                 host.specular.pixels.data(),
                                 tex_bytes,
                                 cudaMemcpyHostToDevice),
                      "cudaMemcpy specular");
            desc.specular_map = static_cast<float*>(buffers.specular.ptr);
            desc.specular_width = host.specular.width;
            desc.specular_height = host.specular.height;
        }

        desc.height_min = bake_params.layers[i].height_min;
        desc.height_max = bake_params.layers[i].height_max;
        desc.blend_amount = bake_params.layers[i].blend_amount;
        desc.specular_constant = bake_params.layers[i].specular_constant;
        desc.roughness_constant = bake_params.layers[i].roughness_constant;

        d_textures.push_back(std::move(buffers));
        device_layers.push_back(desc);
    }

    DeviceBuffer d_layer_desc(device_layers.size() * sizeof(TextureLayerDeviceDesc));
    CudaCheck(cudaMemcpy(d_layer_desc.ptr,
                         device_layers.data(),
                         device_layers.size() * sizeof(TextureLayerDeviceDesc),
                         cudaMemcpyHostToDevice),
              "cudaMemcpy layer desc");

    DeviceBuffer d_base(static_cast<size_t>(total) * 3 * sizeof(float));
    DeviceBuffer d_normals(static_cast<size_t>(total) * 3 * sizeof(float));
    DeviceBuffer d_roughness(static_cast<size_t>(total) * sizeof(float));
    DeviceBuffer d_specular(static_cast<size_t>(total) * sizeof(float));

    blend_layers_cuda(static_cast<float*>(d_heights.ptr),
                      grid_size,
                      min_height,
                      max_height,
                      static_cast<TextureLayerDeviceDesc*>(d_layer_desc.ptr),
                      static_cast<int>(device_layers.size()),
                      static_cast<float*>(d_base.ptr),
                      static_cast<float*>(d_normals.ptr),
                      static_cast<float*>(d_roughness.ptr),
                      static_cast<float*>(d_specular.ptr));
    CudaCheck(cudaGetLastError(), "blend_layers_cuda");

    std::vector<float> base_pixels(static_cast<size_t>(total) * 3);
    std::vector<float> normal_pixels(static_cast<size_t>(total) * 3);
    std::vector<float> roughness_pixels(static_cast<size_t>(total));
    std::vector<float> specular_pixels(static_cast<size_t>(total));

    CudaCheck(cudaMemcpy(base_pixels.data(),
                         d_base.ptr,
                         base_pixels.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy base");
    CudaCheck(cudaMemcpy(normal_pixels.data(),
                         d_normals.ptr,
                         normal_pixels.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy normals");
    CudaCheck(cudaMemcpy(roughness_pixels.data(),
                         d_roughness.ptr,
                         roughness_pixels.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy roughness");
    CudaCheck(cudaMemcpy(specular_pixels.data(),
                         d_specular.ptr,
                         specular_pixels.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "cudaMemcpy specular");

    SaveTexturePng(bake_params.outputs.base_color, grid_size, 3, base_pixels);
    SaveTexturePng(bake_params.outputs.normal, grid_size, 3, normal_pixels);
    SaveTexturePng(bake_params.outputs.roughness, grid_size, 1, roughness_pixels);
    SaveTexturePng(bake_params.outputs.specular, grid_size, 1, specular_pixels);
}

pxr::UsdShadeShader CreatePrimvarReader(const pxr::UsdStageRefPtr& stage,
                                        const pxr::SdfPath& parent_path) {
    pxr::UsdShadeShader primvar =
        pxr::UsdShadeShader::Define(stage, parent_path.AppendChild(pxr::TfToken("Primvar_st")));
    primvar.CreateIdAttr().Set(pxr::TfToken("UsdPrimvarReader_float2"));
    primvar.CreateInput(pxr::TfToken("varname"), pxr::SdfValueTypeNames->String).Set("st");
    primvar.CreateOutput(pxr::TfToken("result"), pxr::SdfValueTypeNames->Float2);
    return primvar;
}

pxr::UsdShadeShader CreateTextureNode(const pxr::UsdStageRefPtr& stage,
                                      const pxr::SdfPath& material_path,
                                      const std::string& name,
                                      const std::string& file_path,
                                      const pxr::UsdShadeOutput& st_output,
                                      const pxr::TfToken& color_space) {
    pxr::UsdShadeShader shader =
        pxr::UsdShadeShader::Define(stage, material_path.AppendChild(pxr::TfToken(name)));
    shader.CreateIdAttr().Set(pxr::TfToken("UsdUVTexture"));
    shader.CreateInput(pxr::TfToken("file"), pxr::SdfValueTypeNames->Asset)
        .Set(pxr::SdfAssetPath(file_path));
    shader.CreateInput(pxr::TfToken("st"), pxr::SdfValueTypeNames->Float2)
        .ConnectToSource(st_output);
    shader.CreateInput(pxr::TfToken("wrapS"), pxr::SdfValueTypeNames->Token)
        .Set(pxr::TfToken("repeat"));
    shader.CreateInput(pxr::TfToken("wrapT"), pxr::SdfValueTypeNames->Token)
        .Set(pxr::TfToken("repeat"));
    shader.CreateInput(pxr::TfToken("sourceColorSpace"), pxr::SdfValueTypeNames->Token)
        .Set(color_space);
    shader.CreateOutput(pxr::TfToken("rgb"), pxr::SdfValueTypeNames->Float3);
    shader.CreateOutput(pxr::TfToken("r"), pxr::SdfValueTypeNames->Float);
    shader.CreateOutput(pxr::TfToken("g"), pxr::SdfValueTypeNames->Float);
    shader.CreateOutput(pxr::TfToken("b"), pxr::SdfValueTypeNames->Float);
    return shader;
}

void ApplyBakedMaterial(const pxr::UsdStageRefPtr& stage,
                        const pxr::SdfPath& mesh_path,
                        const TextureBakeParams& bake_params) {
    if (!stage || !mesh_path.IsAbsolutePath() || !bake_params.ShouldBake()) {
        return;
    }

    pxr::UsdGeomMesh mesh(stage->GetPrimAtPath(mesh_path));
    if (!mesh) {
        return;
    }

    const pxr::SdfPath looks_path("/World/Looks");
    if (!stage->GetPrimAtPath(looks_path)) {
        stage->DefinePrim(looks_path, pxr::TfToken("Scope"));
    }

    const pxr::SdfPath material_path = looks_path.AppendChild(pxr::TfToken("FBM_Material"));
    pxr::UsdShadeMaterial material = pxr::UsdShadeMaterial::Define(stage, material_path);

    pxr::UsdShadeShader primvar_reader = CreatePrimvarReader(stage, material_path);
    pxr::UsdShadeOutput st_output = primvar_reader.GetOutput(pxr::TfToken("result"));

    pxr::UsdShadeShader base_tex = CreateTextureNode(
        stage, material_path, "BaseColorTex", bake_params.outputs.base_color, st_output, pxr::TfToken("sRGB"));
    pxr::UsdShadeShader normal_tex = CreateTextureNode(
        stage, material_path, "NormalTex", bake_params.outputs.normal, st_output, pxr::TfToken("raw"));
    normal_tex.CreateInput(pxr::TfToken("scale"), pxr::SdfValueTypeNames->Float4)
        .Set(pxr::GfVec4f(2.0f, 2.0f, 2.0f, 1.0f));
    normal_tex.CreateInput(pxr::TfToken("bias"), pxr::SdfValueTypeNames->Float4)
        .Set(pxr::GfVec4f(-1.0f, -1.0f, -1.0f, 0.0f));
    pxr::UsdShadeShader rough_tex = CreateTextureNode(
        stage, material_path, "RoughnessTex", bake_params.outputs.roughness, st_output, pxr::TfToken("raw"));
    pxr::UsdShadeShader spec_tex = CreateTextureNode(
        stage, material_path, "SpecularTex", bake_params.outputs.specular, st_output, pxr::TfToken("raw"));

    pxr::UsdShadeShader preview =
        pxr::UsdShadeShader::Define(stage, material_path.AppendChild(pxr::TfToken("PreviewSurface")));
    preview.CreateIdAttr().Set(pxr::TfToken("UsdPreviewSurface"));
    preview.CreateInput(pxr::TfToken("diffuseColor"), pxr::SdfValueTypeNames->Color3f)
        .ConnectToSource(base_tex.GetOutput(pxr::TfToken("rgb")));
    preview.CreateInput(pxr::TfToken("roughness"), pxr::SdfValueTypeNames->Float)
        .ConnectToSource(rough_tex.GetOutput(pxr::TfToken("r")));
    preview.CreateInput(pxr::TfToken("specular"), pxr::SdfValueTypeNames->Float)
        .ConnectToSource(spec_tex.GetOutput(pxr::TfToken("r")));
    preview.CreateInput(pxr::TfToken("normal"), pxr::SdfValueTypeNames->Float3)
        .ConnectToSource(normal_tex.GetOutput(pxr::TfToken("rgb")));
    preview.CreateInput(pxr::TfToken("metallic"), pxr::SdfValueTypeNames->Float).Set(0.0f);
    preview.CreateInput(pxr::TfToken("opacity"), pxr::SdfValueTypeNames->Float).Set(1.0f);

    pxr::UsdShadeOutput preview_surface =
        preview.CreateOutput(pxr::TfToken("surface"), pxr::SdfValueTypeNames->Token);
    pxr::UsdShadeOutput surface_output = material.CreateSurfaceOutput();
    surface_output.ConnectToSource(preview_surface);

    pxr::UsdShadeMaterialBindingAPI binding(mesh.GetPrim());
    binding.Bind(material);
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

    BakeMaterialTextures(heights, grid_size, params.texture_bake);

    pxr::VtArray<pxr::GfVec3f> point_array(sample_count);
    pxr::VtArray<pxr::GfVec3f> normal_array(sample_count);
    pxr::VtArray<pxr::GfVec2f> st_array(sample_count);
    pxr::GfRange3f bounds;
    bounds.SetEmpty();

    for (int i = 0; i < sample_count; ++i) {
        pxr::GfVec3f point(vertices[i * 3 + 0], vertices[i * 3 + 1], vertices[i * 3 + 2]);
        pxr::GfVec3f normal(normals[i * 3 + 0], normals[i * 3 + 1], normals[i * 3 + 2]);
        point_array[i] = point;
        normal_array[i] = normal;
        bounds.ExtendBy(point);

        int v_idx = i / grid_size;
        int u_idx = i % grid_size;
        float u = (grid_size > 1) ? static_cast<float>(u_idx) / static_cast<float>(grid_size - 1) : 0.0f;
        float v = (grid_size > 1) ? static_cast<float>(v_idx) / static_cast<float>(grid_size - 1) : 0.0f;
        st_array[i] = pxr::GfVec2f(u, v);
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
    pxr::UsdGeomPrimvarsAPI primvars_api(mesh);
    pxr::UsdGeomPrimvar st_primvar = primvars_api.CreatePrimvar(
        pxr::TfToken("st"), pxr::SdfValueTypeNames->TexCoord2fArray, pxr::UsdGeomTokens->vertex);
    st_primvar.Set(st_array);

    pxr::VtArray<pxr::GfVec3f> extent(2);
    extent[0] = bounds.GetMin();
    extent[1] = bounds.GetMax();
    mesh.CreateExtentAttr().Set(extent);

    ApplyBakedMaterial(stage, prim_path, params.texture_bake);
}
