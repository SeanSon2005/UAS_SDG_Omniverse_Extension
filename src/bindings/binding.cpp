#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pxr/usd/usd/stage.h>
#include <pxr/external/boost/python.hpp>
#include <pxr/base/tf/pyLock.h>

#include <stdexcept>

#include "mesh_gen.h"

namespace py = pybind11;
namespace bp = PXR_BOOST_PYTHON_NAMESPACE;

namespace {

TextureLayerParams ParseLayerDict(const py::handle& layer_obj) {
    TextureLayerParams layer;
    py::dict dict = py::reinterpret_borrow<py::dict>(layer_obj);

    auto require_key = [&](const char* key) -> py::handle {
        if (!dict.contains(key)) {
            throw std::runtime_error(std::string("Texture layer missing key: ") + key);
        }
        return dict[key];
    };

    layer.usd_name = py::cast<std::string>(require_key("usd_name"));
    layer.diffuse_texture = py::cast<std::string>(require_key("diffuse_texture"));
    layer.height_min = py::cast<float>(require_key("height_min"));
    layer.height_max = py::cast<float>(require_key("height_max"));
    layer.blend_amount = py::cast<float>(require_key("blend_amount"));
    layer.diffuse_texture = py::cast<std::string>(require_key("diffuse_texture"));
    layer.normal_texture = py::cast<std::string>(require_key("normal_texture"));
    layer.roughness_texture = py::cast<std::string>(require_key("roughness_texture"));
    layer.specular_texture = py::cast<std::string>(require_key("specular_texture"));
    if (dict.contains("specular_constant")) {
        layer.specular_constant = py::cast<float>(dict["specular_constant"]);
    }
    if (dict.contains("roughness_constant")) {
        layer.roughness_constant = py::cast<float>(dict["roughness_constant"]);
    }

    return layer;
}

TextureOutputPaths ParseOutputPaths(const py::handle& outputs_obj) {
    TextureOutputPaths outputs;
    if (!outputs_obj || outputs_obj.is_none()) {
        return outputs;
    }
    py::dict dict = py::reinterpret_borrow<py::dict>(outputs_obj);
    auto get_optional = [&](const char* key) -> std::string {
        if (!dict.contains(key)) {
            return std::string();
        }
        return py::cast<std::string>(dict[key]);
    };
    outputs.base_color = get_optional("base_color");
    outputs.normal = get_optional("normal");
    outputs.roughness = get_optional("roughness");
    outputs.specular = get_optional("specular");
    return outputs;
}

void GenerateMeshBinding(py::object stage_obj,
                         int size,
                         float scale,
                         float frequency,
                         float mesh_scale,
                         float lacunarity,
                         float persistence,
                         int seed,
                         int octaves,
                         float height_scale,
                         const std::string& prim_path,
                         py::object texture_layers,
                         py::object texture_output_paths) {
    pxr::TfPyLock pyLock;

    bp::object boost_stage(
        bp::handle<>(bp::borrowed(stage_obj.ptr())));

    bp::extract<pxr::UsdStageRefPtr> extractor(boost_stage);
    if (!extractor.check()) {
        throw std::runtime_error("Unable to convert Python stage object to UsdStage.");
    }

    pxr::UsdStageRefPtr stage = extractor();

    if (!stage) {
        throw std::runtime_error("Stage handle is invalid.");
    }

    FbmMeshParams params;
    params.size = size;
    params.scale = scale;
    params.frequency = frequency;
    params.mesh_scale = mesh_scale;
    params.lacunarity = lacunarity;
    params.persistence = persistence;
    params.seed = seed;
    params.octaves = octaves;
    params.height_scale = height_scale;
    params.prim_path = prim_path;
    params.texture_bake.outputs = ParseOutputPaths(texture_output_paths);
    if (!texture_layers.is_none()) {
        py::list layer_list = texture_layers.cast<py::list>();
        const py::size_t count = py::len(layer_list);
        params.texture_bake.layers.reserve(static_cast<size_t>(count));
        for (const auto& entry : layer_list) {
            params.texture_bake.layers.push_back(ParseLayerDict(entry));
        }
    }

    GenerateFbmMesh(stage, params);
}

} // namespace

PYBIND11_MODULE(_uas_fbm, m) {
    m.doc() = "FBM terrain mesh generator bindings";
    m.def(
        "generate_fbm_mesh",
        &GenerateMeshBinding,
        py::arg("stage"),
        py::arg("size"),
        py::arg("scale"),
        py::arg("frequency"),
        py::arg("mesh_scale"),
        py::arg("lacunarity"),
        py::arg("persistence"),
        py::arg("seed"),
        py::arg("octaves"),
        py::arg("height_scale"),
        py::arg("prim_path") = std::string("/World/FBMTerrain"),
        py::arg("texture_layers") = py::list(),
        py::arg("texture_output_paths") = py::dict());
}
