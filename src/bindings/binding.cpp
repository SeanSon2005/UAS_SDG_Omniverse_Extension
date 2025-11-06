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
                         const std::string& prim_path) {
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
        py::arg("prim_path") = std::string("/World/FBMTerrain"));
}
