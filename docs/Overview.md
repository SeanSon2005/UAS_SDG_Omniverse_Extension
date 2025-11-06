# Overview

The **FBM Terrain Generator** extension turns the Omniverse Kit FBM UI into a fully GPU-backed terrain pipeline.  
At runtime it samples fractal Brownian motion on the GPU, converts the height field into mesh data, and publishes a new `UsdGeomMesh` directly into the active stage.

## What the extension does
- Captures user parameters (frequency, lacunarity, mesh size, etc.) from a custom Kit UI panel.
- Samples a tiled FBM height field with the existing CUDA implementation in `src/cuda/fbm.cu`.
- Launches a custom CUDA kernel to assemble vertex positions and per-vertex normals.
- Replaces a mesh prim on the stage (`/World/FBMTerrain` by default) with the newly generated geometry.
- Exposes a single Python entry-point (`_uas_fbm.generate_fbm_mesh`) that can be called from any Kit UI or script.

## Runtime pipeline
1. **UI** – `python/impl/fbm.py` gathers slider/drag values and the active USD stage reference.
2. **Python ↔ C++ bridge** – `src/bindings/binding.cpp` converts the Python `Usd.Stage` into a `UsdStageRefPtr` using Boost.Python helpers and forwards the call.
3. **Terrain generation** – `src/cpp/mesh_gen.cpp` invokes `fbm2d_cuda`, builds geometry buffers with `build_mesh_geometry_cuda`, and writes a `UsdGeomMesh`.
4. **CUDA kernels** – `src/cuda/mesh_kernels.cu` computes vertex positions and normals entirely on the GPU for maximum throughput.

## Key source files
| Component | Path | Purpose |
|-----------|------|---------|
| UI logic | `python/impl/fbm.py` | Presents parameters and invokes the binding. |
| Pybind module | `src/bindings/binding.cpp` | Bridges Python calls to the C++ terrain generator. |
| Mesh generation | `src/cpp/mesh_gen.h/.cpp` | Orchestrates FBM sampling, CUDA kernels, and USD authoring. |
| CUDA helpers | `src/cuda/mesh_kernels.cuh/.cu` | Construct vertex/normal buffers from the height field. |
| FBM kernel | `src/cuda/fbm.cu` | Existing FBM sampler used by the generator. |

## UI parameters at a glance
- `Frequency`, `Scale`, `Lacunarity`, `Persistence`, `Seed`, `Octaves`
- `Size` (resolution of the FBM grid), `Mesh Scale` (world size), `Height Scale`
- The mesh prim path can be overridden via `self.mesh_prim_path` in `fbm.py`.

## Usage
1. Build the extension: `repo.bat build`.
2. Launch Kit with the extension enabled.
3. Open *FBM Window*, adjust parameters, and click **Generate**.
4. Inspect `/World/FBMTerrain` in the stage to view the regenerated terrain.

## Development tips
- The extension rebuilds a mesh prim each invocation; reuse the prim path if you need persistent references.
- CUDA compilation is handled by a custom NVCC rule in `premake5.lua`; ensure `CUDA_PATH` is configured before building.
- When experimenting with additional mesh data (UVs, tangents, etc.), extend `mesh_kernels.cu` and populate the corresponding USD attributes in `mesh_gen.cpp`.

