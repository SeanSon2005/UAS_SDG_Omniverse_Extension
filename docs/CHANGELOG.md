# Changelog

All notable changes to this extension are documented here.

## [1.1.0] - 2025-11-06
### Added
- CUDA-accelerated terrain pipeline that samples FBM noise, generates vertex/normal buffers, and authors a `UsdGeomMesh`.
- Pybind binding that accepts an active `Usd.Stage` and the full parameter set from the FBM UI.
- New user interface workflow in `python/impl/fbm.py` with terrain-specific controls and automatic prim replacement.
- Updated build script (`premake5.lua`) to pull in USD dependencies and compile the CUDA helpers.

### Changed
- Replaced the template documentation with FBM terrain generation guidance.
- Updated `config/extension.toml` metadata to reflect the new extension identity.

## [1.0.0] - 2022-06-30
### Added
- Initial template extension (historical reference).
