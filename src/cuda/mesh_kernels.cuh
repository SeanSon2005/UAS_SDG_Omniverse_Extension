#pragma once

// Host-callable entry point that builds vertex and normal buffers for a grid mesh.
// The arrays pointed to by vertices_out and normals_out must contain at least
// size * size * 3 contiguous floats.
extern "C" void build_mesh_geometry_cuda(
    const float* heights,
    int size,
    float mesh_scale,
    float height_scale,
    float* vertices_out,
    float* normals_out);
