#pragma once

// Host-callable entry (C ABI) that launches the FBM kernel.
extern "C" void fbm2d_cuda(
    float* h_out,
    const float* h_x, const float* h_y,
    int n,
    float scale,
    float freq, float lacun, float persist,
    int init_seed, int octaves);
