#include <metal_stdlib>

using namespace metal;

kernel void mat_mul(
    const device float* a,
    const device float* b,
    device float* c,
    constant uint& m,
    constant uint& k,
    constant uint& n,
    uint2 id [[thread_position_in_grid]]
)
{
    uint row = id.y;
    uint col = id.x;

    if (row >= m || col >= n)
        return;

    float support = 0.0f;
    for (uint i = 0; i < k; i++)
    {
        support += a[row * k + i] * b[i * n + col];
    }

    c[row * n + col] = support;
}
