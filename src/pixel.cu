#include <radium/pixel.cuh>

__host__ __device__ Radium::Pixel::Pixel(const Vector& vector)
    : r(0xFF * vector.x), g(0xFF * vector.y), b(0xFF * vector.z) {}