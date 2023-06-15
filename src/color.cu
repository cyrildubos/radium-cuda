#include <radium/color.cuh>

__device__ Radium::Color::Color(Vector vector)
    : r(0xFF * vector.x), g(0xFF * vector.y), b(0xFF * vector.z) {}