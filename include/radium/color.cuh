#ifndef RADIUM_COLOR_CUH
#define RADIUM_COLOR_CUH

#include <iostream>

#include "vector.cuh"

namespace Radium {
class Color {
public:
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;

  __device__ Color(Vector);
};
} // namespace Radium

__host__ inline std::ostream& operator<<(std::ostream& stream,
                                         const Radium::Color& color) {
  return stream << int(color.r) << ' ' << int(color.g) << ' ' << int(color.b);
}

#endif