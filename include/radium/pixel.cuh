#ifndef RADIUM_PIXEL_CUH
#define RADIUM_PIXEL_CUH

#include <iostream>

#include "vector.cuh"

namespace Radium {
class Pixel {
public:
  std::uint8_t r;
  std::uint8_t g;
  std::uint8_t b;

  __host__ __device__ Pixel(const Vector&);
};
} // namespace Radium

__host__ inline std::ostream& operator<<(std::ostream& stream,
                                         const Radium::Pixel& pixel) {
  return stream << int(pixel.r) << ' ' << int(pixel.g) << ' ' << int(pixel.b);
}

#endif