#ifndef RADIUM_IMAGE_CUH
#define RADIUM_IMAGE_CUH

#include <fstream>
#include <iostream>

#include "camera.cuh"
#include "color.cuh"
#include "vector.cuh"

namespace Radium {
class Image {
  Radium::Color* data;

public:
  const std::size_t width;
  const std::size_t height;

  const Camera& camera;

  __host__ Image(const std::size_t, const std::size_t, const Camera&);
  __host__ ~Image();

  __host__ void render();

  __host__ void save(const std::string);
};
} // namespace Radium

#endif