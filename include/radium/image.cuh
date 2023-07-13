#ifndef RADIUM_IMAGE_CUH
#define RADIUM_IMAGE_CUH

#include <fstream>
#include <iostream>

#include "hittables/scene.cuh"

#include "camera.cuh"
#include "hit.cuh"
#include "material.cuh"
#include "pixel.cuh"
#include "utilities.cuh"
#include "vector.cuh"

namespace Radium {
class Image {
  Radium::Pixel* pixels;

public:
  const std::size_t width;
  const std::size_t height;

  const Camera& camera;

  const Hittables::Scene& scene;

  Image(const std::size_t, const std::size_t, const Camera&,
        const Hittables::Scene&);
  ~Image();

  void render();

  void save(const std::string) const;
};
} // namespace Radium

#endif