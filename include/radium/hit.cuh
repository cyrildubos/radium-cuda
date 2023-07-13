#ifndef RADIUM_HIT_CUH
#define RADIUM_HIT_CUH

#include <memory>

#include "vector.cuh"

namespace Radium {
class Material;

struct Hit {
  double t;
  Vector position;
  Vector normal;
  Material* material;

  // TODO: front face
};
} // namespace Radium

#endif