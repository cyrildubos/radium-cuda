#ifndef RADIUM_HIT_CUH
#define RADIUM_HIT_CUH

// #include <memory>

#include "vector.cuh"

namespace Radium {
struct Hit {
  Vector position;
  Vector normal;
  double t;

  // TODO: front face
};
} // namespace Radium

#endif