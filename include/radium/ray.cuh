#ifndef RADIUM_RAY_HPP
#define RADIUM_RAY_HPP

#include "vector.cuh"

namespace Radium {
class Ray {
public:
  Vector origin;
  Vector direction;

  __device__ Ray();
  __device__ Ray(const Vector&, const Vector&);

  __device__ Vector at(const double) const;
};
} // namespace Radium

#endif