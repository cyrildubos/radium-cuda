#ifndef RADIUM_RAY_HPP
#define RADIUM_RAY_HPP

#include "vector.cuh"

namespace Radium {
class Ray {
public:
  Vector origin;
  Vector direction;

  __host__ __device__ Ray(const Vector&, const Vector&);

  __host__ __device__ Vector at(double) const;
};
} // namespace Radium

#endif