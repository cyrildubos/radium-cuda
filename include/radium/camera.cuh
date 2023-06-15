#ifndef RADIUM_CAMERA_HPP
#define RADIUM_CAMERA_HPP

#include "ray.cuh"
#include "vector.cuh"

namespace Radium {
class Camera {
public:
  Vector position;

  Vector horizontal;
  Vector vertical;

  Vector origin;

  __host__ Camera(const double);

  __device__ Ray get_ray(double, double) const;
};
} // namespace Radium

#endif