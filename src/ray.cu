#include <radium/ray.cuh>

__device__ Radium::Ray::Ray() {}

__device__ Radium::Ray::Ray(const Vector& origin, const Vector& direction)
    : origin(origin), direction(direction) {}

__device__ Radium::Vector Radium::Ray::at(const double t) const {
  return origin + t * direction;
}