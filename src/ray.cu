#include <radium/ray.cuh>

__host__ __device__ Radium::Ray::Ray(const Vector& origin,
                                     const Vector& direction)
    : origin(origin), direction(direction) {}

__host__ __device__ Radium::Vector Radium::Ray::at(double t) const {
  return origin + direction * t;
}