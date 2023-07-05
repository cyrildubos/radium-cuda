#ifndef RADIUM_HITTABLE_CUH
#define RADIUM_HITTABLE_CUH

#include "hit.cuh"
#include "ray.cuh"

namespace Radium {
class Hittable {
public:
  __device__ virtual bool hit(const Ray&, double, double, Hit&) const = 0;
};
} // namespace Radium

#endif
