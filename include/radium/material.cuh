#ifndef RADIUM_MATERIAL_CUH
#define RADIUM_MATERIAL_CUH

#include "hit.cuh"
#include "ray.cuh"
#include "vector.cuh"

namespace Radium {
class Material {
public:
  __device__ virtual bool scatter(const Ray&, const Hit&, Vector&,
                                  Ray&) const = 0;
};
} // namespace Radium

#endif