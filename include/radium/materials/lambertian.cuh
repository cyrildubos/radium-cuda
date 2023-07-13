#ifndef RADIUM_MATERIALS_LAMBERTIAN_CUH
#define RADIUM_MATERIALS_LAMBERTIAN_CUH

#include "../material.cuh"

namespace Radium::Materials {
class Lambertian : public Material {
public:
  const Vector albedo;

  __host__ Lambertian(const Vector&);

  __device__ virtual bool scatter(const Ray&, const Hit&, Vector&,
                                  Ray&) const override;
};
} // namespace Radium::Materials

#endif