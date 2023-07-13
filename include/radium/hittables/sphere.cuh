#ifndef RADIUM_HITTABLES_SPHERE_HPP
#define RADIUM_HITTABLES_SPHERE_HPP

#include <memory>

#include "../hit.cuh"
#include "../hittable.cuh"
#include "../material.cuh"
#include "../ray.cuh"
#include "../vector.cuh"

namespace Radium::Hittables {
class Sphere : public Hittable {
public:
  const Vector center;
  const double radius;
  Material* material;

  __host__ Sphere(const Vector&, const double, Material*);

  __device__ virtual bool hit(const Ray&, const double, const double,
                              Hit&) const override;
};
} // namespace Radium::Hittables

#endif