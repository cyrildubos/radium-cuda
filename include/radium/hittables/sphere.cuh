#ifndef RADIUM_HITTABLES_SPHERE_HPP
#define RADIUM_HITTABLES_SPHERE_HPP

#include <memory>

#include "../hit.cuh"
#include "../hittable.cuh"
#include "../ray.cuh"
#include "../vector.cuh"

namespace Radium::Hittables {
class Sphere : public Hittable {
public:
  const Vector center;
  const double radius;

  __host__ Sphere(const Vector&, double);

  __device__ virtual bool hit(const Ray&, double, double, Hit&) const override;
};
} // namespace Radium::Hittables

#endif