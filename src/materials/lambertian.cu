#include <radium/materials/lambertian.cuh>

__host__ Radium::Materials::Lambertian::Lambertian(const Vector& albedo)
    : albedo(albedo) {}

__device__ bool Radium::Materials::Lambertian::scatter(const Ray& ray,
                                                       const Hit& hit,
                                                       Vector& attenuation,
                                                       Ray& scattered) const {
  auto direction = hit.normal + Radium::Vector::random_unit();

  if (direction.near_zero())
    direction += hit.normal;

  scattered = Ray(hit.position, direction);

  attenuation = albedo;

  return true;
}