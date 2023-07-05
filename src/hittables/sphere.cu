#include <radium/hittables/sphere.cuh>

__host__ Radium::Hittables::Sphere::Sphere(const Vector& center, double radius)
    : center(center), radius(radius) {}

__device__ bool Radium::Hittables::Sphere::hit(const Ray& ray, double t_min,
                                               double t_max, Hit& hit) const {
  auto direction = ray.origin - center;

  auto a = ray.direction.length_squared();
  auto half_b = dot(direction, ray.direction);
  auto c = direction.length_squared() - radius * radius;

  auto discriminant = half_b * half_b - a * c;

  if (discriminant < 0.0)
    return false;

  auto root = (-half_b - std::sqrt(discriminant)) / a;

  if (root < t_min || t_max < root) {
    root = (-half_b + std::sqrt(discriminant)) / a;

    if (root < t_min || t_max < root)
      return false;
  }

  hit.t = root;
  hit.position = ray.at(hit.t);
  hit.normal = (hit.position - center) / radius;

  return true;
}