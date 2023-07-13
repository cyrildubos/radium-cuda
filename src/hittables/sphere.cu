#include <radium/hittables/sphere.cuh>

__host__ Radium::Hittables::Sphere::Sphere(const Vector& center,
                                           const double radius,
                                           Material* material)
    : center(center), radius(radius), material(material) {}

__device__ bool Radium::Hittables::Sphere::hit(const Ray& ray,
                                               const double t_min,
                                               const double t_max,
                                               Hit& hit) const {
  const auto direction = ray.origin - center;

  const auto a = ray.direction.length_squared();
  const auto half_b = Vector::dot(direction, ray.direction);
  const auto c = direction.length_squared() - radius * radius;

  const auto d = half_b * half_b - a * c;

  if (d < 0.0)
    return false;

  const auto sqrt_d = std::sqrt(d);

  auto root = (-half_b - sqrt_d) / a;

  if (root < t_min || t_max < root) {
    root = (-half_b + sqrt_d) / a;

    if (root < t_min || t_max < root)
      return false;
  }

  hit.t = root;
  hit.position = ray.at(hit.t);
  hit.normal = (hit.position - center) / radius;
  hit.material = material;

  return true;
}