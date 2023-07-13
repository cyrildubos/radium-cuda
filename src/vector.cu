#include <radium/vector.cuh>

__host__ __device__ Radium::Vector::Vector() : x(0.0), y(0.0), z(0.0) {}

__host__ __device__ Radium::Vector::Vector(const double x, const double y,
                                           const double z)
    : x(x), y(y), z(z) {}

__host__ __device__ Radium::Vector Radium::Vector::random() {
  return Radium::Vector(random_double(), random_double(), random_double());
}

__host__ __device__ Radium::Vector Radium::Vector::random(const double min,
                                                          const double max) {
  return Radium::Vector(random_double(min, max),  //
                        random_double(min, max),  //
                        random_double(min, max)); //
}

__host__ __device__ Radium::Vector
Radium::Vector::random_in_sphere(const double radius = 1.0) {
  while (true) {
    auto position = random(-1.0, 1.0);

    if (position.length() < radius)
      return position;
  }
}

__host__ __device__ Radium::Vector Radium::Vector::random_unit() {
  return unit(random_in_sphere());
}

__host__ __device__ Radium::Vector Radium::Vector::sqrt(const Vector& v) {
  return Radium::Vector(std::sqrt(v.x), std::sqrt(v.y), std::sqrt(v.z));
}

__host__ __device__ Radium::Vector Radium::Vector::unit(const Vector& v) {
  return v / v.length();
}

__host__ __device__ double Radium::Vector::dot(const Vector& u,
                                               const Vector& v) {
  return u.x * v.x + u.y * v.y + u.z * v.z;
}

__host__ __device__ double Radium::Vector::length() const {
  return std::sqrt(length_squared());
}

__host__ __device__ double Radium::Vector::length_squared() const {
  return x * x + y * y + z * z;
}

__host__ __device__ bool Radium::Vector::near_zero() const {
  const auto value = 1e-8;

  return std::fabs(x) < value && std::fabs(y) < value && std::fabs(z) < value;
}
