#include <radium/vector.cuh>

__host__ __device__ Radium::Vector::Vector() : x(0.0), y(0.0), z(0.0) {}

__host__ __device__ Radium::Vector::Vector(const double x, const double y,
                                           const double z)
    : x(x), y(y), z(z) {}

__host__ __device__ double Radium::Vector::length() const {
  return std::sqrt(length_squared());
}

__host__ __device__ double Radium::Vector::length_squared() const {
  return x * x + y * y + z * z;
}

// __device__ Radium::Vector Radium::Vector::random() {
//   return Vector(random_double(), random_double(), random_double());
// }

// __device__ Radium::Vector Radium::Vector::random(double min, double max) {
//   return Vector(random_double(min, max),  //
//                 random_double(min, max),  //
//                 random_double(min, max)); //
// }

// __device__ Radium::Vector
// Radium::Vector::random_in_sphere(double radius = 1.0) {
//   while (true) {
//     auto position = random(-1.0, 1.0);

//     if (position.length() < radius)
//       return position;
//   }
// }

// __device__ Radium::Vector Radium::Vector::random_unit() {
//   return unit(random_in_sphere());
// }

__host__ __device__ bool Radium::Vector::near_zero() const {
  const auto value = 1e-8;

  return std::fabs(x) < value && std::fabs(y) < value && std::fabs(z) < value;
}

__host__ __device__ std::uint32_t Radium::Vector::to_color() const {
  const auto r = (std::uint32_t)(255 * x);
  const auto g = (std::uint32_t)(255 * y);
  const auto b = (std::uint32_t)(255 * z);

  return (r << 0) + (g << 8) + (b << 16);
}