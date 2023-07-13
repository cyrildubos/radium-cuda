#ifndef RADIUM_VECTOR_CUH
#define RADIUM_VECTOR_CUH

#include <cmath>
#include <iostream>

#include "utilities.cuh"

namespace Radium {
class Vector {
public:
  double x;
  double y;
  double z;

  __host__ __device__ Vector();
  __host__ __device__ Vector(const double, const double, const double);

  __host__ __device__ static Vector random();
  __host__ __device__ static Vector random(const double, const double);
  __host__ __device__ static Vector random_in_sphere(const double);
  // __host__ __device__ static Vector random_in_hemisphere(const double);
  __host__ __device__ static Vector random_unit();

  __host__ __device__ static Vector sqrt(const Vector&);
  __host__ __device__ static Vector unit(const Vector&);
  __host__ __device__ static double dot(const Vector&, const Vector&);

  __host__ __device__ double length() const;
  __host__ __device__ double length_squared() const;

  __host__ __device__ bool near_zero() const;
};

__host__ __device__ inline Vector operator+(const Vector& u, const Vector& v) {
  return Vector(u.x + v.x, u.y + v.y, u.z + v.z);
}

__host__ __device__ inline Vector operator+(const Vector& v, const double t) {
  return Vector(v.x + t, v.y + t, v.z + t);
}

__host__ __device__ inline Vector operator+(const double t, const Vector& v) {
  return Vector(t + v.x, t + v.y, t + v.z);
}

__host__ __device__ inline Vector operator-(const Vector& u, const Vector& v) {
  return Vector(u.x - v.x, u.y - v.y, u.z - v.z);
}

__host__ __device__ inline Vector operator*(const Vector& u, const Vector& v) {
  return Vector(u.x * v.x, u.y * v.y, u.z * v.z);
}

__host__ __device__ inline Vector operator*(const Vector& v, double t) {
  return Vector(v.x * t, v.y * t, v.z * t);
}

__host__ __device__ inline Vector operator*(double t, const Vector& v) {
  return Vector(t * v.x, t * v.y, t * v.z);
}

__host__ __device__ inline Vector operator/(const Vector& v, double t) {
  return Vector(v.x / t, v.y / t, v.z / t);
}

// TODO
__host__ __device__ inline Vector operator+=(Vector& u, const Vector& v) {
  return u = Vector(u.x + v.x, u.y + v.y, u.z + v.z);
}

// inline std::ostream& operator<<(std::ostream& s, const Vector& v) {
//   return s << v.x << ' ' << v.y << ' ' << v.z;
// }

// inline Vector cross(const Vector& u, const Vector& v) {
//   return Vector(u.y * v.z - u.z * v.y,  //
//                 u.z * v.x - u.x * v.z,  //
//                 u.x * v.y - u.y * v.x); //
// }

// inline Vector reflect(const Vector& v, const Vector& n) {
//   return v - 2 * dot(v, n) * n;
// }
} // namespace Radium

#endif