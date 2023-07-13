#ifndef RADIUM_UTILITIES_CUH
#define RADIUM_UTILITIES_CUH

#include <cstdlib>

namespace Radium {
// TODO
__host__ __device__ inline double random_double() {
  // return rand() / (RAND_MAX + 1.0);
  return 0.5;
}

__host__ __device__ inline double random_double(double min, double max) {
  return min + (max - min) * random_double();
}

//  inline double clamp(double x, double min, double max) {
//   if (x < min)
//     return min;

//   if (x > max)
//     return max;

//   return x;
// }
} // namespace Radium

#endif