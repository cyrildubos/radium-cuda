#ifndef RADIUM_HITTABLES_SCENE_CUH
#define RADIUM_HITTABLES_SCENE_CUH

#include <memory>
#include <vector>

#include "../hit.cuh"
#include "../hittable.cuh"
#include "../ray.cuh"

namespace Radium::Hittables {
class Scene : public Hittable {
public:
  const std::size_t max_size;

  Hittable** hittables;

  std::size_t size;

  __host__ Scene(const std::size_t);

  __host__ void add(Hittable*);

  __host__ void clear();

  __device__ virtual bool hit(const Ray&, const double, const double,
                              Hit&) const override;
};
} // namespace Radium::Hittables

#endif