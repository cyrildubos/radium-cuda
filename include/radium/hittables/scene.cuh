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
  std::vector<std::shared_ptr<Hittable>> hittables;

  __host__ Scene();

  __host__ void add(std::shared_ptr<Hittable>);

  __host__ void clear();

  __device__ virtual bool hit(const Ray&, double, double, Hit&) const override;
};
} // namespace Radium::Hittables

#endif