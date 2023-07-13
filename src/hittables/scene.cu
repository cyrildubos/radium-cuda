#include <radium/hittables/scene.cuh>

__host__ Radium::Hittables::Scene::Scene(const std::size_t max_size)
    : max_size(max_size),
      hittables((Hittable**)malloc(max_size * sizeof(Hittable*))), size(0) {
  ;
}

__host__ void Radium::Hittables::Scene::add(Hittable* hittable) {
  if (size < max_size)
    hittables[size++] = hittable;
}

__host__ void Radium::Hittables::Scene::clear() { size = 0; };

__device__ bool Radium::Hittables::Scene::hit(const Ray& ray,
                                              const double t_min,
                                              const double t_max,
                                              Hit& hit) const {
  auto has_hit = false;

  auto t = t_max;

  Hit tmp_hit;

  // for (auto& hittable : hittables)
  for (auto i = 0; i < size; ++i)
    if (hittables[i]->hit(ray, t_min, t, tmp_hit)) {
      has_hit = true;

      t = tmp_hit.t;

      hit = tmp_hit;
    }

  return has_hit;
}
