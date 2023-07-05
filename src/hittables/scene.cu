#include <radium/hittables/scene.cuh>

__host__ void
Radium::Hittables::Scene::add(std::shared_ptr<Hittable> hittable) {
  hittables.push_back(hittable);
}

__host__ void Radium::Hittables::Scene::clear() { hittables.clear(); };

__device__ bool Radium::Hittables::Scene::hit(const Ray& ray, double t_min,
                                              double t_max, Hit& hit) const {
  auto has_hit = false;

  auto t = t_max;

  Hit tmp_hit;

  for (auto& hittable : hittables)
    if (hittable->hit(ray, t_min, t, tmp_hit)) {
      has_hit = true;

      t = tmp_hit.t;

      hit = tmp_hit;
    }

  return has_hit;
}
