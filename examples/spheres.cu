#include <radium/hittables/scene.cuh>
#include <radium/hittables/sphere.cuh>

#include <radium/materials/lambertian.cuh>

#include <radium/camera.cuh>
#include <radium/image.cuh>

int main() {
  Radium::Camera camera(16.0 / 9.0);

  const Radium::Materials::Lambertian red(Radium::Vector(0.1, 0.0, 0.0));
  const Radium::Materials::Lambertian green(Radium::Vector(0.0, 0.1, 0.0));
  const Radium::Materials::Lambertian blue(Radium::Vector(0.0, 0.0, 0.1));

  //   const Radium::Hittables::Sphere boule(Radium::Vector p(0.0, 0.0, -1.0),
  //   0.5,
  //                                         &blue);

  Radium::Hittables::Scene scene(0);
  //   scene.add(&boule);
  //   scene.add(std::make_shared<Radium::Hittables::Sphere>(
  //       Radium::Vector(0.0, -100.5, -1.0), 100.0, green));

  Radium::Image image(1280, 720, camera, scene);

  image.render();

  // cudaDeviceSynchronize();

  image.save("image.ppm");

  return 0;
}