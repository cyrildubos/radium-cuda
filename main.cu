#include <fstream>
#include <iostream>

#include <radium/camera.cuh>
#include <radium/image.cuh>

int main() {
  std::cout << "Hello, World!" << std::endl;

  const auto width = 16;
  const auto height = 9;

  const auto aspect_ratio = width / height;

  Radium::Camera camera(aspect_ratio);

  Radium::Image image(1920, 1080, camera);

  image.render();

  cudaDeviceSynchronize();

  image.save("image.ppm");

  return 0;
}