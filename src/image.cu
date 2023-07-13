#include <radium/image.cuh>

Radium::Image::Image(const std::size_t width, const std::size_t height,
                     const Camera& camera, const Hittables::Scene& scene)
    : width(width), height(height), camera(camera), scene(scene) {
  cudaMallocManaged(&pixels, width * height * sizeof(Radium::Pixel));
}

Radium::Image::~Image() { cudaFree(pixels); }

__device__ Radium::Vector compute_color(const Radium::Ray& ray,
                                        const Radium::Hittables::Scene& scene,
                                        const std::size_t depth) {
  if (depth <= 0)
    return Radium::Vector();

  const auto direction = Radium::Vector::unit(ray.direction);

  const auto t_min = 0.001;
  const auto t_max = 100.0;

  Radium::Hit hit;

  if (scene.hit(ray, t_min, t_max, hit)) {
    Radium::Vector attenuation;
    Radium::Ray scattered;

    // TODO: std::_shared_ptr_access
    if (hit.material->scatter(ray, hit, attenuation, scattered))
      return attenuation * compute_color(scattered, scene, depth - 1);

    return Radium::Vector();
  }

  const auto t = 0.5 * (direction.y + 1.0);

  return (1.0 - t) * Radium::Vector(1.0, 1.0, 1.0) +
         t * Radium::Vector(0.2, 0.5, 1.0);
}

__global__ void compute_pixels(Radium::Pixel* pixels, const std::size_t width,
                               const std::size_t height,
                               const Radium::Camera camera,
                               const Radium::Hittables::Scene& scene) {
  const auto i = blockIdx.x * blockDim.x + threadIdx.x;
  const auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= width || j >= height)
    return;

  const auto sample_per_pixel = 50;
  const auto depth = 10;

  Radium::Vector color;

  for (auto s = 0; s < sample_per_pixel; ++s) {
    const auto u = (double(i) + Radium::random_double()) / double(width);
    const auto v = (double(j) + Radium::random_double()) / double(height);

    const auto ray = camera.get_ray(u, v);

    color += compute_color(ray, scene, depth); // TODO
  }

  pixels[i + j * width] =
      Radium::Pixel(Radium::Vector::sqrt(color / sample_per_pixel));
}

// void compute_pixels(Radium::Pixel* pixels, const std::size_t i,
//                     const std::size_t j, const std::size_t width,
//                     const std::size_t height, const Radium::Camera camera,
//                     const Radium::Hittables::Scene& scene) {
//   const auto sample_per_pixel = 50;
//   const auto depth = 10;

//   Radium::Vector color;

//   for (auto s = 0; s < sample_per_pixel; ++s) {
//     const auto u = (double(i) + Radium::random_double()) / double(width);
//     const auto v = (double(j) + Radium::random_double()) / double(height);

//     const auto ray = camera.get_ray(u, v);

//     color += compute_color(ray, scene, depth);
//   }

//   pixels[i + j * width] =
//       Radium::Pixel(Radium::Vector::sqrt(color / sample_per_pixel));
// }

void Radium::Image::render() {
  const auto block_width = 32;
  const auto block_height = 32;

  const dim3 grid_size(width / block_width + 1, height / block_height + 1);
  const dim3 block_size(block_width, block_height);

  compute_pixels<<<grid_size, block_size>>>(pixels, width, height, camera,
                                            scene);
}

// void Radium::Image::render() {
//   for (auto i = 0; i < width; i++)
//     for (auto j = 0; j < height; j++)
//       compute_pixels(pixels, i, j, width, height, camera, scene);
// }

void Radium::Image::save(const std::string path) const {
  std::ofstream file_stream(path);

  if (!file_stream.is_open())
    throw std::runtime_error("failed to open file");

  file_stream << "P3\n" << width << ' ' << height << "\n255\n";

  for (auto j = 0; j < height; j++)
    for (auto i = 0; i < width; i++)
      file_stream << pixels[i + j * width] << '\n';

  file_stream << '\n';

  file_stream.close();
}