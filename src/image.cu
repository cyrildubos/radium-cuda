#include <radium/image.cuh>

__host__ Radium::Image::Image(const std::size_t width, const std::size_t height,
                              const Camera& camera)
    : width(width), height(height), camera(camera) {
  cudaMallocManaged(&data, width * height * sizeof(Radium::Color));
}

__host__ Radium::Image::~Image() { cudaFree(data); }

__device__ Radium::Vector compute_color(const Radium::Ray& ray) {
  auto direction = unit(ray.direction);

  auto t = 0.5 * (direction.y + 1.0);

  return (1.0 - t) * Radium::Vector(1.0, 1.0, 1.0) +
         t * Radium::Vector(0.2, 0.5, 1.0);
}

__global__ void compute(Radium::Color* data, const std::size_t width,
                        const std::size_t height, const Radium::Camera camera) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= width || j >= height)
    return;

  auto u = double(i) / double(width);
  auto v = double(j) / double(height);

  auto ray = camera.get_ray(u, v);

  auto color = compute_color(ray);

  auto index = i + j * width;

  data[index] = Radium::Color(color);
}

__host__ void Radium::Image::render() {
  const auto block_width = 32;
  const auto block_height = 32;

  const dim3 grid_size(width / block_width + 1, height / block_height + 1);
  const dim3 block_size(block_width, block_height);

  compute<<<grid_size, block_size>>>(data, width, height, camera);
}

__host__ void Radium::Image::save(const std::string path) {
  std::ofstream file_stream(path);

  if (!file_stream.is_open())
    throw std::runtime_error("failed to open file");

  file_stream << "P3\n" << width << ' ' << height << "\n255\n";

  for (auto j = 0; j < height; j++)
    for (auto i = 0; i < width; i++) {
      auto index = i + j * width;

      // auto r = 0xFF & (data[index] >> 0);
      // auto g = 0xFF & (data[index] >> 8);
      // auto b = 0xFF & (data[index] >> 16);

      file_stream << data[index] << '\n';
    }

  file_stream << '\n';

  file_stream.close();
}