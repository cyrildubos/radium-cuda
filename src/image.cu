#include <radium/image.cuh>

__host__ Radium::Image::Image(const std::size_t width, const std::size_t height,
                              const Camera& camera)
    : width(width), height(height), camera(camera) {
  cudaMallocManaged(&data, 3 * width * height * sizeof(float));
}

__host__ Radium::Image::~Image() { cudaFree(data); }

__global__ void compute(float* data, const std::size_t width,
                        const std::size_t height,
                        const Radium::Camera& camera) {
  auto i = blockIdx.x * blockDim.x + threadIdx.x;
  auto j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i >= width || j >= height)
    return;

  auto u = float(i) / float(width);
  auto v = float(j) / float(height);

  // camera.get_ray(u, v);

  auto index = i + j * width;

  data[3 * index + 0] = u;
  data[3 * index + 1] = v;
  data[3 * index + 2] = 0.0f;
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

      auto r = data[3 * index + 0];
      auto g = data[3 * index + 1];
      auto b = data[3 * index + 2];

      file_stream << int(255 * r) << ' '   //
                  << int(255 * g) << ' '   //
                  << int(255 * b) << '\n'; //
    }

  file_stream << '\n';

  file_stream.close();
}