cmake_minimum_required(VERSION 3.0.0)

project(radium-cuda VERSION 0.1.0 LANGUAGES CUDA)


# library

add_library(radium 
  src/hittables/scene.cu
  src/hittables/sphere.cu

  src/materials/lambertian.cu
  
  src/camera.cu
  src/image.cu
  src/pixel.cu
  src/ray.cu
  src/vector.cu
)

target_include_directories(radium PUBLIC include)

# set_property(TARGET radium PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# executable

add_executable(spheres 
  examples/spheres.cu
)

target_link_libraries(spheres PUBLIC radium)