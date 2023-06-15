cmake_minimum_required(VERSION 3.0.0)

project(radium-cuda VERSION 0.1.0 LANGUAGES CUDA)

add_library(Radium 
  src/camera.cu
  src/image.cu
  src/ray.cu
  src/vector.cu
)

target_include_directories(Radium PUBLIC include)


add_executable(main src/main.cu)

target_link_libraries(main PUBLIC Radium)