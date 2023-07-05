#include <radium/camera.cuh>

__host__ Radium::Camera::Camera(const double aspect_ratio) {
  position = Vector(0.0, 0.0, 0.0);

  auto focal_length = 1.0;

  auto height = 2.0;
  auto width = height * aspect_ratio;

  horizontal = Vector(width, 0.0, 0.0);
  vertical = Vector(0.0, height, 0.0);

  origin = position - Vector(0.0, 0.0, focal_length) //
           - horizontal / 2.0                        //
           + vertical / 2.0;                         //
}

__device__ Radium::Ray Radium::Camera::get_ray(const double u,
                                               const double v) const {
  return Ray(position,            //
             origin - position    //
                 + u * horizontal //
                 - v * vertical); //
}