#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>

/*The MI macro is used to access an element in a matrix represented
  in linear form in a one-dimensional array.*/
#define MI(r,c,width) ((r)*(width)+(c))
#define PARAM_ABSOLUTE_INDEX(cam_index,param_index,number_of_params) ((number_of_params)*(cam_index)+(param_index))

  /*The CUBE macro is used to access an element in a represented cube
  in linear form in a one-dimensional array.*/
#define CUBE(x,y,z,width,height) (((z) * (width) * (height)) + (y) * (width) + (x))

// Utility functions
void computeStats(float gpu_time_ms, float mem_time_ms, int cam_width, int cam_height, int cam_number, int zPlanes, int window_size);
std::chrono::steady_clock::time_point start_cpu_timer();
double end_cpu_timer(std::chrono::steady_clock::time_point start);
cudaEvent_t start_cuda_timer();
float end_cuda_timer(cudaEvent_t start);


void wrap_plane_sweeping(float* result, const float* cam_h_K, const float* ref_h_K_inv, const float* cam_h_R, const float* ref_h_R_inv, const double* cam_h_t, const double* ref_h_t_inv, const int* h_YUV, int cam_number, float ZFar, float ZNear, int ZPlanes, int cam_width, int cam_height, int ref_index, int window);