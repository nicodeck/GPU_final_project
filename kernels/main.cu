#include "main.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>

#include <cstdio>
#include <chrono>
#include <iostream>

//constant use for the warmup
#define BLOCKSIZE 16
const unsigned int N = 64 * BLOCKSIZE;

#define N_THREADS_PER_BLOCK 512
#define N_BLOCKS 81
#define N_PIXELS_PER_THREAD 10

#define SHARED_MEMORY_BLOCK_HEIGHT 54
#define SHARED_MEMORY_BLOCK_WIDTH 32
#define WINDOW 5
#define HALF_WINDOW(window) ((window - 1)/2)
#define CAM_NUMBER 4
#define CAM_HEIGHT 1080
#define CAM_WIDTH 1920

#define LOCAL_COORDINATES(cam_index, row, col) (((cam_index) * (SHARED_MEMORY_BLOCK_HEIGHT + WINDOW - 1) * (SHARED_MEMORY_BLOCK_WIDTH + WINDOW - 1) + (((row) % SHARED_MEMORY_BLOCK_HEIGHT) + HALF_WINDOW(WINDOW)) * (SHARED_MEMORY_BLOCK_WIDTH + WINDOW - 1) + (((col) % SHARED_MEMORY_BLOCK_WIDTH) + HALF_WINDOW(WINDOW))))

#define YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, localRow, localCol) (cam_index * CAM_HEIGHT * CAM_WIDTH + blockVerticalIndex * SHARED_MEMORY_BLOCK_HEIGHT * SHARED_MEMORY_BLOCK_WIDTH + blockHorizontalIndex * SHARED_MEMORY_BLOCK_WIDTH + SHARED_MEMORY_BLOCK_HEIGHT * SHARED_MEMORY_BLOCK_WIDTH * localRow + localCol)

// Constant memory for the camera parameters
#define K_SIZE 9
#define R_SIZE 9
#define t_SIZE 3

__constant__ float const_cam_h_K[CAM_NUMBER * K_SIZE];
__constant__ float const_ref_h_K_inv[K_SIZE];
__constant__ float const_cam_h_R[CAM_NUMBER * R_SIZE];
__constant__ float const_ref_h_R_inv[R_SIZE];
__constant__ double const_cam_h_t[CAM_NUMBER * t_SIZE];
__constant__ double const_ref_h_t_inv[t_SIZE];

#define PRINT_STATS

#define CHK(code) \
do { \
    if ((code) != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s %s %i\n", \
                        cudaGetErrorString((code)), __FILE__, __LINE__); \
        goto Error; \
    } \
} while (0)



__device__ bool pixel_coordinates_are_valid(int blockVerticalIndex, int blockHorizontalIndex, int localRow, int localCol) {
	int row = SHARED_MEMORY_BLOCK_HEIGHT * blockVerticalIndex + localRow;
	int col = SHARED_MEMORY_BLOCK_WIDTH * blockHorizontalIndex + localCol;

	return (row >= 0 && row < CAM_HEIGHT && col >= 0 && col < CAM_WIDTH);
}

//NAIVE VERSION (USING DOUBLES)
__global__ void dev_plane_sweeping_naive(
	float* result, const float* cam_h_K, const float* ref_h_K_inv, const float* cam_h_R, const float* ref_h_R_inv, const double* cam_h_t, const double* ref_h_t_inv, const int* YUV, int cam_number, float ZFar, float ZNear, int ZPlanes, int cam_width, int cam_height, int ref_index, int window
) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;


	for (int pixel_padding = 0; pixel_padding < N_PIXELS_PER_THREAD; ++pixel_padding) {

		if (thread_index == 0) {
			printf("Pixel %d/%d\n", pixel_padding, N_PIXELS_PER_THREAD);
		}
		// a thread processes consecutive pixels
		int pixel_index = N_PIXELS_PER_THREAD * thread_index + pixel_padding;



		int y = pixel_index / cam_width;
		int x = pixel_index % cam_width;


		if (x >= cam_width || y >= cam_height)
			return;


		// Compute values that do not rely on z
		double X_ref = (ref_h_K_inv[0] * x + ref_h_K_inv[1] * y + ref_h_K_inv[2]);
		double Y_ref = (ref_h_K_inv[3] * x + ref_h_K_inv[4] * y + ref_h_K_inv[5]);
		double Z_ref = (ref_h_K_inv[6] * x + ref_h_K_inv[7] * y + ref_h_K_inv[8]);

		double X_b = ref_h_R_inv[0] * X_ref + ref_h_R_inv[1] * Y_ref + ref_h_R_inv[2] * Z_ref;
		double Y_b = ref_h_R_inv[3] * X_ref + ref_h_R_inv[4] * Y_ref + ref_h_R_inv[5] * Z_ref;
		double Z_b = ref_h_R_inv[6] * X_ref + ref_h_R_inv[7] * Y_ref + ref_h_R_inv[8] * Z_ref;

		for (int zi = 0; zi < ZPlanes; zi++) {

			float min_cost = 255.0f; // initialize cost at maximum

			// (i) calculate projection index
			double z = ZNear * ZFar / (ZNear + (((double)zi / (double)ZPlanes) * (ZFar - ZNear)));

			double X = X_b * z - ref_h_t_inv[0];
			double Y = Y_b * z - ref_h_t_inv[1];
			double Z = Z_b * z - ref_h_t_inv[2];

			for (int cam_index = 0; cam_index < cam_number; cam_index++) {
				if (cam_index == ref_index)
					continue;

				double X_proj = cam_h_R[MI(cam_index, 0, 9)] * X + cam_h_R[MI(cam_index, 1, 9)] * Y + cam_h_R[MI(cam_index, 2, 9)] * Z - cam_h_t[MI(cam_index, 0, 3)];
				double Y_proj = cam_h_R[MI(cam_index, 3, 9)] * X + cam_h_R[MI(cam_index, 4, 9)] * Y + cam_h_R[MI(cam_index, 5, 9)] * Z - cam_h_t[MI(cam_index, 1, 3)];
				double Z_proj = cam_h_R[MI(cam_index, 6, 9)] * X + cam_h_R[MI(cam_index, 7, 9)] * Y + cam_h_R[MI(cam_index, 8, 9)] * Z - cam_h_t[MI(cam_index, 2, 3)];

				double XZ_proj = X_proj / Z_proj;
				double YZ_proj = Y_proj / Z_proj;

				int x_proj = (int)(cam_h_K[MI(cam_index, 0, 9)] * XZ_proj + cam_h_K[MI(cam_index, 1, 9)] * YZ_proj + cam_h_K[MI(cam_index, 2, 9)]);
				int y_proj = (int)(cam_h_K[MI(cam_index, 3, 9)] * XZ_proj + cam_h_K[MI(cam_index, 4, 9)] * YZ_proj + cam_h_K[MI(cam_index, 5, 9)]);

				// Calculate cost in a window
				int radius = window / 2;
				int kmin = -radius, kmax = radius, lmin = -radius, lmax = radius;
				if (x_proj + lmin < 0) lmin = -x_proj;
				if (x + lmin < 0) lmin = -x;
				if (y_proj + kmin < 0) kmin = -y_proj;
				if (y + kmin < 0) kmin = -y;
				if (x_proj + lmax >= cam_width) lmax = cam_width - x_proj - 1;
				if (x + lmax >= cam_width) lmax = cam_width - x - 1;
				if (y_proj + kmax >= cam_height) kmax = cam_height - y_proj - 1;
				if (y + kmax >= cam_height) kmax = cam_height - y - 1;

				float cost = 0.0f;
				float cc = 0.0f;
				for (int k = kmin; k <= kmax; k++) {
					for (int l = lmin; l <= lmax; l++) {
						int diff = YUV[CUBE(x_proj + l, y_proj + k, cam_index, cam_width, cam_height)] - YUV[CUBE(x + l, y + k, ref_index, cam_width, cam_height)];
						diff = diff < 0 ? -diff : diff;
						cost += diff;
						cc += 1.0f;
					}
				}
				cost /= cc;

				min_cost = fminf(min_cost, cost);
			}
			result[CUBE(x, y, zi, cam_width, cam_height)] = min_cost;
		}
	}
}

//CHANGING TYPES INTO FLOAT 
__global__ void dev_plane_sweeping_naive_types(
	float* result, const float* cam_h_K, const float* ref_h_K_inv, const float* cam_h_R, const float* ref_h_R_inv, const double* cam_h_t, const double* ref_h_t_inv, const int* YUV, int cam_number, float ZFar, float ZNear, int ZPlanes, int cam_width, int cam_height, int ref_index, int window
) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;


	for (int pixel_padding = 0; pixel_padding < N_PIXELS_PER_THREAD; ++pixel_padding) {

		if (thread_index == 0) {
			printf("Pixel %d/%d\n", pixel_padding, N_PIXELS_PER_THREAD);
		}
		// a thread processes consecutive pixels
		int pixel_index = N_PIXELS_PER_THREAD * thread_index + pixel_padding;



		int y = pixel_index / cam_width;
		int x = pixel_index % cam_width;


		if (x >= cam_width || y >= cam_height)
			return;


		// Compute values that do not rely on z
		float X_ref = (ref_h_K_inv[0] * x + ref_h_K_inv[1] * y + ref_h_K_inv[2]);
		float Y_ref = (ref_h_K_inv[3] * x + ref_h_K_inv[4] * y + ref_h_K_inv[5]);
		float Z_ref = (ref_h_K_inv[6] * x + ref_h_K_inv[7] * y + ref_h_K_inv[8]);

		float X_b = ref_h_R_inv[0] * X_ref + ref_h_R_inv[1] * Y_ref + ref_h_R_inv[2] * Z_ref;
		float Y_b = ref_h_R_inv[3] * X_ref + ref_h_R_inv[4] * Y_ref + ref_h_R_inv[5] * Z_ref;
		float Z_b = ref_h_R_inv[6] * X_ref + ref_h_R_inv[7] * Y_ref + ref_h_R_inv[8] * Z_ref;

		for (int zi = 0; zi < ZPlanes; zi++) {

			float min_cost = 255.0f; // initialize cost at maximum

			// (i) calculate projection index
			float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
			float X = X_b * z - ref_h_t_inv[0];
			float Y = Y_b * z - ref_h_t_inv[1];
			float Z = Z_b * z - ref_h_t_inv[2];

			for (int cam_index = 0; cam_index < cam_number; cam_index++) {
				if (cam_index == ref_index)
					continue;

				float X_proj = cam_h_R[MI(cam_index, 0, 9)] * X + cam_h_R[MI(cam_index, 1, 9)] * Y + cam_h_R[MI(cam_index, 2, 9)] * Z - cam_h_t[MI(cam_index, 0, 3)];
				float Y_proj = cam_h_R[MI(cam_index, 3, 9)] * X + cam_h_R[MI(cam_index, 4, 9)] * Y + cam_h_R[MI(cam_index, 5, 9)] * Z - cam_h_t[MI(cam_index, 1, 3)];
				float Z_proj = cam_h_R[MI(cam_index, 6, 9)] * X + cam_h_R[MI(cam_index, 7, 9)] * Y + cam_h_R[MI(cam_index, 8, 9)] * Z - cam_h_t[MI(cam_index, 2, 3)];

				float XZ_proj = X_proj / Z_proj;
				float YZ_proj = Y_proj / Z_proj;

				int x_proj = (int)(cam_h_K[MI(cam_index, 0, 9)] * XZ_proj + cam_h_K[MI(cam_index, 1, 9)] * YZ_proj + cam_h_K[MI(cam_index, 2, 9)]);
				int y_proj = (int)(cam_h_K[MI(cam_index, 3, 9)] * XZ_proj + cam_h_K[MI(cam_index, 4, 9)] * YZ_proj + cam_h_K[MI(cam_index, 5, 9)]);

				// Calculate cost in a window
				int radius = window / 2;
				int kmin = -radius, kmax = radius, lmin = -radius, lmax = radius;
				if (x_proj + lmin < 0) lmin = -x_proj;
				if (x + lmin < 0) lmin = -x;
				if (y_proj + kmin < 0) kmin = -y_proj;
				if (y + kmin < 0) kmin = -y;
				if (x_proj + lmax >= cam_width) lmax = cam_width - x_proj - 1;
				if (x + lmax >= cam_width) lmax = cam_width - x - 1;
				if (y_proj + kmax >= cam_height) kmax = cam_height - y_proj - 1;
				if (y + kmax >= cam_height) kmax = cam_height - y - 1;

				float cost = 0.0f;
				float cc = 0.0f;
				for (int k = kmin; k <= kmax; k++) {
					for (int l = lmin; l <= lmax; l++) {
						int diff = YUV[CUBE(x_proj + l, y_proj + k, cam_index, cam_width, cam_height)] - YUV[CUBE(x + l, y + k, ref_index, cam_width, cam_height)];
						diff = diff < 0 ? -diff : diff;
						cost += diff;
						cc += 1.0f;
					}
				}
				cost /= cc;

				min_cost = fminf(min_cost, cost);
			}
			result[CUBE(x, y, zi, cam_width, cam_height)] = min_cost;
		}
	}
}

//CHANGING THE ORDER OF THE PIXELS
__global__ void dev_plane_sweeping_change_pixel_order(
	float* result, const float* cam_h_K, const float* ref_h_K_inv, const float* cam_h_R, const float* ref_h_R_inv, const double* cam_h_t, const double* ref_h_t_inv, const int* YUV, int cam_number, float ZFar, float ZNear, int ZPlanes, int cam_width, int cam_height, int ref_index, int window
) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	// threads are divided in blocks of N_PIXELS_PER_THREAD ^ 2
	int virtualBlockIndex = thread_index / N_PIXELS_PER_THREAD;
	int virtualThreadIndex = thread_index % N_PIXELS_PER_THREAD;



	for (int pixel_padding = 0; pixel_padding < N_PIXELS_PER_THREAD; ++pixel_padding) {

		if (thread_index == 0) {
			printf("Pixel %d/%d\n", pixel_padding + 1, N_PIXELS_PER_THREAD);
		}

		// threads of the same virtual block process consecutive pixels of the block at the same time
		int pixel_index = N_PIXELS_PER_THREAD * N_PIXELS_PER_THREAD * virtualBlockIndex + N_PIXELS_PER_THREAD * pixel_padding + virtualThreadIndex;



		int y = pixel_index / cam_width;
		int x = pixel_index % cam_width;


		if (x >= cam_width || y >= cam_height)
			return;


		// Compute values that do not rely on z
		float X_ref = (ref_h_K_inv[0] * x + ref_h_K_inv[1] * y + ref_h_K_inv[2]);
		float Y_ref = (ref_h_K_inv[3] * x + ref_h_K_inv[4] * y + ref_h_K_inv[5]);
		float Z_ref = (ref_h_K_inv[6] * x + ref_h_K_inv[7] * y + ref_h_K_inv[8]);

		float X_b = ref_h_R_inv[0] * X_ref + ref_h_R_inv[1] * Y_ref + ref_h_R_inv[2] * Z_ref;
		float Y_b = ref_h_R_inv[3] * X_ref + ref_h_R_inv[4] * Y_ref + ref_h_R_inv[5] * Z_ref;
		float Z_b = ref_h_R_inv[6] * X_ref + ref_h_R_inv[7] * Y_ref + ref_h_R_inv[8] * Z_ref;

		for (int zi = 0; zi < ZPlanes; zi++) {

			float min_cost = 255.0f; // initialize cost at maximum

			// (i) calculate projection index
			float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
			float X = X_b * z - ref_h_t_inv[0];
			float Y = Y_b * z - ref_h_t_inv[1];
			float Z = Z_b * z - ref_h_t_inv[2];

			for (int cam_index = 0; cam_index < cam_number; cam_index++) {
				if (cam_index == ref_index)
					continue;

				float X_proj = cam_h_R[MI(cam_index, 0, 9)] * X + cam_h_R[MI(cam_index, 1, 9)] * Y + cam_h_R[MI(cam_index, 2, 9)] * Z - cam_h_t[MI(cam_index, 0, 3)];
				float Y_proj = cam_h_R[MI(cam_index, 3, 9)] * X + cam_h_R[MI(cam_index, 4, 9)] * Y + cam_h_R[MI(cam_index, 5, 9)] * Z - cam_h_t[MI(cam_index, 1, 3)];
				float Z_proj = cam_h_R[MI(cam_index, 6, 9)] * X + cam_h_R[MI(cam_index, 7, 9)] * Y + cam_h_R[MI(cam_index, 8, 9)] * Z - cam_h_t[MI(cam_index, 2, 3)];

				float XZ_proj = X_proj / Z_proj;
				float YZ_proj = Y_proj / Z_proj;

				int x_proj = (int)(cam_h_K[MI(cam_index, 0, 9)] * XZ_proj + cam_h_K[MI(cam_index, 1, 9)] * YZ_proj + cam_h_K[MI(cam_index, 2, 9)]);
				int y_proj = (int)(cam_h_K[MI(cam_index, 3, 9)] * XZ_proj + cam_h_K[MI(cam_index, 4, 9)] * YZ_proj + cam_h_K[MI(cam_index, 5, 9)]);

				// Calculate cost in a window
				int radius = window / 2;
				int kmin = -radius, kmax = radius, lmin = -radius, lmax = radius;
				if (x_proj + lmin < 0) lmin = -x_proj;
				if (x + lmin < 0) lmin = -x;
				if (y_proj + kmin < 0) kmin = -y_proj;
				if (y + kmin < 0) kmin = -y;
				if (x_proj + lmax >= cam_width) lmax = cam_width - x_proj - 1;
				if (x + lmax >= cam_width) lmax = cam_width - x - 1;
				if (y_proj + kmax >= cam_height) kmax = cam_height - y_proj - 1;
				if (y + kmax >= cam_height) kmax = cam_height - y - 1;

				float cost = 0.0f;
				float cc = 0.0f;
				for (int k = kmin; k <= kmax; k++) {
					for (int l = lmin; l <= lmax; l++) {
						int diff = YUV[CUBE(x_proj + l, y_proj + k, cam_index, cam_width, cam_height)] - YUV[CUBE(x + l, y + k, ref_index, cam_width, cam_height)];
						diff = diff < 0 ? -diff : diff;
						cost += diff;
						cc += 1.0f;
					}
				}
				cost /= cc;

				min_cost = fminf(min_cost, cost);
			}
			result[CUBE(x, y, zi, cam_width, cam_height)] = min_cost;
		}
	}
}

//SHARED MEMORY (TRY)
__global__ void dev_plane_sweeping_shared_memory(
	float* result, const float* cam_h_K, const float* ref_h_K_inv, const float* cam_h_R, const float* ref_h_R_inv, const double* cam_h_t, const double* ref_h_t_inv, const int* YUV, int cam_number, float ZFar, float ZNear, int ZPlanes, int cam_width, int cam_height, int ref_index, int window
) {
	__shared__ int shared_block[(SHARED_MEMORY_BLOCK_HEIGHT + WINDOW - 1) * (SHARED_MEMORY_BLOCK_WIDTH + WINDOW - 1) * CAM_NUMBER];

	for (int i = 0; i < (SHARED_MEMORY_BLOCK_HEIGHT + WINDOW - 1) * (SHARED_MEMORY_BLOCK_WIDTH + WINDOW - 1) * CAM_NUMBER; i++) {
		shared_block[i] = 0;
	}

	const int numberOfSMBlocksInHeight = cam_height / SHARED_MEMORY_BLOCK_HEIGHT;
	const int numberOfSMBlocksInWidth = cam_width / SHARED_MEMORY_BLOCK_WIDTH;

	const int threadLocalRow = blockIdx.x;
	const int threadLocalCol = threadIdx.x;

	for (int blockVerticalIndex = 0; blockVerticalIndex < numberOfSMBlocksInHeight; blockVerticalIndex++) {
		for (int blockHorizontalIndex = 0; blockHorizontalIndex < numberOfSMBlocksInWidth; blockHorizontalIndex++) {

			if (threadLocalCol == 0 && threadLocalRow == 0) {
				printf("Block %d/%d\n", blockVerticalIndex * cam_width / SHARED_MEMORY_BLOCK_WIDTH + blockHorizontalIndex, cam_width / SHARED_MEMORY_BLOCK_WIDTH * cam_height / SHARED_MEMORY_BLOCK_HEIGHT);
			}

			// fill shared memory
			for (int cam_index = 0; cam_index < cam_number; cam_index++) {

				shared_block[MI(threadLocalRow + HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window), SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow, threadLocalCol)];

				// top rows
				if (threadLocalRow < HALF_WINDOW(window)) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol)) {
						shared_block[MI(threadLocalRow, threadLocalCol + HALF_WINDOW(window), SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol)];
					}
				}

				// bottom rows
				if (threadLocalRow >= SHARED_MEMORY_BLOCK_HEIGHT) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow + HALF_WINDOW(window), threadLocalCol)) {
						shared_block[MI(threadLocalRow + 2 * HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window), SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow + HALF_WINDOW(window), threadLocalCol)];
					}
				}

				// left columns
				if (threadLocalCol < HALF_WINDOW(window)) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow, threadLocalCol - HALF_WINDOW(window))) {
						shared_block[MI(threadLocalRow + HALF_WINDOW(window), threadLocalCol, SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow, threadLocalCol - HALF_WINDOW(window))];
					}
				}

				// right columns
				if (threadLocalCol >= SHARED_MEMORY_BLOCK_WIDTH) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow, threadLocalCol + HALF_WINDOW(window))) {
						shared_block[MI(threadLocalRow + HALF_WINDOW(window), threadLocalCol + 2 * HALF_WINDOW(window), SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow, threadLocalCol + HALF_WINDOW(window))];
					}
				}

				// top left corner
				if (threadLocalRow < HALF_WINDOW(window) && threadLocalCol < HALF_WINDOW(window)) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol - HALF_WINDOW(window))) {
						shared_block[MI(threadLocalRow, threadLocalCol, SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol - HALF_WINDOW(window))];
					}
				}

				// top right corner
				if (threadLocalRow < HALF_WINDOW(window) && threadLocalCol >= SHARED_MEMORY_BLOCK_WIDTH) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window))) {
						shared_block[MI(threadLocalRow, threadLocalCol + 2 * HALF_WINDOW(window), SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window))];
					}
				}

				// bottom left corner
				if (threadLocalRow >= SHARED_MEMORY_BLOCK_HEIGHT && threadLocalCol < HALF_WINDOW(window)) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window))) {
						shared_block[MI(threadLocalRow + 2 * HALF_WINDOW(window), threadLocalCol, SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow - HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window))];
					}
				}

				// bottom right corner
				if (threadLocalRow >= SHARED_MEMORY_BLOCK_HEIGHT && threadLocalCol >= SHARED_MEMORY_BLOCK_WIDTH) {
					if (pixel_coordinates_are_valid(blockVerticalIndex, blockHorizontalIndex, threadLocalRow + HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window))) {
						shared_block[MI(threadLocalRow + 2 * HALF_WINDOW(window), threadLocalCol + 2 * HALF_WINDOW(window), SHARED_MEMORY_BLOCK_WIDTH + window - 1)] = YUV[YUV_INDEX(cam_index, blockVerticalIndex, blockHorizontalIndex, threadLocalRow + HALF_WINDOW(window), threadLocalCol + HALF_WINDOW(window))];
					}

				}


				__syncthreads();

			}


			int y = blockVerticalIndex * SHARED_MEMORY_BLOCK_HEIGHT + threadLocalRow;
			int x = blockHorizontalIndex * SHARED_MEMORY_BLOCK_WIDTH + threadLocalCol;

			int localX = threadLocalCol + HALF_WINDOW(window);
			int localY = threadLocalRow + HALF_WINDOW(window);




			if (x >= cam_width || y >= cam_height)
				return;


			// Compute values that do not rely on z
			float X_ref = (ref_h_K_inv[0] * x + ref_h_K_inv[1] * y + ref_h_K_inv[2]);
			float Y_ref = (ref_h_K_inv[3] * x + ref_h_K_inv[4] * y + ref_h_K_inv[5]);
			float Z_ref = (ref_h_K_inv[6] * x + ref_h_K_inv[7] * y + ref_h_K_inv[8]);

			float X_b = ref_h_R_inv[0] * X_ref + ref_h_R_inv[1] * Y_ref + ref_h_R_inv[2] * Z_ref;
			float Y_b = ref_h_R_inv[3] * X_ref + ref_h_R_inv[4] * Y_ref + ref_h_R_inv[5] * Z_ref;
			float Z_b = ref_h_R_inv[6] * X_ref + ref_h_R_inv[7] * Y_ref + ref_h_R_inv[8] * Z_ref;

			for (int zi = 0; zi < ZPlanes; zi++) {

				float min_cost = 255.0f; // initialize cost at maximum

				// (i) calculate projection index
				float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
				float X = X_b * z - ref_h_t_inv[0];
				float Y = Y_b * z - ref_h_t_inv[1];
				float Z = Z_b * z - ref_h_t_inv[2];

				for (int cam_index = 0; cam_index < cam_number; cam_index++) {
					if (cam_index == ref_index)
						continue;

					float X_proj = cam_h_R[MI(cam_index, 0, 9)] * X + cam_h_R[MI(cam_index, 1, 9)] * Y + cam_h_R[MI(cam_index, 2, 9)] * Z - cam_h_t[MI(cam_index, 0, 3)];
					float Y_proj = cam_h_R[MI(cam_index, 3, 9)] * X + cam_h_R[MI(cam_index, 4, 9)] * Y + cam_h_R[MI(cam_index, 5, 9)] * Z - cam_h_t[MI(cam_index, 1, 3)];
					float Z_proj = cam_h_R[MI(cam_index, 6, 9)] * X + cam_h_R[MI(cam_index, 7, 9)] * Y + cam_h_R[MI(cam_index, 8, 9)] * Z - cam_h_t[MI(cam_index, 2, 3)];

					float XZ_proj = X_proj / Z_proj;
					float YZ_proj = Y_proj / Z_proj;

					int x_proj = (int)(cam_h_K[MI(cam_index, 0, 9)] * XZ_proj + cam_h_K[MI(cam_index, 1, 9)] * YZ_proj + cam_h_K[MI(cam_index, 2, 9)]);
					int y_proj = (int)(cam_h_K[MI(cam_index, 3, 9)] * XZ_proj + cam_h_K[MI(cam_index, 4, 9)] * YZ_proj + cam_h_K[MI(cam_index, 5, 9)]);

					// Calculate cost in a window
					int radius = window / 2;
					int kmin = -radius, kmax = radius, lmin = -radius, lmax = radius;
					if (x_proj + lmin < 0) lmin = -x_proj;
					if (x + lmin < 0) lmin = -x;
					if (y_proj + kmin < 0) kmin = -y_proj;
					if (y + kmin < 0) kmin = -y;
					if (x_proj + lmax >= cam_width) lmax = cam_width - x_proj - 1;
					if (x + lmax >= cam_width) lmax = cam_width - x - 1;
					if (y_proj + kmax >= cam_height) kmax = cam_height - y_proj - 1;
					if (y + kmax >= cam_height) kmax = cam_height - y - 1;

					float cost = 0.0f;
					float cc = 0.0f;
					for (int k = kmin; k <= kmax; k++) {
						for (int l = lmin; l <= lmax; l++) {
							int diff = YUV[CUBE(x_proj + l, y_proj + k, cam_index, cam_width, cam_height)] - shared_block[LOCAL_COORDINATES(ref_index, y + k, x + l)];

							diff = diff < 0 ? -diff : diff;
							cost += diff;
							cc += 1.0f;
						}
					}
					cost /= cc;

					min_cost = fminf(min_cost, cost);
				}
				result[CUBE(x, y, zi, cam_width, cam_height)] = min_cost;

			}
		}
	}
}

//USING CONSTANT MEMORY
__global__ void dev_plane_sweeping_constant_memory(
	float* result, const float* cam_h_K, const float* ref_h_K_inv, const float* cam_h_R, const float* ref_h_R_inv, const double* cam_h_t, const double* ref_h_t_inv, const int* YUV, int cam_number, float ZFar, float ZNear, int ZPlanes, int cam_width, int cam_height, int ref_index, int window
) {
	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;

	// threads are divided in blocks of N_PIXELS_PER_THREAD ^ 2
	int virtualBlockIndex = thread_index / N_PIXELS_PER_THREAD;
	int virtualThreadIndex = thread_index % N_PIXELS_PER_THREAD;

	if (thread_index == 0) {

		float test = 3.141592;
		for (int i = 0; i < 20; i++) {
			printf("%f, %f\n", cam_h_R[i], const_cam_h_R[i]);
		}
		printf("\n");
	}



	for (int pixel_padding = 0; pixel_padding < N_PIXELS_PER_THREAD; ++pixel_padding) {

		if (thread_index == 0) {
			//printf("Pixel %d/%d\n", pixel_padding + 1, N_PIXELS_PER_THREAD);
		}

		// threads of the same virtual block process consecutive pixels of the block at the same time
		int pixel_index = N_PIXELS_PER_THREAD * N_PIXELS_PER_THREAD * virtualBlockIndex + N_PIXELS_PER_THREAD * pixel_padding + virtualThreadIndex;



		int y = pixel_index / cam_width;
		int x = pixel_index % cam_width;


		if (x >= cam_width || y >= cam_height)
			return;


		// Compute values that do not rely on z
		float X_ref = (const_ref_h_K_inv[0] * x + const_ref_h_K_inv[1] * y + const_ref_h_K_inv[2]);
		float Y_ref = (const_ref_h_K_inv[3] * x + const_ref_h_K_inv[4] * y + const_ref_h_K_inv[5]);
		float Z_ref = (const_ref_h_K_inv[6] * x + const_ref_h_K_inv[7] * y + const_ref_h_K_inv[8]);

		float X_b = const_ref_h_R_inv[0] * X_ref + const_ref_h_R_inv[1] * Y_ref + const_ref_h_R_inv[2] * Z_ref;
		float Y_b = const_ref_h_R_inv[3] * X_ref + const_ref_h_R_inv[4] * Y_ref + const_ref_h_R_inv[5] * Z_ref;
		float Z_b = const_ref_h_R_inv[6] * X_ref + const_ref_h_R_inv[7] * Y_ref + const_ref_h_R_inv[8] * Z_ref;

		for (int zi = 0; zi < ZPlanes; zi++) {

			float min_cost = 255.0f; // initialize cost at maximum

			// (i) calculate projection index
			float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
			float X = X_b * z - const_ref_h_t_inv[0];
			float Y = Y_b * z - const_ref_h_t_inv[1];
			float Z = Z_b * z - const_ref_h_t_inv[2];

			for (int cam_index = 0; cam_index < cam_number; cam_index++) {
				if (cam_index == ref_index)
					continue;

				float X_proj = const_cam_h_R[MI(cam_index, 0, 9)] * X + const_cam_h_R[MI(cam_index, 1, 9)] * Y + const_cam_h_R[MI(cam_index, 2, 9)] * Z - const_cam_h_t[MI(cam_index, 0, 3)];
				float Y_proj = const_cam_h_R[MI(cam_index, 3, 9)] * X + const_cam_h_R[MI(cam_index, 4, 9)] * Y + const_cam_h_R[MI(cam_index, 5, 9)] * Z - const_cam_h_t[MI(cam_index, 1, 3)];
				float Z_proj = const_cam_h_R[MI(cam_index, 6, 9)] * X + const_cam_h_R[MI(cam_index, 7, 9)] * Y + const_cam_h_R[MI(cam_index, 8, 9)] * Z - const_cam_h_t[MI(cam_index, 2, 3)];

				float XZ_proj = X_proj / Z_proj;
				float YZ_proj = Y_proj / Z_proj;

				int x_proj = (int)(const_cam_h_K[MI(cam_index, 0, 9)] * XZ_proj + const_cam_h_K[MI(cam_index, 1, 9)] * YZ_proj + const_cam_h_K[MI(cam_index, 2, 9)]);
				int y_proj = (int)(const_cam_h_K[MI(cam_index, 3, 9)] * XZ_proj + const_cam_h_K[MI(cam_index, 4, 9)] * YZ_proj + const_cam_h_K[MI(cam_index, 5, 9)]);

				// Calculate cost in a window
				int radius = window / 2;
				int kmin = -radius, kmax = radius, lmin = -radius, lmax = radius;
				if (x_proj + lmin < 0) lmin = -x_proj;
				if (x + lmin < 0) lmin = -x;
				if (y_proj + kmin < 0) kmin = -y_proj;
				if (y + kmin < 0) kmin = -y;
				if (x_proj + lmax >= cam_width) lmax = cam_width - x_proj - 1;
				if (x + lmax >= cam_width) lmax = cam_width - x - 1;
				if (y_proj + kmax >= cam_height) kmax = cam_height - y_proj - 1;
				if (y + kmax >= cam_height) kmax = cam_height - y - 1;

				float cost = 0.0f;
				float cc = 0.0f;
				for (int k = kmin; k <= kmax; k++) {
					for (int l = lmin; l <= lmax; l++) {
						int diff = YUV[CUBE(x_proj + l, y_proj + k, cam_index, cam_width, cam_height)] - YUV[CUBE(x + l, y + k, ref_index, cam_width, cam_height)];
						diff = diff < 0 ? -diff : diff;
						cost += diff;
						cc += 1.0f;
					}
				}
				cost /= cc;

				min_cost = fminf(min_cost, cost);
			}
			result[CUBE(x, y, zi, cam_width, cam_height)] = min_cost;
		}
	}
}

//USING SHARED MEMORY
		
__global__ void dev_plane_sweeping_shared_memory_2(
	float* result, const float* __restrict__ cam_h_K, const float* __restrict__ ref_h_K_inv, const float* __restrict__ cam_h_R, const float* __restrict__ ref_h_R_inv,
	const double* __restrict__ cam_h_t, const double* __restrict__ ref_h_t_inv, const int* __restrict__ YUV, int cam_number, float ZFar, float ZNear, int ZPlanes,
	int cam_width, int cam_height, int ref_index, int window) {

	extern __shared__ float shared_mem[];

	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_id = threadIdx.x;

	// Load matrices into shared memory
	float* shared_ref_h_K_inv = shared_mem;
	float* shared_ref_h_R_inv = shared_mem + 9;
	float* shared_cam_h_R = shared_mem + 18;
	float* shared_cam_h_K = shared_mem + 18 + 9 * cam_number;

	if (thread_id < 9) {
		shared_ref_h_K_inv[thread_id] = ref_h_K_inv[thread_id];
		shared_ref_h_R_inv[thread_id] = ref_h_R_inv[thread_id];
	}

	for (int i = 0; i < cam_number; i++) {
		if (thread_id < 9) {
			shared_cam_h_R[i * 9 + thread_id] = cam_h_R[i * 9 + thread_id];
			shared_cam_h_K[i * 9 + thread_id] = cam_h_K[i * 9 + thread_id];
		}
	}
	__syncthreads();

	int num_pixels = cam_width * cam_height;
	for (int pixel_index = thread_index; pixel_index < num_pixels; pixel_index += blockDim.x * gridDim.x) {
		int y = pixel_index / cam_width;
		int x = pixel_index % cam_width;

		if (x >= cam_width || y >= cam_height)
			return;

		// Compute values that do not rely on z
		float X_ref = shared_ref_h_K_inv[0] * x + shared_ref_h_K_inv[1] * y + shared_ref_h_K_inv[2];
		float Y_ref = shared_ref_h_K_inv[3] * x + shared_ref_h_K_inv[4] * y + shared_ref_h_K_inv[5];
		float Z_ref = shared_ref_h_K_inv[6] * x + shared_ref_h_K_inv[7] * y + shared_ref_h_K_inv[8];

		float X_b = shared_ref_h_R_inv[0] * X_ref + shared_ref_h_R_inv[1] * Y_ref + shared_ref_h_R_inv[2] * Z_ref;
		float Y_b = shared_ref_h_R_inv[3] * X_ref + shared_ref_h_R_inv[4] * Y_ref + shared_ref_h_R_inv[5] * Z_ref;
		float Z_b = shared_ref_h_R_inv[6] * X_ref + shared_ref_h_R_inv[7] * Y_ref + shared_ref_h_R_inv[8] * Z_ref;

		for (int zi = 0; zi < ZPlanes; zi++) {
			float min_cost = 255.0f; // initialize cost at maximum

			float z = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
			float X = X_b * z - ref_h_t_inv[0];
			float Y = Y_b * z - ref_h_t_inv[1];
			float Z = Z_b * z - ref_h_t_inv[2];

			for (int cam_index = 0; cam_index < cam_number; cam_index++) {
				if (cam_index == ref_index)
					continue;

				float* cam_R = &shared_cam_h_R[cam_index * 9];
				float* cam_K = &shared_cam_h_K[cam_index * 9];
				float X_proj = cam_R[0] * X + cam_R[1] * Y + cam_R[2] * Z - cam_h_t[cam_index * 3];
				float Y_proj = cam_R[3] * X + cam_R[4] * Y + cam_R[5] * Z - cam_h_t[cam_index * 3 + 1];
				float Z_proj = cam_R[6] * X + cam_R[7] * Y + cam_R[8] * Z - cam_h_t[cam_index * 3 + 2];

				float XZ_proj = X_proj / Z_proj;
				float YZ_proj = Y_proj / Z_proj;

				int x_proj = (int)(cam_K[0] * XZ_proj + cam_K[1] * YZ_proj + cam_K[2]);
				int y_proj = (int)(cam_K[3] * XZ_proj + cam_K[4] * YZ_proj + cam_K[5]);

				if (x_proj < 0 || x_proj >= cam_width || y_proj < 0 || y_proj >= cam_height)
					continue; // Skip invalid projections

				int radius = window / 2;
				int kmin = -radius, kmax = radius, lmin = -radius, lmax = radius;
				if (x_proj + lmin < 0) lmin = -x_proj;
				if (x + lmin < 0) lmin = -x;
				if (y_proj + kmin < 0) kmin = -y_proj;
				if (y + kmin < 0) kmin = -y;
				if (x_proj + lmax >= cam_width) lmax = cam_width - x_proj - 1;
				if (x + lmax >= cam_width) lmax = cam_width - x - 1;
				if (y_proj + kmax >= cam_height) kmax = cam_height - y_proj - 1;
				if (y + kmax >= cam_height) kmax = cam_height - y - 1;

				float cost = 0.0f;
				float cc = 0.0f;
				for (int k = kmin; k <= kmax; k++) {
					for (int l = lmin; l <= lmax; l++) {
						int diff = YUV[CUBE(x_proj + l, y_proj + k, cam_index, cam_width, cam_height)] - YUV[CUBE(x + l, y + k, ref_index, cam_width, cam_height)];
						diff = diff < 0 ? -diff : diff;
						cost += diff;
						cc += 1.0f;
					}
				}
				cost /= cc;

				min_cost = fminf(min_cost, cost);
			}
			result[CUBE(x, y, zi, cam_width, cam_height)] = min_cost;
		}
	}
}

__global__ void dev_plane_sweeping_constant_memory_2(
	float* result, const float* __restrict__ cam_h_K, const float* __restrict__ ref_h_K_inv, const float* __restrict__ cam_h_R, const float* __restrict__ ref_h_R_inv,
	const double* __restrict__ cam_h_t, const double* __restrict__ ref_h_t_inv, const int* __restrict__ YUV, int cam_number, float ZFar, float ZNear, int ZPlanes,
	int cam_width, int cam_height, int ref_index, int window) {

	__shared__ float SM_as_registers[N_THREADS_PER_BLOCK * 11];

	/*
	SM_as_registers[11 * threadIdx.x + i]

	0 = X_ref
	1 = Y_ref
	2 = Z_ref
	3 = X_b
	4 = Y_b
	5 = Z_b
	6 = min_cost
	7 = z
	8 = X
	9 = Y
	10 = Z
	*/

	extern __shared__ float shared_mem[];

	int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
	int thread_id = threadIdx.x;

	// Load matrices into shared memory
	float* shared_ref_h_K_inv = shared_mem;
	float* shared_ref_h_R_inv = shared_mem + 9;
	float* shared_cam_h_R = shared_mem + 18;

	if (thread_id < 9) {
		shared_ref_h_K_inv[thread_id] = ref_h_K_inv[thread_id];
		shared_ref_h_R_inv[thread_id] = ref_h_R_inv[thread_id];
	}
	__syncthreads();

	for (int i = 0; i < cam_number; i++) {
		if (thread_id < 9) {
			shared_cam_h_R[i * 9 + thread_id] = cam_h_R[i * 9 + thread_id];
		}
	}
	__syncthreads();

	int num_pixels = cam_width * cam_height;
	for (int pixel_index = thread_index; pixel_index < num_pixels; pixel_index += blockDim.x * gridDim.x) {
		int y = pixel_index / cam_width;
		int x = pixel_index % cam_width;

		if (x >= cam_width || y >= cam_height)
			return;

		// Compute values that do not rely on z
		SM_as_registers[11 * threadIdx.x] = shared_ref_h_K_inv[0] * x + shared_ref_h_K_inv[1] * y + shared_ref_h_K_inv[2];
		SM_as_registers[11 * threadIdx.x + 1] = shared_ref_h_K_inv[3] * x + shared_ref_h_K_inv[4] * y + shared_ref_h_K_inv[5];
		SM_as_registers[11 * threadIdx.x + 2] = shared_ref_h_K_inv[6] * x + shared_ref_h_K_inv[7] * y + shared_ref_h_K_inv[8];

		SM_as_registers[11 * threadIdx.x + 3] = shared_ref_h_R_inv[0] * SM_as_registers[11 * threadIdx.x] + shared_ref_h_R_inv[1] * SM_as_registers[11 * threadIdx.x + 1] + shared_ref_h_R_inv[2] * SM_as_registers[11 * threadIdx.x + 2];
		SM_as_registers[11 * threadIdx.x + 4] = shared_ref_h_R_inv[3] * SM_as_registers[11 * threadIdx.x] + shared_ref_h_R_inv[4] * SM_as_registers[11 * threadIdx.x + 1] + shared_ref_h_R_inv[5] * SM_as_registers[11 * threadIdx.x + 2];
		SM_as_registers[11 * threadIdx.x + 5] = shared_ref_h_R_inv[6] * SM_as_registers[11 * threadIdx.x] + shared_ref_h_R_inv[7] * SM_as_registers[11 * threadIdx.x + 1] + shared_ref_h_R_inv[8] * SM_as_registers[11 * threadIdx.x + 2];
		for (int zi = 0; zi < ZPlanes; zi++) {
			SM_as_registers[11 * threadIdx.x + 6] = 255.0f; // initialize cost at maximum

			SM_as_registers[11 * threadIdx.x + 7] = ZNear * ZFar / (ZNear + (((float)zi / (float)ZPlanes) * (ZFar - ZNear)));
			SM_as_registers[11 * threadIdx.x + 8] = SM_as_registers[11 * threadIdx.x + 3] * SM_as_registers[11 * threadIdx.x + 7] - ref_h_t_inv[0];
			SM_as_registers[11 * threadIdx.x + 9] = SM_as_registers[11 * threadIdx.x + 4] * SM_as_registers[11 * threadIdx.x + 7] - ref_h_t_inv[1];
			SM_as_registers[11 * threadIdx.x + 10] = SM_as_registers[11 * threadIdx.x + 5] * SM_as_registers[11 * threadIdx.x + 7] - ref_h_t_inv[2];

			for (int cam_index = 0; cam_index < cam_number; cam_index++) {
				if (cam_index == ref_index)
					continue;
				float* cam_R = &shared_cam_h_R[cam_index * 9];
				const float X_proj = cam_R[0] * SM_as_registers[11 * threadIdx.x + 8] + cam_R[1] * SM_as_registers[11 * threadIdx.x + 9] + cam_R[2] * SM_as_registers[11 * threadIdx.x + 10] - cam_h_t[cam_index * 3];
				const float Y_proj = cam_R[3] * SM_as_registers[11 * threadIdx.x + 8] + cam_R[4] * SM_as_registers[11 * threadIdx.x + 9] + cam_R[5] * SM_as_registers[11 * threadIdx.x + 10] - cam_h_t[cam_index * 3 + 1];
				const float Z_proj = cam_R[6] * SM_as_registers[11 * threadIdx.x + 8] + cam_R[7] * SM_as_registers[11 * threadIdx.x + 9] + cam_R[8] * SM_as_registers[11 * threadIdx.x + 10] - cam_h_t[cam_index * 3 + 2];

				const float XZ_proj = X_proj / Z_proj;
				const float YZ_proj = Y_proj / Z_proj;

				const int x_proj = (int)(cam_h_K[cam_index * 9] * XZ_proj + cam_h_K[cam_index * 9 + 1] * YZ_proj + cam_h_K[cam_index * 9 + 2]);
				int y_proj = (int)(cam_h_K[cam_index * 9 + 3] * XZ_proj + cam_h_K[cam_index * 9 + 4] * YZ_proj + cam_h_K[cam_index * 9 + 5]);

				if (x_proj < 0 || x_proj >= cam_width || y_proj < 0 || y_proj >= cam_height)
					continue; // Skip invalid projections

				const int radius = window / 2;
				int kmin = -radius, kmax = radius, lmin = -radius, lmax = radius;
				if (x_proj + lmin < 0) lmin = -x_proj;
				if (x + lmin < 0) lmin = -x;
				if (y_proj + kmin < 0) kmin = -y_proj;
				if (y + kmin < 0) kmin = -y;
				if (x_proj + lmax >= cam_width) lmax = cam_width - x_proj - 1;
				if (x + lmax >= cam_width) lmax = cam_width - x - 1;
				if (y_proj + kmax >= cam_height) kmax = cam_height - y_proj - 1;
				if (y + kmax >= cam_height) kmax = cam_height - y - 1;

				float cost = 0.0f;
				float cc = 0.0f;
				for (int k = kmin; k <= kmax; k++) {
					for (int l = lmin; l <= lmax; l++) {
						int diff = YUV[CUBE(x_proj + l, y_proj + k, cam_index, cam_width, cam_height)] - YUV[CUBE(x + l, y + k, ref_index, cam_width, cam_height)];
						diff = diff < 0 ? -diff : diff;
						cost += diff;
						cc += 1.0f;
					}
				}
				cost /= cc;

				SM_as_registers[11 * threadIdx.x + 6] = fminf(SM_as_registers[11 * threadIdx.x + 6], cost);
			}
			result[CUBE(x, y, zi, cam_width, cam_height)] = SM_as_registers[11 * threadIdx.x + 6];
		}
	}
}

//kernel used to warup the gpu
__global__ void warmup(float* A, float* B, int N) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int j = blockDim.y * blockIdx.y + threadIdx.y;

	if (i >= N || j >= N)
		return;

	A[MI(i, j, N)] = B[MI(i, j, N)];
}


				

void wrap_plane_sweeping(float* result, const float* cam_h_K, const float* ref_h_K_inv, const float* cam_h_R, const float* ref_h_R_inv, const double* cam_h_t, const double* ref_h_t_inv, const int* h_YUV, int cam_number, float ZFar, float ZNear, int ZPlanes, int cam_width, int cam_height, int ref_index, int window) {

	printf("Plane Sweeping CUDA:\n");
	float* dev_result, * dev_cam_h_K, * dev_ref_h_K_inv, * dev_cam_h_R, * dev_ref_h_R_inv;
	double* dev_cam_h_t, * dev_ref_h_t_inv;
	int* dev_h_YUV;

	// OUTPUT
	cudaMalloc((void**)&dev_result, cam_width * cam_height * ZPlanes * sizeof(float));

	// INPUTS
	cudaMalloc((void**)&dev_cam_h_K, 9 * cam_number * sizeof(float));
	cudaMalloc((void**)&dev_ref_h_K_inv, 9 * sizeof(float));
	cudaMalloc((void**)&dev_cam_h_R, 9 * cam_number * sizeof(float));
	cudaMalloc((void**)&dev_ref_h_R_inv, 9 * sizeof(float));
	cudaMalloc((void**)&dev_cam_h_t, 9 * cam_number * sizeof(double));
	cudaMalloc((void**)&dev_ref_h_t_inv, 9 * sizeof(double));
	cudaMalloc((void**)&dev_h_YUV, cam_number * cam_width * cam_height * sizeof(int));


	// Send inputs to GPU memory
	cudaMemcpy(dev_cam_h_K, cam_h_K, 9 * cam_number * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ref_h_K_inv, ref_h_K_inv, 9 * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cam_h_R, cam_h_R, 9 * cam_number * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ref_h_R_inv, ref_h_R_inv, 9 * sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cam_h_t, cam_h_t, 9 * cam_number * sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ref_h_t_inv, ref_h_t_inv, 9 * sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_h_YUV, h_YUV, cam_number * cam_width * cam_height * sizeof(int),cudaMemcpyHostToDevice);

	// Constant memory
	cudaMemcpyToSymbol(const_cam_h_K, cam_h_K, CAM_NUMBER * K_SIZE * sizeof(float));
	cudaMemcpyToSymbol(const_ref_h_K_inv, ref_h_K_inv, K_SIZE * sizeof(float));
	cudaMemcpyToSymbol(const_cam_h_R, cam_h_R, CAM_NUMBER * R_SIZE * sizeof(float));
	cudaMemcpyToSymbol(const_ref_h_R_inv, ref_h_R_inv, R_SIZE * sizeof(float));
	cudaMemcpyToSymbol(const_cam_h_t, cam_h_t, CAM_NUMBER * t_SIZE * sizeof(double));
	cudaMemcpyToSymbol(const_ref_h_t_inv, ref_h_t_inv, t_SIZE * sizeof(double));

	// WARM UP --> Needed for better timing !
	int N_threads = BLOCKSIZE;
	dim3 thread_size(BLOCKSIZE, BLOCKSIZE);
	dim3 block_size(
		(N + (thread_size.x - 1)) / thread_size.x,
		(N + (thread_size.y - 1)) / thread_size.y,
		1);
	float* dev_A = 0;
	float* dev_B = 0;
	float* dev_C = 0;

	cudaMalloc((void**)&dev_A, N * N * sizeof(float));
	cudaMalloc((void**)&dev_B, N * N * sizeof(float));
	cudaMalloc((void**)&dev_C, N * N * sizeof(float));
	for (int k = 0; k < 20; k++) {
		warmup << <block_size, thread_size >> > (dev_C, dev_A, N);
		cudaDeviceSynchronize();
		cudaGetLastError();
	}
	// END WARM UP

	
	int num_elements = CAM_WIDTH * CAM_HEIGHT;
	int b_size = 512;  // Example value, tune for your specific GPU
	int grid_size = (num_elements + b_size - 1) / b_size;
	int shared_mem_size = (9 * 3 * cam_number) * sizeof(float); // Example calculation
	

	
	// Launch kernel
	cudaEvent_t start = start_cuda_timer();
	dev_plane_sweeping_naive << < grid_size, b_size >> > (dev_result, dev_cam_h_K, dev_ref_h_K_inv, dev_cam_h_R, dev_ref_h_R_inv, dev_cam_h_t, dev_ref_h_t_inv, dev_h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);
	
	//dev_plane_sweeping_naive_types << < grid_size, b_size >> > (dev_result, dev_cam_h_K, dev_ref_h_K_inv, dev_cam_h_R, dev_ref_h_R_inv, dev_cam_h_t, dev_ref_h_t_inv, dev_h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);
	
	//dev_plane_sweeping_change_pixel_order << < grid_size, b_size >> > (dev_result, dev_cam_h_K, dev_ref_h_K_inv, dev_cam_h_R, dev_ref_h_R_inv, dev_cam_h_t, dev_ref_h_t_inv, dev_h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);

	//dev_plane_sweeping_shared_memory << < SHARED_MEMORY_BLOCK_HEIGHT, SHARED_MEMORY_BLOCK_WIDTH >> > (dev_result, dev_cam_h_K, dev_ref_h_K_inv, dev_cam_h_R, dev_ref_h_R_inv, dev_cam_h_t, dev_ref_h_t_inv, dev_h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);

	//dev_plane_sweeping_shared_memory_2 << < grid_size, b_size >> > (dev_result, dev_cam_h_K, dev_ref_h_K_inv, dev_cam_h_R, dev_ref_h_R_inv, dev_cam_h_t, dev_ref_h_t_inv, dev_h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);

	//dev_plane_sweeping_constant_memory << < grid_size, b_size >> > (dev_result, dev_cam_h_K, dev_ref_h_K_inv, dev_cam_h_R, dev_ref_h_R_inv, dev_cam_h_t, dev_ref_h_t_inv, dev_h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);

	//dev_plane_sweeping_constant_memory_2 << < grid_size, b_size >> > (dev_result, dev_cam_h_K, dev_ref_h_K_inv, dev_cam_h_R, dev_ref_h_R_inv, dev_cam_h_t, dev_ref_h_t_inv, dev_h_YUV, cam_number, ZFar, ZNear, ZPlanes, cam_width, cam_height, ref_index, window);

	float gpu_time_ms = end_cuda_timer(start);
	cudaDeviceSynchronize();


	// Send GPU outputs to CPU memory
	start = start_cuda_timer();
	cudaMemcpy(result, dev_result, cam_width * cam_height * ZPlanes * sizeof(float),
		cudaMemcpyDeviceToHost);
	float mem_time_ms = end_cuda_timer(start);

	computeStats(gpu_time_ms, mem_time_ms, cam_width, cam_height, cam_number, ZPlanes, window);

	cudaDeviceSynchronize();

	printf("%s\n", cudaGetErrorString(cudaGetLastError()));


}

std::chrono::steady_clock::time_point start_cpu_timer()
{
	return std::chrono::high_resolution_clock::now();
}

double end_cpu_timer(std::chrono::steady_clock::time_point start)
{
	auto stop = std::chrono::high_resolution_clock::now();
	return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
}

cudaEvent_t start_cuda_timer()
{
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEventRecord(start, NULL);
	return start;
}

float end_cuda_timer(cudaEvent_t start)
{
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	float millisec;
	cudaEventElapsedTime(&millisec, start, stop);

	return millisec;
}

void computeStats(float gpu_time_ms, float mem_time_ms, int cam_width, int cam_height, int cam_number, int zPlanes, int window_size) {
	float image_size = cam_number * cam_width * cam_height * sizeof(int);
	float output_size = cam_width * cam_height * zPlanes * sizeof(float);
	float paramSize = cam_number * (9 * 2 * sizeof(float) + 3 * sizeof(double));

	float memoryUsed = image_size + output_size + paramSize;
	float memoryThroughput = memoryUsed / gpu_time_ms / 1e+6; //Divide by 1 000 000 to have GB/s

	float totalOperations = 1e-9 * cam_width * cam_height * (zPlanes * ((cam_number - 1) * (window_size * window_size)));
	float computationThroughput = totalOperations / gpu_time_ms * 1e+3;
	float computeIntensity = computationThroughput / memoryThroughput;

#ifdef PRINT_STATS
	std::cout << "GPU Kernel time : " << gpu_time_ms << " ms" << std::endl;
	std::cout << "DtoH Memcpy time : " << mem_time_ms << " ms" << std::endl;
	std::cout << "Memory used : " << memoryUsed / 1e+9 << " GB" << std::endl;
	//std::cout << "Memory throughput : " << memoryThroughput << " GB/s " << std::endl;
	//std::cout << "Operations : " << totalOperations << " GOPS" << std::endl;
	//std::cout << "Computation throughput : " << computationThroughput << " GOPS/s " << std::endl;
	//std::cout << "Compute intensity : " << computeIntensity << " OPS/Byte" << std::endl;
#endif
}





