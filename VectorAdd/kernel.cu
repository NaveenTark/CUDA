
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <chrono>
#include <iostream>

#define N 1000000
#define ITERATIONS 20
#define BLOCK_SIZE 256


#define CUDA_CHECK(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

void CPU_add(float* a, float* b, float* c, size_t n)
{
	for (size_t i = 0;i < n;i++)
	{
		c[i] = a[i] + b[i];
	}
}

__global__ void GPU_add(float* a, float* b, float* c, size_t n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < n)
	{
		c[i] = a[i] + b[i];
	}
}


int main()
{
	float* h_a = new float[N];
	float* h_b = new float[N];
	float* h_c = new float[N];
	float* h_c_gpu = new float[N];

	std::random_device rd; // Obtain a random number from hardware
	std::mt19937 rng(rd()); // Seed the generator
	std::uniform_real_distribution<float> dist(0.0f, 100.0f);

	// Initialize the array with random floats
	for (int i = 0; i < N; ++i) {
		h_a[i] = dist(rng); // Generate random float
		h_b[i] = dist(rng);
	}

	
	
	float* d_a, *d_b, *d_c;
	size_t size = N * sizeof(float);
	CUDA_CHECK(cudaMalloc(&d_a,size));
	CUDA_CHECK(cudaMalloc(&d_b,size));
	CUDA_CHECK(cudaMalloc(&d_c,size));

	// Copy data to device
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	// Warm-up runs
	printf("Performing warm-up runs...\n");
	for (int i = 0; i < 3; i++) {
		CPU_add(h_a, h_b, h_c, N);
		GPU_add << <GRID_SIZE, BLOCK_SIZE >> > (d_a, d_b, d_c, N);
		CUDA_CHECK(cudaDeviceSynchronize());
	}

	// Benchmarking CPU_add over multiple iterations
	double total_duration = 0.0;

	for (int iter = 0; iter < ITERATIONS; ++iter) {
		auto start = std::chrono::high_resolution_clock::now();
		CPU_add(h_a, h_b, h_c, N);
		auto end = std::chrono::high_resolution_clock::now();

		// Calculate the duration for this iteration
		std::chrono::duration<double> duration = end - start;
		total_duration += duration.count();
	}

	// Calculate average time taken
	double average_duration = total_duration / ITERATIONS;
	std::cout << "Average time taken for CPU_add over " << ITERATIONS
		<< " iterations: " << average_duration << " seconds" << std::endl;

	// Benchmarking GPU_add over multiple iterations
	total_duration = 0.0;

	for (int iter = 0; iter < ITERATIONS; ++iter) {
		auto start = std::chrono::high_resolution_clock::now();
		GPU_add << <GRID_SIZE, BLOCK_SIZE >> > (d_a, d_b, d_c, N);
		CUDA_CHECK(cudaDeviceSynchronize());
		auto end = std::chrono::high_resolution_clock::now();

		// Calculate the duration for this iteration
		std::chrono::duration<double> duration = end - start;
		total_duration += duration.count();
	}

	
	// Calculate average time taken
	average_duration = total_duration / ITERATIONS;
	std::cout << "Average time taken for GPU_add over " << ITERATIONS
		<< " iterations: " << average_duration << " seconds" << std::endl;


	// Verify results (optional)
	cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
	bool correct = true;
	for (int i = 0; i < N; i++) {
		if (fabs(h_c[i] - h_c_gpu[i]) > 1e-5) {
			correct = false;
			break;
		}
	}
	std::cout << "Results are " << (correct ? "correct" : "incorrect") << '\n';
	
	

	// Clean up
	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	delete[] h_c_gpu;
	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));
	return 0;
}