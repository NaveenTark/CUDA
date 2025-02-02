
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <chrono>
#include <iostream>

#define N 1000000
#define ITERATIONS 20


void CPU_add(float* a, float* b, float* c, size_t n)
{
	for (size_t i = 0;i < n;i++)
	{
		c[i] = a[i] + b[i];
	}
}



int main()
{
	float* h_a = new float[N];
	float* h_b = new float[N];
	float* h_c = new float[N];

	std::random_device rd; // Obtain a random number from hardware
	std::mt19937 rng(rd()); // Seed the generator
	std::uniform_real_distribution<float> dist(0.0f, 100.0f);

	// Initialize the array with random floats
	for (int i = 0; i < N; ++i) {
		h_a[i] = dist(rng); // Generate random float
		h_b[i] = dist(rng);
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




	// Clean up
	delete[] h_a;
	delete[] h_b;
	delete[] h_c;

	return 0;
}