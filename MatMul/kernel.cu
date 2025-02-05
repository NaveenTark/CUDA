
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>
#include <chrono>


constexpr int ITERATIONS = 20;
constexpr size_t M = 1024; // Number of rows in A and C
constexpr size_t K = 2048; // Number of columns in A and rows in B
constexpr size_t N = 1024; // Number of columns in B and C

void Basic_mul(float* a, float* b, float* out, int R, int C, int Z)
{
	for (int i = 0;i < R;i++)
	{
		for (int k = 0; k < Z;k++)
		{
			for (int j = 0;j < C;j++)
			{
				out[i * Z + k] += a[i * C + j] * b[j * Z + k];
			}
		}
	}
}
//Store sum in register and update output infrequently to avoid memory access
void Basic_mul_opt1(float* a, float* b, float* out, int R, int C, int Z)
{
	for (int i = 0;i < R;i++)
	{
		for (int k = 0; k < Z;k++)
		{
			float sum = 0.0f;
			for (int j = 0;j < C;j++)
			{
				sum += a[i * C + j] * b[j * Z + k];
			}
			out[i * Z + k] = sum;
		}
	}
}
//cache friendly access from B
void Basic_mul_opt2(float* a, float* b, float* out, int R, int C, int Z)
{
	for (int i = 0;i < R;i++)
	{
		for (int j = 0;j < C;j++)
		
		{
			float a_ij = a[i * C + j];
			for (int k = 0; k < Z;k++)
			{
				out[i * Z + k]+= a_ij * b[j * Z + k];
			}
			
		}
	}
}

int main()
{
	float* h_a = new float[M*K];
	float* h_b = new float[K*N];
	float* h_c_basic = new float[M*N]();
	float* h_c_basic_opt1 = new float[M * N]();
	float* h_c_basic_opt2 = new float[M * N]();


	std::random_device rd;
	std::mt19937 rng(rd());
	std::uniform_real_distribution<float> dist(0.0f, 100.0f);

	for (int i = 0; i < N; ++i) {
		h_a[i] = dist(rng);
		h_b[i] = dist(rng);
	}

	// Single-threaded CPU benchmarking
	double cpu_seq_total_duration = 0.0;
	for (int iter = 0; iter < ITERATIONS; ++iter) {
		auto start = std::chrono::high_resolution_clock::now();
		Basic_mul(h_a, h_b, h_c_basic, M,K,N);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		cpu_seq_total_duration += duration.count();
	}
	double cpu_seq_avg_duration = cpu_seq_total_duration / ITERATIONS;
	std::cout << "Avg. time for single-threaded CPU basic mul: " << cpu_seq_avg_duration << " sec\n";

	// Single-threaded CPU benchmarking optimization 1
	double cpu_seq_total_duration1 = 0.0;
	for (int iter = 0; iter < ITERATIONS; ++iter) {
		auto start = std::chrono::high_resolution_clock::now();
		Basic_mul(h_a, h_b, h_c_basic_opt1, M, K, N);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		cpu_seq_total_duration1 += duration.count();
	}
	double cpu_seq_avg_duration1 = cpu_seq_total_duration1 / ITERATIONS;
	std::cout << "Avg. time for single-threaded CPU basic mul optimization 1: " << cpu_seq_avg_duration1 << " sec\n";

	// Single-threaded CPU benchmarking optimization 2
	double cpu_seq_total_duration2 = 0.0;
	for (int iter = 0; iter < ITERATIONS; ++iter) {
		auto start = std::chrono::high_resolution_clock::now();
		Basic_mul(h_a, h_b, h_c_basic_opt2, M, K, N);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration = end - start;
		cpu_seq_total_duration2 += duration.count();
	}
	double cpu_seq_avg_duration2 = cpu_seq_total_duration2 / ITERATIONS;
	std::cout << "Avg. time for single-threaded CPU basic mul optimization 2: " << cpu_seq_avg_duration2 << " sec\n";

	delete[] h_a;
	delete[] h_b;
	delete[] h_c_basic;
	delete[] h_c_basic_opt1;
	delete[] h_c_basic_opt2;



}
