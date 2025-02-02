#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <random>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>

#define N 100000000
#define ITERATIONS 20
#define BLOCK_SIZE 512

#define CUDA_CHECK(call)                                                 \
    {                                                                    \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    }

void CPU_add(float* a, float* b, float* c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// Multi-threaded CPU addition
void CPU_add_multithreaded(float* a, float* b, float* c, size_t n, int num_threads) {
    int block_size = n  / num_threads;
    std::vector<std::thread>threads;
    for (int i = 0;i < num_threads;i++)
    {
        int start = i * block_size;
        int end = (i == num_threads - 1) ? n  : start + block_size ;
        auto worker = [&](int start, int end) {
            for (int j = start;j < end;j++)
            {
                c[j] = a[j] + b[j];
            }
            };
        threads.emplace_back(worker, start, end);

    }
    for (auto& i : threads)
    {
        i.join();
    }
}

__global__ void GPU_add(float* a, float* b, float* c, size_t n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    float* h_a = new float[N];
    float* h_b = new float[N];
    float* h_c = new float[N];
    float* h_c_gpu = new float[N];
    float* h_c_mt = new float[N];

    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    for (int i = 0; i < N; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    float* d_a, * d_b, * d_c;
    size_t size = N * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_a, size));
    CUDA_CHECK(cudaMalloc(&d_b, size));
    CUDA_CHECK(cudaMalloc(&d_c, size));

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int GRID_SIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        CPU_add(h_a, h_b, h_c, N);
        GPU_add << <GRID_SIZE, BLOCK_SIZE >> > (d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Single-threaded CPU benchmarking
    double cpu_seq_total_duration = 0.0;
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        CPU_add(h_a, h_b, h_c, N);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        cpu_seq_total_duration += duration.count();
    }
    double cpu_seq_avg_duration = cpu_seq_total_duration / ITERATIONS;
    std::cout << "Avg. time for single-threaded CPU_add: " << cpu_seq_avg_duration << " sec\n";

    // Multi-threaded CPU benchmarking
    int max_threads = std::thread::hardware_concurrency() / 2;
    double cpu_mt_total_duration = 0.0;
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        CPU_add_multithreaded(h_a, h_b, h_c_mt, N, max_threads);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        cpu_mt_total_duration += duration.count();
    }
    double cpu_mt_avg_duration = cpu_mt_total_duration / ITERATIONS;
    std::cout << "Avg. time for multi-threaded CPU_add (" << max_threads << " threads): "
        << cpu_mt_avg_duration << " sec\n";

    // GPU benchmarking
    double gpu_total_duration = 0.0;
    for (int iter = 0; iter < ITERATIONS; ++iter) {
        auto start = std::chrono::high_resolution_clock::now();
        GPU_add << <GRID_SIZE, BLOCK_SIZE >> > (d_a, d_b, d_c, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        gpu_total_duration += duration.count();
    }
    double gpu_avg_duration = gpu_total_duration / ITERATIONS;
    std::cout << "Avg. time for GPU_add: " << gpu_avg_duration << " sec\n";

    // Speedup calculations
    std::cout << "Speedup (GPU vs single-threaded CPU): " << cpu_seq_avg_duration / gpu_avg_duration << '\n';
    std::cout << "Speedup (GPU vs multi-threaded CPU): " << cpu_mt_avg_duration / gpu_avg_duration << '\n';
    std::cout << "Speedup (Multi-threaded CPU vs single-threaded CPU): "
        << cpu_seq_avg_duration / cpu_mt_avg_duration << '\n';

    // Verify results
    cudaMemcpy(h_c_gpu, d_c, size, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (fabs(h_c[i] - h_c_gpu[i]) > 1e-5 || fabs(h_c_mt[i] - h_c[i]) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "Results are " << (correct ? "correct" : "incorrect") << '\n';

    

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    delete[] h_c_mt;
    delete[] h_c_gpu;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}
