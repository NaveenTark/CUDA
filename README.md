# CUDA Learning Repository

This repository contains CUDA C++ projects to learn and experiment with parallel computing using NVIDIA's CUDA framework.

## Current Projects

### 1. Indexing
This project explores how CUDA threads and blocks are indexed in a 3D grid. The `whoami` kernel prints information about each thread and its corresponding block.

### 2. Matrix Multiplication (MatMul)
This project implements matrix multiplication using different approaches:
- **Basic CPU implementation**
- **Optimized CPU implementations**
- **Naive GPU implementation**
- **Tiled GPU implementation** (using shared memory for improved performance)

#### Key Optimizations:
- Storing intermediate sums in registers before updating output memory
- Using shared memory to reduce global memory access latency
- Optimizing memory access patterns for better cache efficiency

### 3. Vector Addition (VectorAdd)
Efficient vector addition using:

- **CPU Implementation**
- **Parallelized GPU Kernel**
-- **Thread-level Optimizations**

#### Key Concepts:

Launching 1D and 2D thread blocks
Grid-stride loops for large datasets

## Future Work
- Implementing more advanced CUDA optimizations and projects
- Profiling and benchmarking different implementations

## Requirements
- CUDA-enabled GPU
- NVIDIA CUDA Toolkit
- C++ Compiler with CUDA support (e.g., NVCC)


## License
This repository is for learning purposes and is open for contributions!

