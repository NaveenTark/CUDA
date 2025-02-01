#include <iostream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>

__global__ void whoami(void)
{
	int block_id = blockIdx.x + (gridDim.x * gridDim.y) * blockIdx.z + gridDim.x * (blockIdx.y);
	int block_offset = block_id * blockDim.x * blockDim.y * blockDim.z;
	int thread_offset = threadIdx.x + (blockDim.x * blockDim.y) * threadIdx.z +blockDim.x * (threadIdx.y);
	int id = block_offset + thread_offset; 

	printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
		id,
		blockIdx.x, blockIdx.y, blockIdx.z, block_id,
		threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}



int main()
{
	const int b_x = 3, b_y = 4, b_z = 2;
	const int t_x = 2, t_y = 3, t_z = 4;

	dim3 blocksPerGrid(b_x, b_y, b_z);
	dim3 threadsPerBlock(t_x, t_y, t_z);
	whoami <<<blocksPerGrid, threadsPerBlock >>> ();


}