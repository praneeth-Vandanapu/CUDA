/**
 * @file        Day02_Vector_addition.cu
 * @brief       Parallel Vector Addition using CUDA
 * @details     This program adds two vectors using GPU parallelism.
 *              Each CUDA thread handles one element of the result.
 *
 * @author      Praneeth Vandanapu
 * @date        July 23, 2025
 * @license     MIT License
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 256  // Size of vectors

/**
 * @brief CUDA kernel to add two vectors element-wise
 * 
 * Each thread computes one element of the result vector.
 */
__global__ void vectorAdd(const int *A, const int *B, int *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Host vectors (on CPU)
    int h_A[N], h_B[N], h_C[N];

    // Device vectors (on GPU)
    int *d_A, *d_B, *d_C;

    // Initialize host input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    // Allocate device memory
    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: enough threads to cover all elements
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the first 10 results
    printf("First 10 results of vector addition:\n");
    for (int i = 0; i < 256; i++) {
        printf("h_C[%d] = %d\n", i, h_C[i]);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
