/**
 * @file        CPU_GPU_Vector_addition.cu
 * @brief       Compares CPU and GPU vector addition performance with timing
 * @details     Measures time taken by CPU and CUDA-based vector addition,
 *              prints timestamps and durations for comparison.
 *
 * @author      Praneeth Vandanapu
 * @date        July 23, 2025
 * @license     MIT License
 */

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#define N 10000000  // 10 million elements

// CUDA kernel to add two vectors
__global__ void vectorAddCUDA(const int *A, const int *B, int *C, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n)
        C[idx] = A[idx] + B[idx];
}

// CPU function to add two vectors
void vectorAddCPU(const int *A, const int *B, int *C, int n) {
    for (int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

// Utility to print current time and message
void log_timestamp(const char* message) {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
    printf("[Time %lld ms] %s\n", value.count(), message);
}

int main() {
    log_timestamp("Program started.");

    // Allocate host memory
    int *h_A = (int*)malloc(N * sizeof(int));
    int *h_B = (int*)malloc(N * sizeof(int));
    int *h_C_CPU = (int*)malloc(N * sizeof(int));
    int *h_C_GPU = (int*)malloc(N * sizeof(int));

    // Initialize host input vectors
    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = 2 * i;
    }

    // ============ CPU Execution =============
    log_timestamp("Starting CPU vector addition...");
    auto cpu_start = std::chrono::high_resolution_clock::now();

    vectorAddCPU(h_A, h_B, h_C_CPU, N);

    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();
    log_timestamp("CPU vector addition completed.");

    printf("🔹 CPU Execution Time: %lld ms\n\n", cpu_duration);

    // Pause before GPU run
    log_timestamp("Sleeping for 3 seconds before starting GPU part...");
    std::this_thread::sleep_for(std::chrono::seconds(3));

    // ============ GPU Execution =============
    log_timestamp("Starting GPU vector addition...");

    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, N * sizeof(int));
    cudaMalloc((void**)&d_B, N * sizeof(int));
    cudaMalloc((void**)&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    auto gpu_start = std::chrono::high_resolution_clock::now();

    vectorAddCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();

    cudaMemcpy(h_C_GPU, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);
    log_timestamp("GPU vector addition completed.");

    printf("🔹 GPU Execution Time (excluding memcpy): %lld ms\n\n", gpu_duration);

    // ============ Validation =============
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_C_CPU[i] != h_C_GPU[i]) {
            printf("Mismatch at index %d: CPU=%d, GPU=%d\n", i, h_C_CPU[i], h_C_GPU[i]);
            success = false;
            break;
        }
    }

    if (success)
        printf("Vector addition is correct!\n");

    // Free all memory
    free(h_A); free(h_B); free(h_C_CPU); free(h_C_GPU);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    log_timestamp("Program finished.\n");
    return 0;
}
