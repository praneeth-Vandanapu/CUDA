/**
 * @file        CPU_vs_GPU_Vector_addition.cu
 * @brief       A simple CUDA kernel program to basic vector addition.
 * @details     This program demonstrates how to peform vector addition using both CPU and GPU.
*               It initializes two vectors, performs addition on the CPU, and then on the GPU,
 * @author      Praneeth Vandanapu
 * @date        July 23, 2025
 * @Linkedin    https://www.linkedin.com/in/praneeth-vandanapu-28889419b/
 * @license     MIT License (see LICENSE file in root directory)
 */
#include <iostream>
#include <chrono>
#include <thread>
#include <cuda_runtime.h>

#define N 10000000  // 1 crore elements

// CUDA kernel for vector addition with timestamp and thread info
__global__ void vectorAddGPU(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];

        // Print only a few threads for performance and clarity
        if (i % (n / 10) == 0) {
            printf("[GPU][Thread %d] Added a[%d] + b[%d] = %d\n", i, i, i, c[i]);
        }
    }
}

// CPU function for vector addition
void vectorAddCPU(int *a, int *b, int *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    std::cout << "----- Vector Addition: CPU vs GPU -----\n";
    std::cout << "Data size: " << N << " integers (~" << (N * sizeof(int)) / (1024 * 1024) << " MB)\n\n";

    // Allocate host memory
    int *h_a = new int[N];
    int *h_b = new int[N];
    int *h_c_cpu = new int[N];
    int *h_c_gpu = new int[N];

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    // -------- CPU Vector Addition --------
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(h_a, h_b, h_c_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();

    std::cout << "\n✅ CPU Addition Done.\n";
    std::cout << "⏱️ CPU Time Taken: " << duration_cpu << " ms\n";

    // Sleep before starting GPU to separate output
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // -------- GPU Vector Addition --------
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * sizeof(int));
    cudaMalloc(&d_b, N * sizeof(int));
    cudaMalloc(&d_c, N * sizeof(int));

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaDeviceSynchronize();
    auto start_gpu = std::chrono::high_resolution_clock::now();

    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    auto end_gpu = std::chrono::high_resolution_clock::now();
    auto duration_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_gpu - start_gpu).count();

    // Copy result back to host
    cudaMemcpy(h_c_gpu, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "\n✅ GPU Addition Done.\n";
    std::cout << "⏱️ GPU Time Taken: " << duration_gpu << " ms\n";

    // -------- Verify Result --------
    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (h_c_cpu[i] != h_c_gpu[i]) {
            std::cout << "❌ Mismatch at index " << i << ": CPU = " << h_c_cpu[i] << ", GPU = " << h_c_gpu[i] << "\n";
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "\n✅ Result Verified: CPU and GPU results match.\n";
    }

    // -------- Clean Up --------
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    std::cout << "\n🚀 Finished.\n";
    return 0;
}
