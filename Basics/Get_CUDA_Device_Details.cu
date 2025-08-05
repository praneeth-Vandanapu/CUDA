/**
 * @file        Get_CUDA_Device_Details.cu
 * @brief       A simple CUDA program to get the Complete deteails of GPU.
 * @details     This program retrieves and displays detailed information about all available CUDA devices on the system.
 * @author      Praneeth Vandanapu
 * @date        July 23, 2025
 * @Linkedin    https://www.linkedin.com/in/praneeth-vandanapu-28889419b/
 * @license     MIT License (see LICENSE file in root directory)
 */

 // C Code Snippet
/*
#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        return 1;
    }
    
    printf("=== CUDA Device Information ===\n");
    printf("Number of CUDA devices: %d\n\n", deviceCount);
    
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        
        printf("Device %d: %s\n", dev, prop.name);
        printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("  Total Global Memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("  Registers per Block: %d\n", prop.regsPerBlock);
        printf("  Warp Size: %d\n", prop.warpSize);
        printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("  Max Block Dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dimensions: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("  Peak Memory Bandwidth: %.2f GB/s\n", 
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
        printf("  Number of SMs: %d\n", prop.multiProcessorCount);
        printf("  Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  L2 Cache Size: %d bytes\n", prop.l2CacheSize);
        printf("  Concurrent Kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("  Unified Addressing: %s\n", prop.unifiedAddressing ? "Yes" : "No");
        printf("  ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("  PCI Bus ID: %d\n", prop.pciBusID);
        printf("  PCI Device ID: %d\n", prop.pciDeviceID);
        printf("  Texture Alignment: %zu bytes\n", prop.textureAlignment);
        printf("  Surface Alignment: %zu bytes\n", prop.surfaceAlignment);
        printf("  Kernel Execution Timeout: %s\n", prop.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU: %s\n", prop.integrated ? "Yes" : "No");
        printf("  Can Map Host Memory: %s\n", prop.canMapHostMemory ? "Yes" : "No");
        printf("  Async Engine Count: %d\n", prop.asyncEngineCount);
        printf("\n");
    }
    
    printf("Program completed successfully.\n");
    return 0;
}*/

 // CPP code snippet

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

int main() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA devices found!" << std::endl;
        return 1;
    }

    std::cout << "=== CUDA Device Information ===\n";
    std::cout << "Number of CUDA devices: " << deviceCount << "\n\n";

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);

        std::cout << "Device " << dev << ": " << prop.name << "\n";
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "  Total Global Memory: " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << "  Shared Memory per Block: " << prop.sharedMemPerBlock << " bytes\n";
        std::cout << "  Registers per Block: " << prop.regsPerBlock << "\n";
        std::cout << "  Warp Size: " << prop.warpSize << "\n";
        std::cout << "  Max Threads per Block: " << prop.maxThreadsPerBlock << "\n";
        std::cout << "  Max Block Dimensions: (" 
                  << prop.maxThreadsDim[0] << ", " 
                  << prop.maxThreadsDim[1] << ", " 
                  << prop.maxThreadsDim[2] << ")\n";
        std::cout << "  Max Grid Dimensions: ("
                  << prop.maxGridSize[0] << ", "
                  << prop.maxGridSize[1] << ", "
                  << prop.maxGridSize[2] << ")\n";
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate * 1e-3f << " MHz\n";
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits\n";
        std::cout << "  Peak Memory Bandwidth: "
                  << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << " GB/s\n";
        std::cout << "  Number of SMs: " << prop.multiProcessorCount << "\n";
        std::cout << "  Max Threads per SM: " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << "  L2 Cache Size: " << prop.l2CacheSize << " bytes\n";
        std::cout << "  Concurrent Kernels: " << (prop.concurrentKernels ? "Yes" : "No") << "\n";
        std::cout << "  Unified Addressing: " << (prop.unifiedAddressing ? "Yes" : "No") << "\n";
        std::cout << "  ECC Enabled: " << (prop.ECCEnabled ? "Yes" : "No") << "\n";
        std::cout << "  PCI Bus ID: " << prop.pciBusID << "\n";
        std::cout << "  PCI Device ID: " << prop.pciDeviceID << "\n";
        std::cout << "  Texture Alignment: " << prop.textureAlignment << " bytes\n";
        std::cout << "  Surface Alignment: " << prop.surfaceAlignment << " bytes\n";
        std::cout << "  Kernel Execution Timeout: " << (prop.kernelExecTimeoutEnabled ? "Yes" : "No") << "\n";
        std::cout << "  Integrated GPU: " << (prop.integrated ? "Yes" : "No") << "\n";
        std::cout << "  Can Map Host Memory: " << (prop.canMapHostMemory ? "Yes" : "No") << "\n";
        std::cout << "  Async Engine Count: " << prop.asyncEngineCount << "\n\n";
    }

    std::cout << "Program completed successfully." << std::endl;
    return 0;
}
