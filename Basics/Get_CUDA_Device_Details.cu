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
}