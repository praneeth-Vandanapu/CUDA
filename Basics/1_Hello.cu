/**
 * @file        Hello.cu
 * @brief       A simple CUDA "Hello World" kernel program
 * @details     This program demonstrates how to launch a basic CUDA kernel 
 *              that prints a message from the GPU side. It uses <<<1, 1>>> launch
 *              configuration — i.e., one block with one thread.
 *
 * @author      Praneeth Vandanapu
 * @date        July 23, 2025
 * @Linkedin    https://www.linkedin.com/in/praneeth-vandanapu-28889419b/
 * @license     MIT License (see LICENSE file in root directory)
 */

#include <stdio.h>      // For printf
#include <cuda_runtime.h> // For cudaDeviceSynchronize and CUDA API functions

/**
 * @brief CUDA Kernel function
 * 
 * A kernel is a function that runs on the GPU. This kernel uses the `printf` 
 * function to print a simple message. Note that this output appears in the 
 * host console only after the kernel finishes and synchronizes.
 */
__global__ void helloCUDA() {
    printf("Hello from CUDA kernel!\n");
}

int main() {
    // Launch the kernel with <<<1, 1>>>
    // Syntax: <<<number_of_blocks, number_of_threads_per_block>>>
    helloCUDA<<<1, 1>>>();

    // Ensure the host waits for the device (GPU) to finish execution
    cudaDeviceSynchronize();

    return 0;
}
