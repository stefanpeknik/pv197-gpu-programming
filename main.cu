/**
 * @file main.cu
 * @brief A simple CUDA program to add two arrays element-wise.
 *
 * This program demonstrates the use of CUDA to perform parallel computation
 * on the GPU. It adds two integer arrays element-wise and stores the result
 * in a third array.
 *
 * The program performs the following steps:
 * 1. Defines a CUDA kernel function to add two arrays.
 * 2. Allocates memory on the GPU for the input and output arrays.
 * 3. Copies the input arrays from the host (CPU) to the device (GPU).
 * 4. Launches the CUDA kernel to perform the addition on the GPU.
 * 5. Copies the result array from the device back to the host.
 * 6. Prints the result array.
 * 7. Frees the allocated GPU memory.
 *
 * @note This example uses a fixed array size of 5 for simplicity.
 *       The block size for the CUDA kernel launch is set to 256 threads.
 *
 * @author Stefan
 * @date 2023
 */
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel function to add elements of two arrays
__global__ void add(int *a, int *b, int *c, int n) {
  int index =
      threadIdx.x + blockIdx.x * blockDim.x; // Calculate global thread index
  if (index < n) {                           // Ensure index is within bounds
    c[index] = a[index] + b[index];          // Perform element-wise addition
  }
}

int main() {
  const int arraySize = 5;                        // Size of the arrays
  const int arrayBytes = arraySize * sizeof(int); // Size in bytes

  // Host arrays
  int h_a[arraySize] = {1, 2, 3, 4, 5};
  int h_b[arraySize] = {10, 20, 30, 40, 50};
  int h_c[arraySize]; // Result array

  // Device pointers
  int *d_a, *d_b, *d_c;

  // Allocate memory on the GPU
  cudaMalloc((void **)&d_a, arrayBytes);
  cudaMalloc((void **)&d_b, arrayBytes);
  cudaMalloc((void **)&d_c, arrayBytes);

  // Copy input arrays from host to device
  cudaMemcpy(d_a, h_a, arrayBytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, arrayBytes, cudaMemcpyHostToDevice);

  // Define block size and number of blocks
  int blockSize = 256;
  int numBlocks = (arraySize + blockSize - 1) / blockSize;

  // Launch the CUDA kernel
  add<<<numBlocks, blockSize>>>(d_a, d_b, d_c, arraySize);

  // Copy the result array from device to host
  cudaMemcpy(h_c, d_c, arrayBytes, cudaMemcpyDeviceToHost);

  // Print the result array
  std::cout << "Result: ";
  for (int i = 0; i < arraySize; i++) {
    std::cout << h_c[i] << " ";
  }
  std::cout << std::endl;

  // Free the allocated GPU memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}