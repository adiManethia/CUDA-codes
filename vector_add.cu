#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <chrono>

// Kernel function that runs on the GPU
// __global__ indicates this is a CUDA kernel that runs on the device (GPU)
// The function must return void and be declared with __global__
__global__ void vectorAddKernel(const float* A, const float* B, float* C, int n) {
    // Calculate global thread ID
    // threadIdx.x is the thread index within a block
    // blockIdx.x is the block index
    // blockDim.x is the number of threads per block
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Check if this thread should process data (prevent buffer overflow)
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

// CPU version of vector addition for comparison
void vectorAddCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int n) {
    // Simple sequential addition
    for (int i = 0; i < n; ++i) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Problem size: 2^20 = 1,048,576 elements
    // Try increasing this to see better GPU performance
    const int N = 1 << 24; // 16 million elements
    
    // Number of threads per block (must be a power of 2)
    // This is a common block size that usually gives good performance
    const int BLOCK_SIZE = 256;

    // Initialize host (CPU) vectors
    // Using std::vector for automatic memory management on host side
    std::vector<float> h_A(N, 1.0f);    // First input vector, initialized with 1.0
    std::vector<float> h_B(N, 2.0f);    // Second input vector, initialized with 2.0
    std::vector<float> h_C(N, 0.0f);    // Output vector for CPU results
    std::vector<float> h_C_GPU(N, 0.0f); // Output vector for GPU results

    // Declare device (GPU) pointers
    float *d_A, *d_B, *d_C;

    // Start timing total GPU operations including memory transfers
    auto start_gpu_total = std::chrono::high_resolution_clock::now();

    // Allocate memory on GPU
    // cudaMalloc is similar to malloc but allocates on GPU
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Copy data from host to device (CPU to GPU)
    cudaMemcpy(d_A, h_A.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    // We need enough blocks to cover N elements with BLOCK_SIZE threads per block
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE); // Ceiling division to ensure we have enough blocks

    // Warm-up run to initialize GPU
    vectorAddKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Time kernel execution only
    auto start_kernel = std::chrono::high_resolution_clock::now();
    
    // Launch kernel
    // <<<grid, block>>> is CUDA syntax for kernel launch configuration
    vectorAddKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    auto end_kernel = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(h_C_GPU.data(), d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    auto end_gpu_total = std::chrono::high_resolution_clock::now();

    // Calculate and print GPU timings
    std::chrono::duration<float> duration_kernel = end_kernel - start_kernel;
    std::chrono::duration<float> duration_gpu_total = end_gpu_total - start_gpu_total;

    std::cout << "GPU kernel execution time: " << duration_kernel.count() * 1000 << "ms" << std::endl;
    std::cout << "GPU total time (including memory transfers): " << duration_gpu_total.count() * 1000 << "ms" << std::endl;

    // Time CPU version
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    vectorAddCPU(h_A, h_B, h_C, N);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration_cpu = end_cpu - start_cpu;
    std::cout << "CPU execution time: " << duration_cpu.count() * 1000 << "ms" << std::endl;

    // Verify results
    bool match = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - h_C_GPU[i]) > 1e-5) {
            match = false;
            std::cout << "Results do not match at index " << i << "!" << std::endl;
            std::cout << "CPU: " << h_C[i] << ", GPU: " << h_C_GPU[i] << std::endl;
            break;
        }
    }
    if (match) {
        std::cout << "Results match! Verification successful!" << std::endl;
    }

    // Free GPU memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Calculate and print speedup
    float speedup = duration_cpu.count() / duration_kernel.count();
    std::cout << "GPU Speedup (kernel only): " << speedup << "x" << std::endl;
    
    float total_speedup = duration_cpu.count() / duration_gpu_total.count();
    std::cout << "GPU Speedup (including memory transfers): " << total_speedup << "x" << std::endl;

    return 0;
}