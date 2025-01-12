#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

// Matrix dimensions as constants
const int M = 1024;  // Rows of matrix A and result matrix C
const int N = 1024;  // Columns of matrix A and rows of matrix B
const int K = 1024;  // Columns of matrix B and result matrix C

// CUDA kernel for matrix multiplication
__global__ void matrixMulCUDA(const float* A, const float* B, float* C, int m, int n, int k) {
    // Calculate global thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread is within matrix bounds
    if (row < m && col < k) {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
            sum += A[row * n + i] * B[i * k + col];
        }
        C[row * k + col] = sum;
    }
}

// Helper function to initialize matrices
void initializeMatrix(std::vector<float>& mat, int rows, int cols, float value) {
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = value;
    }
}

// Macro for timing measurements
#define TIMEIT(start, end) std::chrono::duration<float, std::milli>(end-start).count()

int main() {
    // Allocate host memory
    std::vector<float> h_A(M * N);
    std::vector<float> h_B(N * K);
    std::vector<float> h_C(M * K, 0.0f);
    std::vector<float> h_C_CPU(M * K, 0.0f);

    // Initialize matrices
    initializeMatrix(h_A, M, N, 1.0f);
    initializeMatrix(h_B, N, K, 2.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_A, h_A.data(), M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 blockDim(16, 16);
    dim3 gridDim((K + blockDim.x - 1) / blockDim.x, 
                 (M + blockDim.y - 1) / blockDim.y);

    // Warm-up run
    matrixMulCUDA<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    // Time GPU implementation
    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCUDA<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    float gpu_time = TIMEIT(start, end);
    std::cout << "GPU Time: " << gpu_time << " ms" << std::endl;

    // Copy result back to host
    cudaMemcpy(h_C.data(), d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // CPU implementation
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                sum += h_A[i * N + k] * h_B[k * K + j];
            }
            h_C_CPU[i * K + j] = sum;
        }
    }
    end = std::chrono::high_resolution_clock::now();
    
    float cpu_time = TIMEIT(start, end);
    std::cout << "CPU Time: " << cpu_time << " ms" << std::endl;

    // Verify results
    bool match = true;
    float maxError = 0.0f;
    for (int i = 0; i < M * K; ++i) {
        float error = fabs(h_C[i] - h_C_CPU[i]);
        maxError = std::max(maxError, error);
        if (error > 1e-5) {
            match = false;
            std::cout << "Mismatch at index " << i 
                     << ": GPU=" << h_C[i] 
                     << ", CPU=" << h_C_CPU[i] << std::endl;
            break;
        }
    }

    if (match) {
        std::cout << "Results match! Max error: " << maxError << std::endl;
        std::cout << "Speedup: " << cpu_time / gpu_time << "x" << std::endl;
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}