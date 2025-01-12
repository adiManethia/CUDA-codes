import torch
import time

# Matrix dimensions for multiplication
# For matrices: (M x N) * (N x K) = (M x K)
M, N, K = 1024, 1024, 1024  # Using 1024 as it's a power of 2, good for GPU computation

# Function to measure execution time of GPU operations
def measure_time(func, *args, desc=""):
    # Ensure all previous GPU operations are completed
    torch.cuda.synchronize()
    
    # Record start time
    start = time.time()
    
    # Execute the function with provided arguments
    func(*args)
    
    # Ensure all GPU operations are completed before stopping timer
    torch.cuda.synchronize()
    
    # Record end time and calculate duration
    end = time.time()
    print(f"{desc} Execution Time: {(end - start) * 1000:.2f} ms ")

# ====== CUDA Cores Matrix Multiplication (FP32) ======
# Generate random matrices with 32-bit floating point precision
# These will run on regular CUDA cores
A_fp32 = torch.randn(M, N, device="cuda", dtype=torch.float32)  # Matrix on GPU with FP32
B_fp32 = torch.randn(N, K, device="cuda", dtype=torch.float32)  # Matrix on GPU with FP32

# Function to perform matrix multiplication using FP32 precision
def matmul_fp32(A, B):
    return torch.matmul(A, B)  # Standard matrix multiplication on CUDA cores

# Measure performance of FP32 matrix multiplication
measure_time(matmul_fp32, A_fp32, B_fp32, desc="CUDA-Cores (FP32)")

# ====== Tensor Cores Matrix Multiplication (FP16) ======
# Convert FP32 matrices to FP16 (half precision)
# Tensor Cores can use FP16 for faster computation
A_fp16 = A_fp32.half()  # Convert to FP16 precision
B_fp16 = B_fp32.half()  # Convert to FP16 precision

# Function to perform matrix multiplication using FP16 precision
# This will automatically utilize Tensor Cores if available
def matmul_fp16(A, B):
    return torch.matmul(A, B)  # Matrix multiplication using Tensor Cores

# Measure performance of FP16 matrix multiplication
measure_time(matmul_fp16, A_fp16, B_fp16, desc="Tensor-Cores (FP16)")

# ====== Result Comparison and Validation ======
# Compute results using both methods
C_fp32 = matmul_fp32(A_fp32, B_fp32)  # Result using CUDA cores (FP32)
C_fp16 = matmul_fp16(A_fp16, B_fp16)  # Result using Tensor cores (FP16)

# Convert FP16 result back to FP32 for comparison
C_fp16_converted = C_fp16.float()

# Calculate maximum absolute difference between results
# This helps verify if the reduced precision significantly affects accuracy
difference = torch.abs(C_fp32 - C_fp16_converted).max().item()
print(f"Maximum Difference between FP32 and FP16 results: {difference:.6e}")