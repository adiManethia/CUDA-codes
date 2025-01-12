# CUDA and PyTorch Matrix Operations

This repository contains CUDA and PyTorch-based implementations for matrix operations and performance comparisons. It includes:

## 1. CUDA Matrix Multiplication (C++)
- Implements matrix multiplication using CUDA cores.
- Compares GPU performance with CPU performance.
- Demonstrates the use of CUDA kernels for parallel computation.

## 2. PyTorch Matrix Multiplication (Python)
- Compares matrix multiplication using CUDA cores (FP32) and Tensor cores (FP16).
- Highlights the performance benefits of Tensor cores for half-precision computations.
- Includes result validation to ensure accuracy between FP32 and FP16 computations.

---

## Requirements

### For the CUDA C++ Code:
- NVIDIA GPU with CUDA support.
- CUDA Toolkit installed.
- C++ compiler (e.g., `nvcc`).

### For the PyTorch Python Code:
- PyTorch installed with CUDA support.
- NVIDIA GPU with Tensor Core support (for FP16 acceleration).

---

## How to Run

### 1. CUDA C++ Code:
- Compile and run using `nvcc`:
  ```bash
  nvcc -o MatMul MatMul.cu
  ./MatMul
  ```

### 2. PyTorch Python Code:
- Run the Python script:
  ```bash
  python matmul_pytorch.py
  ```

---

## Features
- Demonstrates GPU acceleration for matrix operations.
- Compares CPU and GPU performance for matrix multiplication.
- Highlights the use of Tensor cores for FP16 computations in PyTorch.

---

## Notes
- Ensure your system has the required hardware and software for CUDA and PyTorch.
- The code includes result validation to ensure correctness of GPU computations.
