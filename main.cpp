#include <iostream>
#include <cuda_runtime.h>
#include "kernel.cu" // 包含所有的核函数

void runCUDA(int operationType, float* inputA, float* inputB, int size) {
    // 根据指令类型选择要执行的CUDA操作
    switch (operationType) {
    case 0: // Vector Add
        vectorAdd << <blocks, threads >> > (inputA, inputB, result, size);
        break;
    case 1: // Matrix Multiply
        matrixMultiply << <dim3(N, N), dim3(N, N) >> > (inputA, inputB, result, N);
        break;
    case 2: // Matrix Transpose
        matrixTranspose << <dim3(N, N), dim3(N, N) >> > (inputA, result, N);
        break;
    case 3: // FFT
        FFT << <blocks, threads >> > (inputA, size);
        break;
    case 4: // Sparse Matrix-Vector Multiply
        sparseMatrixVectorMultiply << <blocks, threads >> > (val, row, col, inputB, result, size);
        break;
        // 添加其他操作
    }
}

int main() {
    // 假设用户输入0, 1, 2等操作类型，并根据操作选择对应的核函数
    int operationType;
    std::cout << "Select operation: 0 for Vector Add, 1 for Matrix Multiply, etc: ";
    std::cin >> operationType;

    float* inputA, * inputB;
    // 初始化输入，分配GPU内存
    runCUDA(operationType, inputA, inputB, size);

    return 0;
}
