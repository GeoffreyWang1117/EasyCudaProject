
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>  // 用于格式化输出
#include "matrix_utils.cuh"  // 包含打印矩阵的函数

// CUDA Kernel for Matrix Addition
static static __global__ void matrixAdd(float* a, float* b, float* c, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int index = idy * width + idx;

    if (idx < width && idy < height) {
        c[index] = a[index] + b[index];
    }
}

void testMatrixAddition(int rows, int cols) {
    // 初始化矩阵 A 和 B
    float* h_a = new float[rows * cols];
    float* h_b = new float[rows * cols];
    float* h_c = new float[rows * cols];  // 用于存放结果的矩阵 C

    std::cout << "Enter elements of matrix A:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cin >> h_a[i * cols + j];
        }
    }

    // 确认矩阵 A 是否已正确输入
   // std::cout << "Before calling printMatrix: Matrix A[0] = " << h_a[0] << std::endl;

    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_a, rows, cols);

    std::cout << "Enter elements of matrix B:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cin >> h_b[i * cols + j];
        }
    }

    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_b, rows, cols);

    // 设备上的指针
    float* d_a, * d_b, * d_c;

    // 为设备上的矩阵分配内存
    cudaMalloc((void**)&d_a, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_b, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_c, rows * cols * sizeof(float));

    // 将主机上的矩阵 A 和 B 复制到设备上
    cudaMemcpy(d_a, h_a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块大小和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动 CUDA 核函数执行矩阵加法
    matrixAdd << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, cols, rows);

    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果矩阵 C
    std::cout << "Matrix C (Result of A + B):\n";
    printMatrix(h_c, rows, cols);

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}
