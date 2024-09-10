#include <iostream>
#include "matrix_utils.cuh"  // 包含打印矩阵的函数
static static __global__ void matrixTranspose(float* a, float* b, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        b[idx * height + idy] = a[idy * width + idx];
    }
}

void testMatrixTranspose(int rows, int cols) {
    // 初始化矩阵 A
    float* h_a = new float[rows * cols];
    float* h_b = new float[cols * rows];  // 用于存放结果的转置矩阵

    std::cout << "Enter elements of matrix A:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cin >> h_a[i * cols + j];
        }
    }

    // 设备上的指针
    float* d_a, * d_b;

    // 为设备上的矩阵分配内存
    cudaMalloc((void**)&d_a, rows * cols * sizeof(float));
    cudaMalloc((void**)&d_b, cols * rows * sizeof(float));

    // 将主机上的矩阵 A 复制到设备上
    cudaMemcpy(d_a, h_a, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块大小和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动 CUDA 核函数执行矩阵转置
    matrixTranspose << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, cols, rows);

    // 将结果从设备复制回主机
    cudaMemcpy(h_b, d_b, cols * rows * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果矩阵 B
    std::cout << "Matrix B (Transpose of A):\n";
    printMatrix(h_b, cols, rows);

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    delete[] h_a;
    delete[] h_b;
}
