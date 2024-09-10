#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include "matrix_utils.cuh"  // 包含打印矩阵的函数

static void invertMatrix(float* h_matrix, int n) {
    float* d_matrix, * d_invMatrix;
    int* P, * INFO;

    // 分配设备内存
    cudaMalloc((void**)&d_matrix, n * n * sizeof(float));
    cudaMalloc((void**)&d_invMatrix, n * n * sizeof(float));
    cudaMalloc((void**)&P, n * sizeof(int));
    cudaMalloc((void**)&INFO, sizeof(int));

    // 复制矩阵到设备
    cudaMemcpy(d_matrix, h_matrix, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // 创建 cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // LU 分解
    cublasSgetrfBatched(handle, n, &d_matrix, n, P, INFO, 1);

    // 求逆
    cublasSgetriBatched(handle, n, (const float**)&d_matrix, n, P, &d_invMatrix, n, INFO, 1);

    // 复制结果回主机
    cudaMemcpy(h_matrix, d_invMatrix, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放资源
    cudaFree(d_matrix);
    cudaFree(d_invMatrix);
    cudaFree(P);
    cudaFree(INFO);
    cublasDestroy(handle);
}

void testMatrixInverse(int size) {
    // 初始化方阵
    float* h_a = new float[size * size];

    std::cout << "Enter elements of the matrix (must be square):\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cin >> h_a[i * size + j];
        }
    }

    // 调用矩阵求逆
    invertMatrix(h_a, size);

    // 打印结果
    std::cout << "Inverse matrix:\n";
    printMatrix(h_a, size, size);

    // 释放内存
    delete[] h_a;
}
