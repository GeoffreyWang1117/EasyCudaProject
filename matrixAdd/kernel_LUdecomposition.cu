#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>

#include "matrix_utils.cuh"  // 包含 printMatrix 函数

// LU 分解的 CUDA 函数
static void LUdecomposition(float* h_A, int n) {
    cusolverDnHandle_t cusolverH = NULL;
    float* d_A = NULL;
    int* d_info = NULL;
    int* h_info = NULL;
    int* devIpiv = NULL; // pivoting indices
    float* d_work = NULL;
    int work_size = 0;
    int lda = n;

    // 初始化 cuSolver 句柄
    cusolverDnCreate(&cusolverH);

    // 分配设备内存
    cudaMalloc((void**)&d_A, sizeof(float) * n * n);
    cudaMalloc((void**)&d_info, sizeof(int));
    cudaMalloc((void**)&devIpiv, sizeof(int) * n);
    h_info = (int*)malloc(sizeof(int));

    // 将矩阵 A 复制到设备
    cudaMemcpy(d_A, h_A, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // 查询 LU 分解的 workspace 大小
    cusolverDnSgetrf_bufferSize(cusolverH, n, n, d_A, lda, &work_size);

    cudaMalloc((void**)&d_work, sizeof(float) * work_size);

    // 进行 LU 分解
    cusolverDnSgetrf(cusolverH, n, n, d_A, lda, d_work, devIpiv, d_info);

    // 复制分解结果回主机
    cudaMemcpy(h_A, d_A, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    // 检查结果是否成功
    if (*h_info != 0) {
        std::cerr << "LU decomposition failed!" << std::endl;
    }
    else {
        std::cout << "LU decomposition succeeded!" << std::endl;
        std::cout << "L and U matrices (combined) in A:" << std::endl;
        printMatrix(h_A, n, n);
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_info);
    cudaFree(devIpiv);
    cudaFree(d_work);
    free(h_info);
    cusolverDnDestroy(cusolverH);
}

void testLUdecomposition(int size) {
    float* h_A = new float[size * size];

    std::cout << "Enter elements of the matrix:\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cin >> h_A[i * size + j];
        }
    }
    printMatrix(h_A, size, size);
    LUdecomposition(h_A, size);  // 调用之前的 LU 分解函数

    delete[] h_A;
}
