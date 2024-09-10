#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include "matrix_utils.cuh"  // 包含 printMatrix 函数

static void QRdecomposition(float* h_A, int m, int n) {
    cusolverDnHandle_t cusolverH = NULL;
    float* d_A = NULL;
    float* d_tau = NULL;
    int* d_info = NULL;
    int* h_info = NULL;
    float* d_work = NULL;
    int work_size = 0;
    int lda = m;

    // 初始化 cuSolver 句柄
    cusolverDnCreate(&cusolverH);

    // 分配设备内存
    cudaMalloc((void**)&d_A, sizeof(float) * m * n);
    cudaMalloc((void**)&d_tau, sizeof(float) * n);  // Tau 是中间存储
    cudaMalloc((void**)&d_info, sizeof(int));
    h_info = (int*)malloc(sizeof(int));

    // 将矩阵 A 复制到设备
    cudaMemcpy(d_A, h_A, sizeof(float) * m * n, cudaMemcpyHostToDevice);

    // 查询 QR 分解的 workspace 大小
    cusolverDnSgeqrf_bufferSize(cusolverH, m, n, d_A, lda, &work_size);

    cudaMalloc((void**)&d_work, sizeof(float) * work_size);

    // 进行 QR 分解
    cusolverDnSgeqrf(cusolverH, m, n, d_A, lda, d_tau, d_work, work_size, d_info);

    // 复制分解结果回主机
    cudaMemcpy(h_A, d_A, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);

    // 检查结果是否成功
    if (*h_info != 0) {
        std::cerr << "QR decomposition failed!" << std::endl;
    }
    else {
        std::cout << "QR decomposition succeeded!" << std::endl;
        std::cout << "Q and R matrices (combined) in A:" << std::endl;
        printMatrix(h_A, m, n);
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_tau);
    cudaFree(d_info);
    cudaFree(d_work);
    free(h_info);
    cusolverDnDestroy(cusolverH);
}

void testQRdecomposition(int rows, int cols) {
    float* h_A = new float[rows * cols];

    std::cout << "Enter elements of the matrix:\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cin >> h_A[i * cols + j];
        }
    }
    printMatrix(h_A, rows, cols);
    QRdecomposition(h_A, rows, cols);  // 调用之前的 QR 分解函数

    delete[] h_A;
}
