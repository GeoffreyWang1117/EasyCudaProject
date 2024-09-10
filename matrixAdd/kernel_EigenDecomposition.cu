#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <iostream>
#include "matrix_utils.cuh"  // 包含 printMatrix 函数

static void eigenDecomposition(float* h_A, int n) {
    cusolverDnHandle_t cusolverH = NULL;
    float* d_A = NULL;
    float* d_W = NULL;  // 存储特征值
    int* d_info = NULL;
    int lwork = 0;
    float* d_work = NULL;
    int* h_info = NULL;
    int lda = n;

    // 分配主机内存用于存储特征值
    float* h_W = (float*)malloc(n * sizeof(float));  // 分配 n 个 float 大小的数组，存放特征值

    // 初始化 cuSolver 句柄
    cusolverDnCreate(&cusolverH);

    // 分配设备内存
    cudaMalloc((void**)&d_A, sizeof(float) * n * n);
    cudaMalloc((void**)&d_W, sizeof(float) * n);
    cudaMalloc((void**)&d_info, sizeof(int));
    h_info = (int*)malloc(sizeof(int));

    // 将矩阵 A 复制到设备
    cudaMemcpy(d_A, h_A, sizeof(float) * n * n, cudaMemcpyHostToDevice);

    // 查询特征值/特征向量计算的工作空间大小
    cusolverDnSsyevd_bufferSize(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_A, lda, d_W, &lwork);

    cudaMalloc((void**)&d_work, sizeof(float) * lwork);

    // 计算特征值和特征向量
    cusolverDnSsyevd(cusolverH, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, d_A, lda, d_W, d_work, lwork, d_info);

    // 复制结果回主机
    cudaMemcpy(h_A, d_A, sizeof(float) * n * n, cudaMemcpyDeviceToHost);  // 特征向量
    cudaMemcpy(h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_W, d_W, sizeof(float) * n, cudaMemcpyDeviceToHost);  // 特征值

    if (*h_info != 0) {
        std::cerr << "Eigenvalue decomposition failed!" << std::endl;
    }
    else {
        std::cout << "Eigenvalue decomposition succeeded!" << std::endl;
        std::cout << "Eigenvalues:" << std::endl;
        for (int i = 0; i < n; ++i) {
            std::cout << h_W[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "Eigenvectors (in A):" << std::endl;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                std::cout << h_A[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // 释放内存
    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_info);
    cudaFree(d_work);
    free(h_info);
    cusolverDnDestroy(cusolverH);
}


void testEigenDecomposition(int size) {
    float* h_A = new float[size * size];

    std::cout << "Enter elements of the matrix:\n";
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            std::cin >> h_A[i * size + j];
        }
    }
    printMatrix(h_A, size,size);
    eigenDecomposition(h_A, size);  // 调用之前的特征值分解函数

    delete[] h_A;
}
