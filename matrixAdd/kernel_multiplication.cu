#include <cuda_runtime.h>
#include <iostream>

#define M 3  // 矩阵 A 的行数
#define K 3  // 矩阵 A 的列数 (矩阵 B 的行数)
#define N 3  // 矩阵 B 的列数

// CUDA 核函数用于执行矩阵乘法
static static __global__ void matrixMultiply(float* a, float* b, float* c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float result = 0.0;
        for (int i = 0; i < k; ++i) {
            result += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = result;
    }
}

void testMatrixMultiplication() {
    // 初始化矩阵 A 和 B
    float h_a[M * K] = { 1, 2, 3, 4, 5, 6, 7, 8, 9 };  // 3x3 矩阵 A
    float h_b[K * N] = { 9, 8, 7, 6, 5, 4, 3, 2, 1 };  // 3x3 矩阵 B
    float h_c[M * N];  // 用于存放结果的矩阵 C

    // 设备上的指针
    float* d_a, * d_b, * d_c;

    // 分配设备内存
    cudaMalloc((void**)&d_a, M * K * sizeof(float));
    cudaMalloc((void**)&d_b, K * N * sizeof(float));
    cudaMalloc((void**)&d_c, M * N * sizeof(float));

    // 将主机数据复制到设备
    cudaMemcpy(d_a, h_a, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // 设置线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动 CUDA 核函数执行矩阵乘法
    matrixMultiply << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_c, M, K, N);

    // 将结果从设备复制回主机
    cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 打印结果矩阵 C
    std::cout << "Matrix C (Result of A * B):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_c[i * N + j] << " ";
        }
        std::cout << "\n";
    }

    // 释放设备内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
