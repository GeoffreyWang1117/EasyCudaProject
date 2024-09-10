#include <cufft.h>
#include <iostream>

// 定义 FFT 的大小
#define SIZE 8

// 检查 cuFFT API 调用的返回状态
static void checkCudaError(cufftResult status, const char* message) {
    if (status != CUFFT_SUCCESS) {
        std::cerr << "Error: " << message << std::endl;
        exit(EXIT_FAILURE);
    }
}

void testFFT() {
    // 初始化主机上的输入数据（复数）
    cufftComplex h_signal[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        h_signal[i].x = float(i);  // 实部
        h_signal[i].y = 0.0f;      // 虚部
    }

    // 分配设备内存
    cufftComplex* d_signal;
    cudaMalloc((void**)&d_signal, sizeof(cufftComplex) * SIZE);

    // 将主机数据复制到设备
    cudaMemcpy(d_signal, h_signal, sizeof(cufftComplex) * SIZE, cudaMemcpyHostToDevice);

    // 创建 cuFFT 计划
    cufftHandle plan;
    checkCudaError(cufftPlan1d(&plan, SIZE, CUFFT_C2C, 1), "Failed to create plan");

    // 执行 FFT
    checkCudaError(cufftExecC2C(plan, d_signal, d_signal, CUFFT_FORWARD), "Failed to execute FFT");

    // 将结果从设备复制回主机
    cudaMemcpy(h_signal, d_signal, sizeof(cufftComplex) * SIZE, cudaMemcpyDeviceToHost);

    // 打印 FFT 结果
    std::cout << "FFT result:\n";
    for (int i = 0; i < SIZE; ++i) {
        std::cout << "Element " << i << ": " << h_signal[i].x << " + " << h_signal[i].y << "i\n";
    }

    // 销毁 cuFFT 计划并释放内存
    cufftDestroy(plan);
    cudaFree(d_signal);
}
