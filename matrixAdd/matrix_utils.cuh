#ifndef MATRIX_UTILS_CUH
#define MATRIX_UTILS_CUH

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>

// 内联函数用于格式化输出矩阵
inline static void printMatrix(float* matrix, int rows, int cols) {
    std::cout << "Matrix:" << std::endl;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// 从文件中读取矩阵数据，成功返回 true，失败返回 false
inline static bool readMatrixFromFile(const std::string& filename, float*& matrix, int& rows, int& cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return false;
    }

    // 读取行列信息
    file >> rows >> cols;
    if (rows <= 0 || cols <= 0) {
        std::cerr << "Invalid matrix dimensions in file: " << filename << std::endl;
        return false;
    }

    // 分配矩阵内存
    matrix = new float[rows * cols];

    // 读取矩阵元素
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file >> matrix[i * cols + j];
        }
    }

    file.close();
    return true;
}

// 将矩阵写入文件，成功返回 true，失败返回 false
inline static bool writeMatrixToFile(const std::string& filename, const float* matrix, int rows, int cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return false;
    }

    // 写入行列信息
    file << rows << " " << cols << std::endl;

    // 写入矩阵元素
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            file << std::setw(8) << std::fixed << std::setprecision(4) << matrix[i * cols + j] << " ";
        }
        file << std::endl;
    }

    file.close();
    return true;
}

#endif  // MATRIX_UTILS_CUH
