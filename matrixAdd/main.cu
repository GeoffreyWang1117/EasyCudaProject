#include <iostream>
#include <string>
#include "matrix_utils.cuh"  // 包含矩阵打印、文件读取/写入功能

// Function declarations from other CUDA files
void testMatrixAddition(int rows, int cols);
void testMatrixMultiplication();
void testFFT();
void testMatrixTranspose(int rows, int cols);
void testMatrixInverse(int size);
void testLUdecomposition(int size);
void testQRdecomposition(int rows, int cols);
void testEigenDecomposition(int size);

int main() {
    int operationType;
    int rows = 0, cols = 0;
    float* matrixA = nullptr;
    float* matrixB = nullptr;
    bool matrixLoaded = false;

    // 询问用户是否有矩阵数据文件
    std::string filename;
    std::cout << "Do you have a matrix data file? (y/n): ";
    char fileChoice;
    std::cin >> fileChoice;

    if (fileChoice == 'y' || fileChoice == 'Y') {
        std::cout << "Enter the matrix file name: ";
        std::cin >> filename;

        // 从文件读取矩阵数据
        if (!readMatrixFromFile(filename, matrixA, rows, cols)) {
            std::cerr << "Failed to load matrix from file." << std::endl;
            return -1;  // 如果读取失败，退出程序
        }
        matrixLoaded = true;
        std::cout << "Matrix successfully loaded from " << filename << "!" << std::endl;
        printMatrix(matrixA, rows, cols);  // 打印加载的矩阵
    }

    std::cout << "Select operation: 1 for Matrix Addition, 2 for Matrix Multiplication, 3 for FFT, 4 for Matrix Transpose, 5 for Matrix Inverse, ";
    std::cout << "6 for LU Decomposition, 7 for QR Decomposition, 8 for Eigenvalue Decomposition: ";
    std::cin >> operationType;

    if (!matrixLoaded) {
        // 如果没有从文件加载矩阵，则从命令行输入
        if (operationType == 1 || operationType == 2) {
            std::cout << "Enter the number of rows: ";
            std::cin >> rows;
            std::cout << "Enter the number of columns: ";
            std::cin >> cols;
        }
        matrixA = new float[rows * cols];
        std::cout << "Enter elements of matrix A:\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cin >> matrixA[i * cols + j];
            }
        }
    }

    // 根据用户选择调用不同的操作函数
    if (operationType == 1) {
        testMatrixAddition(rows, cols);
    }
    else if (operationType == 2) {
        testMatrixMultiplication();
    }
    else if (operationType == 3) {
        testFFT();
    }
    else if (operationType == 4) {
        testMatrixTranspose(rows, cols);
    }
    else if (operationType == 5) {
        std::cout << "Enter the size of the matrix (must be square): ";
        std::cin >> rows;  // Inverse only for square matrices
        if (rows != cols) {
            std::cerr << "Matrix must be square for inversion!" << std::endl;
            return -1;
        }
        testMatrixInverse(rows);
    }
    else if (operationType == 6) {
        testLUdecomposition(rows);
    }
    else if (operationType == 7) {
        testQRdecomposition(rows, cols);
    }
    else if (operationType == 8) {
        testEigenDecomposition(rows);
    }
    else {
        std::cerr << "Invalid operation type selected!" << std::endl;
    }

    // 提示用户是否保存结果到文件
    std::cout << "Do you want to save the matrix result to a file? (y/n): ";
    char saveChoice;
    std::cin >> saveChoice;

    if (saveChoice == 'y' || saveChoice == 'Y') {
        std::string saveFilename;
        std::cout << "Enter the file name to save the result: ";
        std::cin >> saveFilename;

        if (writeMatrixToFile(saveFilename, matrixA, rows, cols)) {
            std::cout << "Matrix successfully saved to " << saveFilename << "!" << std::endl;
        }
        else {
            std::cerr << "Failed to save matrix to file." << std::endl;
        }
    }

    // 清理内存
    delete[] matrixA;
    if (matrixB) delete[] matrixB;

    return 0;
}
