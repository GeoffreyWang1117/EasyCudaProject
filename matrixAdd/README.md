
# matrixAdd - CUDA Matrix Operations

`matrixAdd` is a CUDA-based project that provides several matrix operations, including matrix addition, multiplication, FFT, matrix transpose, matrix inversion, LU decomposition, QR decomposition, and eigenvalue decomposition. The project supports reading matrix data from files and writing results to files.

## Features

- Matrix Addition
- Matrix Multiplication
- Fast Fourier Transform (FFT)
- Matrix Transpose
- Matrix Inversion (only for square matrices)
- LU Decomposition
- QR Decomposition
- Eigenvalue and Eigenvector Decomposition
- File I/O: Supports reading matrix data from a file and saving results to a file (`.txt` format).

## Prerequisites

Before you begin, ensure you have met the following requirements:

- **CUDA Toolkit**: Make sure you have the CUDA Toolkit installed on your machine.
- **cuBLAS** and **cuSolver**: These libraries are used for matrix decompositions and are included in the CUDA Toolkit.
- **C++ Compiler**: A C++ compiler supporting CUDA (such as `nvcc`).
- **Visual Studio (optional)**: If you're using Windows, you can use Visual Studio for compiling and running the project.

## Project Structure

```
matrixAdd/
├── kernel_addition.cu
├── kernel_multiplication.cu
├── kernel_fft.cu
├── kernel_lu_decomposition.cu
├── kernel_qr_decomposition.cu
├── kernel_eigen_decomposition.cu
├── matrix_utils.cuh
├── main.cu
├── README.md
```

- **kernel_addition.cu**: Implements matrix addition using CUDA.
- **kernel_multiplication.cu**: Implements matrix multiplication using CUDA.
- **kernel_fft.cu**: Implements the Fast Fourier Transform (FFT) using CUDA.
- **kernel_lu_decomposition.cu**: Implements LU decomposition using cuSolver.
- **kernel_qr_decomposition.cu**: Implements QR decomposition using cuSolver.
- **kernel_eigen_decomposition.cu**: Implements eigenvalue and eigenvector decomposition using cuSolver.
- **matrix_utils.cuh**: Utility functions for matrix operations, file I/O, and printing matrices.
- **main.cu**: The main driver program that interacts with the user and calls the relevant matrix operations.

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/matrixAdd.git
   cd matrixAdd
   ```

2. **Ensure you have the necessary libraries**:

   Make sure the CUDA Toolkit is installed, and cuBLAS and cuSolver are linked correctly in your environment.

## Build

### Using nvcc (Command Line)

You can compile the project using `nvcc` from the CUDA Toolkit.

```bash
nvcc -o matrix_operations main.cu kernel_addition.cu kernel_multiplication.cu kernel_fft.cu kernel_lu_decomposition.cu kernel_qr_decomposition.cu kernel_eigen_decomposition.cu -lcufft -lcublas -lcusolver
```

### Using Visual Studio (Windows)

1. Open the project in Visual Studio.
2. Add all `.cu` files to the project.
3. Set up CUDA compilation in the project properties.
4. Build and run the project.

## Usage

### Running the Program

After building the project, you can run the executable.

```bash
./matrix_operations
```

### File Input

When you run the program, it will ask if you have a matrix data file. If you provide a `.txt` file, the matrix data will be loaded from the file. If not, you can manually input the matrix data through the command line.

#### File Format

The file should be formatted as follows:

```
rows cols
val11 val12 ... val1n
val21 val22 ... val2n
...
valm1 valm2 ... valmn
```

- `rows`: Number of rows in the matrix.
- `cols`: Number of columns in the matrix.
- `valij`: The value at the ith row and jth column of the matrix.

#### Example File

```txt
3 3
1 2 3
4 5 6
7 8 9
```

### Example Matrix Operations

1. **Matrix Addition**:
   - Select `1` for Matrix Addition.
   - Input matrix dimensions and elements (or load from a file).

2. **Matrix Multiplication**:
   - Select `2` for Matrix Multiplication.
   - Input matrix dimensions and elements (or load from a file).

3. **FFT (Fast Fourier Transform)**:
   - Select `3` for FFT.
   - Input matrix data.

4. **Matrix Transpose**:
   - Select `4` for Matrix Transpose.
   - Input matrix dimensions and elements (or load from a file).

5. **Matrix Inversion**:
   - Select `5` for Matrix Inversion.
   - Input a square matrix.

6. **LU Decomposition**:
   - Select `6` for LU Decomposition.
   - Input a square matrix.

7. **QR Decomposition**:
   - Select `7` for QR Decomposition.
   - Input matrix dimensions and elements (or load from a file).

8. **Eigenvalue Decomposition**:
   - Select `8` for Eigenvalue Decomposition.
   - Input a square matrix.

### Saving Results

After performing an operation, you will be asked if you want to save the result to a file. If you choose "yes", you can specify a filename (e.g., `result.txt`), and the result will be saved in the specified file.

## Example Output

Here is an example output for a 3x3 matrix addition:

```
Do you have a matrix data file? (y/n): n
Select operation: 1 for Matrix Addition, 2 for Matrix Multiplication, 3 for FFT, 4 for Matrix Transpose, 5 for Matrix Inverse, 6 for LU Decomposition, 7 for QR Decomposition, 8 for Eigenvalue Decomposition: 1
Enter the number of rows: 3
Enter the number of columns: 3
Enter elements of matrix A:
1 2 3
4 5 6
7 8 9
Enter elements of matrix B:
9 8 7
6 5 4
3 2 1
Matrix C (Result of A + B):
10.0000 10.0000 10.0000 
10.0000 10.0000 10.0000 
10.0000 10.0000 10.0000
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
