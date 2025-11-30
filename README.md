Project 0 Getting Started
====================

**University of Pennsylvania, CIS 5650: GPU Programming and Architecture, Project 0**

* Summerfruity
  
* Tested on: Windows 11, GTX 3060 laptop

---

## Project Description

This project serves as an introduction to CUDA programming. It implements three basic parallel algorithms to demonstrate the interaction between the host (CPU) and the device (GPU), memory management, and thread scheduling.

The three main components implemented are:

1.  **SAXPY**: Single-precision A * X + Y vector arithmetic.
2.  **Matrix Transpose**: Naive matrix transposition using 2D indexing.
3.  **Matrix Multiplication**: Naive matrix multiplication ($P = M \times N$).

## Implementation Details

### 1. SAXPY
* **Goal**: Compute $Z = A \cdot X + Y$ for vectors $X$ and $Y$ and scalar $A$.
* **Implementation**:
    * Launched a 1D grid of threads.
    * Each thread computes one element of the result vector.
    * Used `index = blockIdx.x * blockDim.x + threadIdx.x` to map threads to data elements.
    * Handled boundary conditions where the number of threads exceeds the vector size.

### 2. Matrix Transpose
* **Goal**: Copy matrix $A$ to matrix $B$ such that $B_{ji} = A_{ij}$.
* **Implementation**:
    * Launched a 2D grid of threads to map the 2D matrix structure.
    * Calculated 1D memory indices from 2D thread coordinates $(x, y)$ using row-major mapping: `index = y * width + x`.
    * Implemented a copy kernel (identity) and a transpose kernel (swapping coordinates for the write operation).

### 3. Matrix Multiplication (Naive)
* **Goal**: Compute $P = M \times N$.
* **Implementation**:
    * Used a 2D grid configuration where each thread is responsible for computing one element of the result matrix $P$.
    * Each thread performs a dot product between a row of $M$ and a column of $N$.
    * **Memory Access**: This is a "Naive" implementation because it relies entirely on Global Memory. [cite_start]For every element calculated, the thread reads an entire row from $M$ and an entire column from $N$ from global memory, which is bandwidth-intensive[cite: 936].
    * *(Note: This sets the stage for future optimizations using Shared Memory tiling to reduce global memory bandwidth pressure).*




## Building and Running

1.  Clone the repository.
2.  Create a build directory: `mkdir build` and `cd build`.
3.  Generate project files using CMake:
    * Windows: `cmake .. -G "Visual Studio 17 2022"`
4.  Open `CUDAIntroduction.sln` in Visual Studio.
5.  Set `CUDAIntroduction` as the StartUp Project.
6.  Build and Run (Debug or Release mode).

---

