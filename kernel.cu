#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>



#define CHECK(call) \
{ \
 const cudaError_t error = call; \
 if (error != cudaSuccess) \
 { \
 printf("Error: %s:%d, ", __FILE__, __LINE__); \
 printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
 exit(1); \
 } \
}

void initialData(float* ip, int size) {
    float val = 0;
    for (int i = 0; i < size; i++) {
        float random_number = rand() % 101;
        ip[i] = i;
    }
}

float* linspace(float start, float stop, int size) {
    float* arr = new float[size];
    float arr_increment = (stop - start) / size;
    for (int i = 0; i < size; i++) {
        arr[i] = start + i * arr_increment;
    };
    return arr;
};

void initialSolutions(float* ip,
    float start,
    float stop,
    int nx,
    int ny) {
    int id = 0;
    float* solutionsArr = linspace(start, stop, nx);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            ip[id] = solutionsArr[j];
            id++;
        }
    }
    delete[] solutionsArr;
}


void printMatrix(float* C, const int nx, const int ny) {
    float* ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            printf("%0f", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

void writeCSV(std::string filename, float* C, const int nx, const int ny) {
    std::ofstream file(filename);
    float* ic = C;
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
    }
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            file << ic[ix];
            file << ",";
        }
        ic += nx;
        file << "\n";
    }
}

__global__ void funcMatrix(float* matA, float* matB, float well_width, int nx, int ny) {
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        float n = (iy + 1);
        float psi = sqrt(2 / well_width) * sin((n * 3.14 / well_width) * matA[idx]);
        matB[idx] = psi;
    }
}

int main(int argc, char** argv)
{
    printf("Starting application %s...", argv[0]);

    //Get device properties
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //Set up well size
    float well_width = 10;
    float well_start = -well_width / 2;
    float well_stop = well_width / 2;

    //Set up matrix
    int nx = 100;
    int ny = 20;

    //Total size
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    //Initialize unified memory matrices
    float* A, * gpuRef;
    printf("The size of your matrix is % d by % d, with an area of % d.\n", nx, ny, nxy);

    //Allocate memory on host
    cudaMallocManaged((float**)&A, nBytes);
    cudaMallocManaged((float**)&gpuRef, nBytes);

    //Initialize data on matrices on host side
    initialSolutions(A, well_start, well_stop, nx, ny);

    //Print initial matrix data
    printMatrix(A, nx, ny);

    //memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    //Invoke kernel at host side
    int dimx = 2;
    int dimy = 2;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    //Execute matrix function on GPU
    funcMatrix << < grid, block >> > (A, gpuRef, well_width, nx, ny);
    cudaDeviceSynchronize();

    //Print the results of the GPU computation
    printMatrix(gpuRef, nx, ny);

    //Write out matrix as csv file
    std::string filepath = "C:/Users/elibr/OneDrive/Desktop/simulations/CUDA/data3.csv";
    writeCSV(filepath, gpuRef, nx, ny);

    //Free unified memory
    cudaFree(A);
    cudaFree(gpuRef);

    //Reset GPU
    cudaDeviceReset();

    return 0;
}


