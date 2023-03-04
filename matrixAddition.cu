#include <cuda.h>
#include <iostream>

struct Matrix {
    int height, width;
    float* elements; 
};

__global__ void addMatrix(const Matrix a, const Matrix b, Matrix c)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    if(col < c.width && row < c.height)
        c.elements[row * c.width + col] = a.elements[row * a.width + col] + b.elements[row * b.width + col];
}

int main()
{
    cudaError_t error = cudaSuccess;
    int N = 1024;

    Matrix h_a, h_b, h_c;

    h_a.height = h_a.width = N;
    h_b.height = h_b.width = N;
    h_c.height = h_b.width = N;

    h_a.elements = (float*)malloc(h_a.height * h_a.width * sizeof(float));
    h_b.elements = (float*)malloc(h_b.height * h_b.width * sizeof(float));
    h_c.elements = (float*)malloc(h_c.height * h_c.widht * sizeof(float));

    for(int i=0; i< N*N; ++i)
    {
        h_a.elements[i] = rand() / (float)RAND_MAX;
        h_b.elements[i] = rand() / (float)RAND_MAX;
    }
    
    Matrix d_a, d_b, d_c;

    d_a.width = h_a.width;
    d_a.height = h_a.height;
    error = cudaMalloc(&d_a.elements, h_a.height * h_a.width * sizeof(float));
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to allocate gpu memory to device_A : " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    d_b.width = h_b.width;
    d_b.height = h_b.height;
    error = cudaMalloc(&d_b.elements, h_b.height * h_b.width * sizeof(float));
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to allocate gpu memory to device_B : " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    d_c.width = h_c.width;
    d_c.height = h_c.height;
    error = cudaMalloc(&d_c.elements, h_c.height * h_c.width * sizeof(float));
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to allocate gpu memory to device_C : " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_a.elements, h_a.elements, h_a.height * h_a.width * sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to copy memory from host_A to device_A : " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }


    error = cudaMemcpy(d_b.elements, h_b.elements, h_b.height * h_b.width * sizeof(float), cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to copy memory from host_A to device_A : " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    dim3 dimBlock(32,32);
    dim3 dimGrid(32,32);    

    addMatrix <<< dimGrid, dimBlock >>> (d_a, d_b, d_c);
    cudaMemcpy(h_c.elements, d_c.elements, h_c.height * h_c.width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a.elements);
    cudaFree(d_b.elements);
    cudaFree(d_c.elements);

    free(h_a.elements);
    free(h_b.elements);
    free(h_c.elements);

    return 0;
}
