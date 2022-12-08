#include <cuda.h>
#include <iostream>

__global__ void addVectors(float* a, float* b, float* c, int N)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < N )
    {
        c[index] = a[index] + b[index];
    }
}

int main()
{
    cudaError_t error = cudaSuccess;
    int N = 50000;
    size_t sizeN = N * sizeof(float);
    
    float* h_a = (float*)malloc(sizeN);
    float* h_b = (float*)malloc(sizeN);
    float* h_c = (float*)malloc(sizeN);

    for(int i=0; i<N; i++)
    {
        h_a[i] = rand() / (float)RAND_MAX;
        h_b[i] = rand() / (float)RAND_MAX;
    }

    float* d_a=NULL, *d_b=NULL, *d_c=NULL;

    error = cudaMalloc(&d_a, sizeN); 
    if(error != cudaSuccess)
    {
        std::cerr << "failed to allocate gpu memory to device A :" << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_b, sizeN); 
    if(error != cudaSuccess)
    {
        std::cerr << "failed to allocate gpu memory to device B :" << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMalloc(&d_c, sizeN); 
    if(error != cudaSuccess)
    {
        std::cerr << "failed to allocate gpu memory to device C :" << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_a, h_a, sizeN, cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        std::cerr << " failed to copy memory from host_A to device_A : " cudaGetErrorString(error) <<std::endl;
        exit(EXIT_FAILURE);
    }

    error = cudaMemcpy(d_b, h_b, sizeN, cudaMemcpyHostToDevice);
    if(error != cudaSuccess)
    {
        std::cerr << " failed to copy memory from host_B to device_B : " cudaGetErrorString(error) <<std::endl;
        exit(EXIT_FAILURE);
    }

    int blockDimValue = 256;
    int gridDimValue = (N + blockDimValue - 1)/blockDimValue;

    addVectors <<< gridDimValue, blockDimValue >>> (d_a, d_b, d_c, N);


    error = cudaMemcpy(h_c, d_c, sizeN, cudaMemcpyDeviceToHost);
    if(error != cudaSuccess)
    {
        std::cerr << " failed to copy memory from device_C to host_C : " cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaFree(d_a);
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to free device memory of device A : " cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaFree(d_b);
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to free device memory of device B : " cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    cudaFree(d_c);
    if(error != cudaSuccess)
    {
        std::cerr << "Failed to free device memory of device C : " cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
