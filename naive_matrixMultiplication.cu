#include <iostream>
#include <cuda.h>

struct Matrix
{
    int rows, cols;
    float* data;
};

__global__ void naive_matMul(const Matrix a, const Matrix b, Matrix c)
{
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    if(col<c.cols && row<c.rows)
    {
        float cValue = 0;
        for(int i=0; i<a.cols; i++)
            cValue += a.data[row * a.cols + i] * b.data[i * b.cols + col];

        c.data[row * c.cols + col] = cValue;
    }
    
}

int main()
{
    int N = 1024;

    Matrix a, b, c, dev_a, dev_b, dev_c;

    a.rows = dev_a.rows = N;
    a.cols = dev_a.cols = N;
    a.data = (float*)malloc( a.rows * a.cols * sizeof(float));
    cudaMalloc(&dev_a.data, a.rows * a.cols * sizeof(float));
    for(int i=0; i < a.rows * a.cols; i++)
        a.data[i] = rand() / (float)RAND_MAX;
    cudaMemcpy(dev_a.data, a.data, a.rows * a.cols * sizeof(float), cudaMemcpyHostToDevice);

    b.rows = dev_b.rows = N;
    b.cols = dev_b.cols = N;
    b.data = (float*)malloc( b.rows * b.cols * sizeof(float));
    cudaMalloc(&dev_b.data, b.rows * b.cols * sizeof(float));
    for(int i=0; i < b.rows * b.cols; i++)
        b.data[i] = rand() / (float)RAND_MAX;
    cudaMemcpy(dev_b.data, b.data, b.rows * b.cols * sizeof(float), cudaMemcpyHostToDevice);


    c.rows = dev_c.rows = N;
    c.cols = dev_c.cols = N;
    c.data = (float*)malloc( c.rows * c.cols * sizeof(float));
    cudaMalloc(&dev_c.data, c.rows * c.cols * sizeof(float));

    dim3 threads_(32,32);
    dim3 blocks(32,32);
    
    naive_matMul <<< blocks, threads_ >>> (dev_a, dev_b, dev_c);

    cudaMemcpy(c.data, dev_c.data, c.rows * c.cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a.data);
    cudaFree(dev_b.data);
    cudaFree(dev_c.data);

    free(a.data);
    free(b.data);
    free(c.data);

    return 0;

}
