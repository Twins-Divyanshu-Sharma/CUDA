#include <iostream>
#include <cuda.h>

#define TILESIZE 8

struct Matrix
{
    int rows, cols;
    float* data;
};

__global__ void tiled_matMul(const Matrix a, const Matrix b, Matrix c)
{

    __shared__ float sa[TILESIZE][TILESIZE];
    __shared__ float sb[TILESIZE][TILESIZE];

    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    
    float val= 0;
    sa[threadIdx.y][threadIdx.x] = 0;
    sb[threadIdx.y][threadIdx.x] = 0;

    for( int k=0; k<(a.cols - 1)/TILESIZE +1; k++)
    {
        if(row < a.rows && (threadIdx.x + k*TILESIZE) < a.cols)
            sa[threadIdx.y][threadIdx.x] = a.data[(row*a.cols) + threadIdx.x + k*TILESIZE];
        else
            sa[threadIdx.y][threadIdx.x] = 0;

        if(col < b.cols && (threadIdx.y + k*TILESIZE) < b.rows)
            sb[threadIdx.y][threadIdx.x] = b.data[(threadIdx.y + k*TILESIZE)*b.cols + col];
        else
            sb[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        for(int j=0; j<TILESIZE; j++)
            val += sa[threadIdx.y][j] * sb[j][threadIdx.x];

    }

    if(row < c.rows && col < c.cols)
        c.data[row * c.cols + col] = val;
   
}

int main()
{
    int N = 64;

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

    dim3 threads_(8,8);
    dim3 blocks(8,8);
    
    tiled_matMul <<< blocks, threads_ >>> (dev_a, dev_b, dev_c);

    cudaMemcpy(c.data, dev_c.data, c.rows * c.cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_a.data);
    cudaFree(dev_b.data);
    cudaFree(dev_c.data);

    free(a.data);
    free(b.data);
    free(c.data);

    return 0;

}
