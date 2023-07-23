#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>

#define BLOCK_SIZE 32
#define VECTOR_SIZE (1 << 26)
#define OPERATIONS_PER_THREAD 1000

__global__ void vectorAdd(double* a, double* b, double* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        for (int j = 0; j < OPERATIONS_PER_THREAD; j++) {  // 执行多个加法操作
            c[i] = a[i] + b[i];
        }
        if (c[i] > VECTOR_SIZE) {
            printf("%f\n", c[i]);
        }
    }
}

int main() {
    double* hostA, * hostB, * hostC;
    double* devA, * devB, * devC;

    // 分配主机内存
    hostA = new double[VECTOR_SIZE];
    hostB = new double[VECTOR_SIZE];
    hostC = new double[VECTOR_SIZE];

    // 使用随机数生成器初始化输入向量
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 13.0);
    for (int i = 0; i < VECTOR_SIZE; i++) {
        hostA[i] = dis(gen);
        hostB[i] = dis(gen);
    }

    // 分配设备内存
    cudaMalloc((void**)&devA, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&devB, VECTOR_SIZE * sizeof(double));
    cudaMalloc((void**)&devC, VECTOR_SIZE * sizeof(double));

    // 将输入向量从主机内存复制到设备内存
    cudaMemcpy(devA, hostA, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, VECTOR_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 block(BLOCK_SIZE);
    dim3 grid((VECTOR_SIZE + block.x - 1) / block.x / 2);  // 2个block
    vectorAdd<<<grid, block>>>(devA, devB, devC, VECTOR_SIZE);

    // 将结果从设备内存复制到主机内存
    cudaMemcpy(hostC, devC, VECTOR_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    // 释放内存
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}
