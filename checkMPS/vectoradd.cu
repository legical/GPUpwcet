#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <random>

#define ARRAY_SIZE (1 << 18)
#define BLOCK_NUM 2
#define THREAD_PER_BLOCK 32

__global__ void vectorAdd(double* a, double* b, double* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int i = tid; i < size; i += BLOCK_NUM * THREAD_PER_BLOCK) {
        for (int j = 0; j < THREAD_PER_BLOCK; j++) {  // 执行多个加法操作
            c[i] = a[i] + b[i];
        }
        if (c[i] > ARRAY_SIZE) {
            printf("%f\n", c[i]);
        }
    }
}

int main() {
    double* hostA, * hostB, * hostC;
    double* devA, * devB, * devC;

    // 分配主机内存
    hostA = new double[ARRAY_SIZE];
    hostB = new double[ARRAY_SIZE];
    hostC = new double[ARRAY_SIZE];

    // 使用随机数生成器初始化输入向量
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.0, 13.0);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        hostA[i] = dis(gen);
        hostB[i] = dis(gen);
    }

    // 分配设备内存
    cudaMalloc((void**)&devA, ARRAY_SIZE * sizeof(double));
    cudaMalloc((void**)&devB, ARRAY_SIZE * sizeof(double));
    cudaMalloc((void**)&devC, ARRAY_SIZE * sizeof(double));

    // 将输入向量从主机内存复制到设备内存
    cudaMemcpy(devA, hostA, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, hostB, ARRAY_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 block(THREAD_PER_BLOCK);
    dim3 grid(BLOCK_NUM);  // 2个block
    vectorAdd<<<grid, block>>>(devA, devB, devC, ARRAY_SIZE);

    // 只拷贝一个结果，节省内存操作时间
    cudaMemcpy(hostC, devC, sizeof(double), cudaMemcpyDeviceToHost);

    // 释放内存
    delete[] hostA;
    delete[] hostB;
    delete[] hostC;
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    return 0;
}
