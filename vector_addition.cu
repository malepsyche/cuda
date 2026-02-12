#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <random>

#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = call;                                          \
    if (err != cudaSuccess) {                                        \
        std::cerr << "CUDA Error: "                                  \
                  << cudaGetErrorString(err)                         \
                  << " at " << __FILE__ << ":" << __LINE__           \
                  << std::endl;                                      \
        exit(err);                                                   \
    }                                                                \
} while(0)

__global__ void add_kernel(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void compute_gflops(int threads) {
    int n = 1 << 30;
    size_t bytes = n * sizeof(float);

    // rng
    std::mt19937 rng(std::random_device{}());  
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);
    for (int i=0; i<n; i++) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    add_kernel<<<blocks, threads>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double gflops = (double)n / ((elapsed_ms / 1000.0) * 1e9);
    
    std::cout << "Blocks: " << blocks 
              << " Threads: " << threads 
              << " GFlop/s: " << gflops 
              << std::endl; 

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    std::vector<int> blockSizes = {32, 64, 128, 256};
    for (int i=0; i<blockSizes.size(); ++i) {
        compute_gflops(blockSizes[i]);
    }

    return 0;
}