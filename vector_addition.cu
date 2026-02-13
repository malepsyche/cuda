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

__global__ void vector_add(float* d_a, float* d_b, float* d_c, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

template<typename F>
double compute_gflops(F kernel, int num_elements) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    kernel();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double gflops = (double)num_elements / ((elapsed_ms / 1000.0) * 1e9); 

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return gflops;
}

int main() {
    int num_elements = 1 << 30;
    size_t bytes = num_elements * sizeof(float);

    // rng
    std::mt19937 rng(std::random_device{}());  
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);

    std::vector<float> h_a(num_elements);
    std::vector<float> h_b(num_elements);
    std::vector<float> h_c(num_elements, 0.0f);
    for (int i=0; i<num_elements; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));
    
    std::vector<int> blockSizes = {32, 64, 128, 256};
    for (int i=0; i<blockSizes.size(); ++i) {
        int threadsPerBlock = blockSizes[i];
        int blocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
        double gflops = compute_gflops([&]() {
            vector_add<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, num_elements);
        }, num_elements);
        // results
        std::cout << "Blocks: " << blocks 
                << " threadsPerBlock: " << threadsPerBlock 
                << " GFlop/s: " << gflops 
                << std::endl;
    }

    // cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}