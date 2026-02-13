#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <utility>
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

// C (M * N) = A (M * K) * B (K * N)
__global__ void matrix_multiply_2d_untiled(float* d_a, float* d_b, float* d_c, int M, int N, int K) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) {
        float sum = 0.0f;
        for (int i=0; i<K; ++i) {
            // naive, untiled, uncontiguous memory access
            sum += d_a[r * K + i] * d_b[i * N + c];
        }      
        d_c[r * N + c] = sum;  
    } 
}

template<typename F>
double compute_gflops(F kernel, int M, int N, int K) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    //warmup
    kernel();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // compute gflops
    CUDA_CHECK(cudaEventRecord(start));
    kernel();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    double gflops = (double)(2.0 * M * N * K) / ((elapsed_ms / 1000.0) * 1e9);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop)); 

    return gflops;
}

int main () {
    int M = 8192;
    int N = 8192;
    int K = 8192;

    // rng
    std::mt19937 rng(std::random_device{}());  
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // host memory
    std::vector<float> h_a(M * K);
    std::vector<float> h_b(K * N);
    std::vector<float> h_c(M * N, 0.0f);
    for (int i=0; i<h_a.size(); ++i) {
        h_a[i] = dist(rng);
    }
    for (int i=0; i<h_b.size(); ++i) {
        h_b[i] = dist(rng);
    }

    // device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
    

    std::vector<std::pair<int, int>> blockSizes = {
        {8,8}, {16,16}, {32,32}
    };
    for (auto& pair : blockSizes) {
        dim3 block(pair.first, pair.second);
        dim3 grid((N + block.x -1)/block.x, (M + block.y -1)/block.y);
        double gflops = compute_gflops([&]() {
            matrix_multiply_2d_untiled<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
        }, M, N, K);

        // results
        std::cout << "Grid size: " << grid.y << " x " << grid.x << std::endl;
        std::cout << "Block size: " << block.y << " x " << block.x << std::endl;
        std::cout << "Matrix size: " << M << " x " << N << std::endl;
        std::cout << "GFLOP/s: " << gflops << std::endl;
    }

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}