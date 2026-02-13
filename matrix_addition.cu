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

__global__ void matrix_add_1d(float* d_a, float* d_b, float* d_c, int num_elements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_elements) {
        d_c[i] = d_a[i] + d_b[i];
    }
}

__global__ void matrix_add_2d(float* d_a, float* d_b, float* d_c, int matrix_width, int matrix_height) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < matrix_height && c < matrix_width) {
        int i = r * matrix_width + c;
        d_c[i] = d_a[i] + d_b[i];
    }
}

template<typename F>
double compute_gflops(F kernel, int num_elements) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warmup
    kernel();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

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
    int matrix_height = 8192;
    int matrix_width = 8192;
    int num_elements = matrix_height * matrix_width;
    size_t bytes = (num_elements) * sizeof(float);

    // rng
    std::mt19937 rng(std::random_device{}());  
    std::uniform_real_distribution<float> dist(0.0f, 100.0f);
    
    // host memory
    std::vector<float> h_a(num_elements);
    std::vector<float> h_b(num_elements);
    std::vector<float> h_c(num_elements);
    for (int i=0; i<num_elements; ++i) {
        h_a[i] = dist(rng);
        h_b[i] = dist(rng);
    }

    // device memory
    float* d_a, float* d_b, float* d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));
    
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice));

    // calculate gflops for 1d grid/block configuration
    int threadsPerBlock = 256;
    int numBlocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    double matrix_add_1d_gflops = compute_gflops([&]() {
        matrix_add_1d<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, num_elements);
    }, num_elements);
    
    // calculate gflops for 2d grid/block configuration
    dim3 block(16, 16);
    dim3 grid((matrix_width + block.x - 1) / block.x, 
              (matrix_height + block.y - 1) / block.y);
    double matrix_add_2d_gflops = compute_gflops([&]() {
        matrix_add_2d<<<grid, block>>>(d_a, d_b, d_c, matrix_width, matrix_height);
    }, num_elements);

    // results
    std::cout << "Matrix size: " << matrix_height << " x " << matrix_width << std::endl;
    std::cout << "1D kernel GFLOP/s: " << matrix_add_1d_gflops << std::endl;
    std::cout << "2D kernel GFLOP/s: " << matrix_add_2d_gflops << std::endl;

    // cleanup
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return 0;
}