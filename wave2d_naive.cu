#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__     \
                      << " -> " << cudaGetErrorString(err) << std::endl;     \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Flatten 2D index
__host__ __device__ inline int idx(int i, int j, int Nx) {
    return i * Nx + j;
}

// Initialize u(x,y,0) = sin(pi x) sin(pi y), boundary = 0
void initialize_u0(std::vector<double>& u, int Nx, int Ny, double dx, double dy) {
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            double x = j * dx;
            double y = i * dy;

            if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
                u[idx(i, j, Nx)] = 0.0;
            } else {
                u[idx(i, j, Nx)] = std::sin(M_PI * x) * std::sin(M_PI * y);
            }
        }
    }
}

// First time step:
// u^1 = u^0 + 0.5 * lambda^2 * Laplacian(u^0)
// since initial velocity du/dt at t=0 is zero
__global__ void first_step_kernel(
    const double* u0,
    double* u1,
    int Nx,
    int Ny,
    double lambda2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= Ny || j >= Nx) return;

    // Dirichlet boundary
    if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
        u1[idx(i, j, Nx)] = 0.0;
        return;
    }

    double center = u0[idx(i, j, Nx)];
    double up     = u0[idx(i - 1, j, Nx)];
    double down   = u0[idx(i + 1, j, Nx)];
    double left   = u0[idx(i, j - 1, Nx)];
    double right  = u0[idx(i, j + 1, Nx)];

    double lap = up + down + left + right - 4.0 * center;
    u1[idx(i, j, Nx)] = center + 0.5 * lambda2 * lap;
}

// Standard recurrence:
// u^{n+1} = 2u^n - u^{n-1} + lambda^2 * Laplacian(u^n)
__global__ void update_kernel(
    const double* u_prev,
    const double* u_curr,
    double* u_next,
    int Nx,
    int Ny,
    double lambda2)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= Ny || j >= Nx) return;

    // Dirichlet boundary
    if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
        u_next[idx(i, j, Nx)] = 0.0;
        return;
    }

    double center = u_curr[idx(i, j, Nx)];
    double up     = u_curr[idx(i - 1, j, Nx)];
    double down   = u_curr[idx(i + 1, j, Nx)];
    double left   = u_curr[idx(i, j - 1, Nx)];
    double right  = u_curr[idx(i, j + 1, Nx)];

    double lap = up + down + left + right - 4.0 * center;

    u_next[idx(i, j, Nx)] = 2.0 * center - u_prev[idx(i, j, Nx)] + lambda2 * lap;
}

// Save a field to CSV for later plotting
void save_csv(const std::string& filename, const std::vector<double>& u, int Nx, int Ny) {
    std::ofstream fout(filename);
    for (int i = 0; i < Ny; ++i) {
        for (int j = 0; j < Nx; ++j) {
            fout << u[idx(i, j, Nx)];
            if (j != Nx - 1) fout << ",";
        }
        fout << "\n";
    }
    fout.close();
}

int main() {
    // ===== Assignment baseline parameters =====
    const double c  = 1.0;
    const double dx = 0.01;
    const double dy = 0.01;
    const double dt = 0.005;

    // Domain [0,1] x [0,1], include endpoints
    const int Nx = static_cast<int>(1.0 / dx) + 1; // 101
    const int Ny = static_cast<int>(1.0 / dy) + 1; // 101

    // Number of time steps
    const int Nt = 1000;

    // Since dx = dy, lambda^2 = (c dt / dx)^2
    const double lambda = c * dt / dx;
    const double lambda2 = lambda * lambda;

    std::cout << "Nx = " << Nx << ", Ny = " << Ny << ", Nt = " << Nt << "\n";
    std::cout << "lambda = " << lambda << ", lambda^2 = " << lambda2 << "\n";

    if (lambda > 1.0 / std::sqrt(2.0)) {
        std::cerr << "Warning: CFL condition may be violated.\n";
    }

    size_t bytes = static_cast<size_t>(Nx) * Ny * sizeof(double);

    // ===== Host arrays =====
    std::vector<double> h_u0(Nx * Ny, 0.0);
    std::vector<double> h_result(Nx * Ny, 0.0);

    initialize_u0(h_u0, Nx, Ny, dx, dy);

    // ===== Device arrays =====
    double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_curr, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, bytes));

    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u0.data(), bytes, cudaMemcpyHostToDevice));

    // 2D thread blocks (good default for A1)
    dim3 blockDim(16, 16);
    dim3 gridDim((Nx + blockDim.x - 1) / blockDim.x,
                 (Ny + blockDim.y - 1) / blockDim.y);

    std::cout << "Block = (" << blockDim.x << ", " << blockDim.y << ")\n";
    std::cout << "Grid  = (" << gridDim.x << ", " << gridDim.y << ")\n";

    // ===== First step =====
    first_step_kernel<<<gridDim, blockDim>>>(d_u_prev, d_u_curr, Nx, Ny, lambda2);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // ===== Timing the main update kernel =====
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int n = 1; n < Nt; ++n) {
        update_kernel<<<gridDim, blockDim>>>(d_u_prev, d_u_curr, d_u_next, Nx, Ny, lambda2);
        CUDA_CHECK(cudaGetLastError());

        // rotate pointers
        double* temp = d_u_prev;
        d_u_prev = d_u_curr;
        d_u_curr = d_u_next;
        d_u_next = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    // ===== Copy final field back =====
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_u_curr, bytes, cudaMemcpyDeviceToHost));

    // ===== Save output =====
    save_csv("wave_final.csv", h_result, Nx, Ny);

    // ===== Performance metrics =====
    // For 5-point stencil in double precision:
    // 5 reads + 1 write = 6 doubles = 48 bytes per grid update
    // Use only interior points for a more faithful estimate
    long long interior_points = static_cast<long long>(Nx - 2) * (Ny - 2);
    long long total_updates = interior_points * (Nt - 1); // timed loop only
    double total_bytes = static_cast<double>(total_updates) * 48.0;

    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = total_bytes / elapsed_sec / 1e9;
    double avg_kernel_ms = elapsed_ms / (Nt - 1);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== A1 Global Memory Results ===\n";
    std::cout << "Total kernel time (main loop): " << elapsed_ms << " ms\n";
    std::cout << "Average kernel time/step:      " << avg_kernel_ms << " ms\n";
    std::cout << "Estimated effective BW:        " << bandwidth_GBps << " GB/s\n";
    std::cout << "Output saved to: wave_final.csv\n";

    // ===== Cleanup =====
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u_curr));
    CUDA_CHECK(cudaFree(d_u_next));

    return 0;
}