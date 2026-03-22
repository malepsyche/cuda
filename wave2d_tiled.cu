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

__host__ __device__ inline int idx(int i, int j, int Nx) {
    return i * Nx + j;
}

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

// Load center + halo of u_curr into shared memory.
// Then compute first step:
// u^1 = u^0 + 0.5 * lambda^2 * Laplacian(u^0)
__global__ void first_step_shared_kernel(
    const double* u0,
    double* u1,
    int Nx,
    int Ny,
    double lambda2)
{
    extern __shared__ double tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tx;
    int i = blockIdx.y * blockDim.y + ty;

    int tile_w = blockDim.x + 2;
    int sj = tx + 1;
    int si = ty + 1;

    // Center
    if (i < Ny && j < Nx) {
        tile[si * tile_w + sj] = u0[idx(i, j, Nx)];
    } else {
        tile[si * tile_w + sj] = 0.0;
    }

    // Left halo
    if (tx == 0) {
        int gj = j - 1;
        if (i < Ny && gj >= 0) {
            tile[si * tile_w + 0] = u0[idx(i, gj, Nx)];
        } else {
            tile[si * tile_w + 0] = 0.0;
        }
    }

    // Right halo
    if (tx == blockDim.x - 1) {
        int gj = j + 1;
        if (i < Ny && gj < Nx) {
            tile[si * tile_w + (blockDim.x + 1)] = u0[idx(i, gj, Nx)];
        } else {
            tile[si * tile_w + (blockDim.x + 1)] = 0.0;
        }
    }

    // Top halo
    if (ty == 0) {
        int gi = i - 1;
        if (gi >= 0 && j < Nx) {
            tile[0 * tile_w + sj] = u0[idx(gi, j, Nx)];
        } else {
            tile[0 * tile_w + sj] = 0.0;
        }
    }

    // Bottom halo
    if (ty == blockDim.y - 1) {
        int gi = i + 1;
        if (gi < Ny && j < Nx) {
            tile[(blockDim.y + 1) * tile_w + sj] = u0[idx(gi, j, Nx)];
        } else {
            tile[(blockDim.y + 1) * tile_w + sj] = 0.0;
        }
    }

    // Top-left corner
    if (tx == 0 && ty == 0) {
        int gi = i - 1;
        int gj = j - 1;
        if (gi >= 0 && gj >= 0) {
            tile[0 * tile_w + 0] = u0[idx(gi, gj, Nx)];
        } else {
            tile[0 * tile_w + 0] = 0.0;
        }
    }

    // Top-right corner
    if (tx == blockDim.x - 1 && ty == 0) {
        int gi = i - 1;
        int gj = j + 1;
        if (gi >= 0 && gj < Nx) {
            tile[0 * tile_w + (blockDim.x + 1)] = u0[idx(gi, gj, Nx)];
        } else {
            tile[0 * tile_w + (blockDim.x + 1)] = 0.0;
        }
    }

    // Bottom-left corner
    if (tx == 0 && ty == blockDim.y - 1) {
        int gi = i + 1;
        int gj = j - 1;
        if (gi < Ny && gj >= 0) {
            tile[(blockDim.y + 1) * tile_w + 0] = u0[idx(gi, gj, Nx)];
        } else {
            tile[(blockDim.y + 1) * tile_w + 0] = 0.0;
        }
    }

    // Bottom-right corner
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int gi = i + 1;
        int gj = j + 1;
        if (gi < Ny && gj < Nx) {
            tile[(blockDim.y + 1) * tile_w + (blockDim.x + 1)] = u0[idx(gi, gj, Nx)];
        } else {
            tile[(blockDim.y + 1) * tile_w + (blockDim.x + 1)] = 0.0;
        }
    }

    __syncthreads();

    if (i >= Ny || j >= Nx) return;

    if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
        u1[idx(i, j, Nx)] = 0.0;
        return;
    }

    double center = tile[si * tile_w + sj];
    double up     = tile[(si - 1) * tile_w + sj];
    double down   = tile[(si + 1) * tile_w + sj];
    double left   = tile[si * tile_w + (sj - 1)];
    double right  = tile[si * tile_w + (sj + 1)];

    double lap = up + down + left + right - 4.0 * center;
    u1[idx(i, j, Nx)] = center + 0.5 * lambda2 * lap;
}

// Standard recurrence using shared memory tile for u_curr:
// u^{n+1} = 2u^n - u^{n-1} + lambda^2 * Laplacian(u^n)
__global__ void update_shared_kernel(
    const double* u_prev,
    const double* u_curr,
    double* u_next,
    int Nx,
    int Ny,
    double lambda2)
{
    extern __shared__ double tile[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int j = blockIdx.x * blockDim.x + tx;
    int i = blockIdx.y * blockDim.y + ty;

    int tile_w = blockDim.x + 2;
    int sj = tx + 1;
    int si = ty + 1;

    // Center
    if (i < Ny && j < Nx) {
        tile[si * tile_w + sj] = u_curr[idx(i, j, Nx)];
    } else {
        tile[si * tile_w + sj] = 0.0;
    }

    // Left halo
    if (tx == 0) {
        int gj = j - 1;
        if (i < Ny && gj >= 0) {
            tile[si * tile_w + 0] = u_curr[idx(i, gj, Nx)];
        } else {
            tile[si * tile_w + 0] = 0.0;
        }
    }

    // Right halo
    if (tx == blockDim.x - 1) {
        int gj = j + 1;
        if (i < Ny && gj < Nx) {
            tile[si * tile_w + (blockDim.x + 1)] = u_curr[idx(i, gj, Nx)];
        } else {
            tile[si * tile_w + (blockDim.x + 1)] = 0.0;
        }
    }

    // Top halo
    if (ty == 0) {
        int gi = i - 1;
        if (gi >= 0 && j < Nx) {
            tile[0 * tile_w + sj] = u_curr[idx(gi, j, Nx)];
        } else {
            tile[0 * tile_w + sj] = 0.0;
        }
    }

    // Bottom halo
    if (ty == blockDim.y - 1) {
        int gi = i + 1;
        if (gi < Ny && j < Nx) {
            tile[(blockDim.y + 1) * tile_w + sj] = u_curr[idx(gi, j, Nx)];
        } else {
            tile[(blockDim.y + 1) * tile_w + sj] = 0.0;
        }
    }

    // Top-left corner
    if (tx == 0 && ty == 0) {
        int gi = i - 1;
        int gj = j - 1;
        if (gi >= 0 && gj >= 0) {
            tile[0 * tile_w + 0] = u_curr[idx(gi, gj, Nx)];
        } else {
            tile[0 * tile_w + 0] = 0.0;
        }
    }

    // Top-right corner
    if (tx == blockDim.x - 1 && ty == 0) {
        int gi = i - 1;
        int gj = j + 1;
        if (gi >= 0 && gj < Nx) {
            tile[0 * tile_w + (blockDim.x + 1)] = u_curr[idx(gi, gj, Nx)];
        } else {
            tile[0 * tile_w + (blockDim.x + 1)] = 0.0;
        }
    }

    // Bottom-left corner
    if (tx == 0 && ty == blockDim.y - 1) {
        int gi = i + 1;
        int gj = j - 1;
        if (gi < Ny && gj >= 0) {
            tile[(blockDim.y + 1) * tile_w + 0] = u_curr[idx(gi, gj, Nx)];
        } else {
            tile[(blockDim.y + 1) * tile_w + 0] = 0.0;
        }
    }

    // Bottom-right corner
    if (tx == blockDim.x - 1 && ty == blockDim.y - 1) {
        int gi = i + 1;
        int gj = j + 1;
        if (gi < Ny && gj < Nx) {
            tile[(blockDim.y + 1) * tile_w + (blockDim.x + 1)] = u_curr[idx(gi, gj, Nx)];
        } else {
            tile[(blockDim.y + 1) * tile_w + (blockDim.x + 1)] = 0.0;
        }
    }

    __syncthreads();

    if (i >= Ny || j >= Nx) return;

    if (i == 0 || i == Ny - 1 || j == 0 || j == Nx - 1) {
        u_next[idx(i, j, Nx)] = 0.0;
        return;
    }

    double center = tile[si * tile_w + sj];
    double up     = tile[(si - 1) * tile_w + sj];
    double down   = tile[(si + 1) * tile_w + sj];
    double left   = tile[si * tile_w + (sj - 1)];
    double right  = tile[si * tile_w + (sj + 1)];

    double lap = up + down + left + right - 4.0 * center;
    u_next[idx(i, j, Nx)] = 2.0 * center - u_prev[idx(i, j, Nx)] + lambda2 * lap;
}

int main() {
    const double c  = 1.0;
    const double dx = 0.01;
    const double dy = 0.01;
    const double dt = 0.005;

    const int Nx = static_cast<int>(1.0 / dx) + 1; // 101
    const int Ny = static_cast<int>(1.0 / dy) + 1; // 101
    const int Nt = 1000;

    const double lambda = c * dt / dx;
    const double lambda2 = lambda * lambda;

    std::cout << "Nx = " << Nx << ", Ny = " << Ny << ", Nt = " << Nt << "\n";
    std::cout << "lambda = " << lambda << ", lambda^2 = " << lambda2 << "\n";

    if (lambda > 1.0 / std::sqrt(2.0)) {
        std::cerr << "Warning: CFL condition may be violated.\n";
    }

    size_t bytes = static_cast<size_t>(Nx) * Ny * sizeof(double);

    std::vector<double> h_u0(Nx * Ny, 0.0);
    std::vector<double> h_result(Nx * Ny, 0.0);

    initialize_u0(h_u0, Nx, Ny, dx, dy);

    double *d_u_prev = nullptr, *d_u_curr = nullptr, *d_u_next = nullptr;
    CUDA_CHECK(cudaMalloc(&d_u_prev, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_curr, bytes));
    CUDA_CHECK(cudaMalloc(&d_u_next, bytes));

    CUDA_CHECK(cudaMemcpy(d_u_prev, h_u0.data(), bytes, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((Nx + blockDim.x - 1) / blockDim.x,
                 (Ny + blockDim.y - 1) / blockDim.y);

    size_t shared_bytes = static_cast<size_t>(blockDim.x + 2) * (blockDim.y + 2) * sizeof(double);

    std::cout << "Block = (" << blockDim.x << ", " << blockDim.y << ")\n";
    std::cout << "Grid  = (" << gridDim.x << ", " << gridDim.y << ")\n";
    std::cout << "Shared memory per block = " << shared_bytes << " bytes\n";

    first_step_shared_kernel<<<gridDim, blockDim, shared_bytes>>>(
        d_u_prev, d_u_curr, Nx, Ny, lambda2
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int n = 1; n < Nt; ++n) {
        update_shared_kernel<<<gridDim, blockDim, shared_bytes>>>(
            d_u_prev, d_u_curr, d_u_next, Nx, Ny, lambda2
        );
        CUDA_CHECK(cudaGetLastError());

        double* temp = d_u_prev;
        d_u_prev = d_u_curr;
        d_u_curr = d_u_next;
        d_u_next = temp;
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaMemcpy(h_result.data(), d_u_curr, bytes, cudaMemcpyDeviceToHost));

    save_csv("wave_final_a2.csv", h_result, Nx, Ny);

    long long interior_points = static_cast<long long>(Nx - 2) * (Ny - 2);
    long long total_updates = interior_points * (Nt - 1);
    double total_bytes = static_cast<double>(total_updates) * 48.0;

    double elapsed_sec = elapsed_ms / 1000.0;
    double bandwidth_GBps = total_bytes / elapsed_sec / 1e9;
    double avg_kernel_ms = elapsed_ms / (Nt - 1);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== A2 Shared Memory Results ===\n";
    std::cout << "Total kernel time (main loop): " << elapsed_ms << " ms\n";
    std::cout << "Average kernel time/step:      " << avg_kernel_ms << " ms\n";
    std::cout << "Estimated effective BW:        " << bandwidth_GBps << " GB/s\n";
    std::cout << "Output saved to: wave_final_a2.csv\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_u_prev));
    CUDA_CHECK(cudaFree(d_u_curr));
    CUDA_CHECK(cudaFree(d_u_next));

    return 0;
}