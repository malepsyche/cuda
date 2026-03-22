#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err__) << std::endl;      \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

#define CUSPARSE_CHECK(call)                                                 \
    do {                                                                     \
        cusparseStatus_t st__ = (call);                                      \
        if (st__ != CUSPARSE_STATUS_SUCCESS) {                               \
            std::cerr << "cuSPARSE error at " << __FILE__ << ":" << __LINE__   \
                      << " -> status " << static_cast<int>(st__) << std::endl;   \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

namespace fs = std::filesystem;

static __host__ __device__ inline int idx(int i, int j, int Nx) { return i * nx + j; }

struct Config {
    std::string method = "global";          // global | shared | cusparse
    int nx = 101;
    int ny = 101;
    int steps = 400;
    int snapshot_every = 100;
    double c = 1.0;
    double dx = 0.01;
    double dy = 0.01;
    double dt = 0.005;
    int block_x = 16;
    int block_y = 16;
    std::string output_dir = "output";
    bool write_snapshots = true;
    bool quiet = false;
};

struct TimingStats {
    float avg_kernel_ms = 0.0f;
    float total_kernel_ms = 0.0f;
    float total_sim_ms = 0.0f;
    float avg_library_ms = 0.0f;
    float total_library_ms = 0.0f;
};

struct RunSummary {
    Config cfg;
    TimingStats timing;
    double effective_bandwidth_gbps = 0.0;
    size_t interior_points = 0;
    std::string snapshot_manifest;
};

void print_usage() {
    std::cout
        << "Usage: ./wave2d_solver [options]\n"
        << "  --method global|shared|cusparse\n"
        << "  --nx N --ny N\n"
        << "  --steps N\n"
        << "  --snapshot-every N\n"
        << "  --dx X --dy Y --dt T --c C\n"
        << "  --block-x BX --block-y BY\n"
        << "  --output-dir DIR\n"
        << "  --no-snapshots\n"
        << "  --quiet\n";
}

Config parse_args(int argc, char** argv) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto need_value = [&](const std::string& flag) {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + flag);
            }
        };
        if (arg == "--method") {
            need_value(arg);
            cfg.method = argv[++i];
        } else if (arg == "--nx") {
            need_value(arg);
            cfg.nx = std::stoi(argv[++i]);
        } else if (arg == "--ny") {
            need_value(arg);
            cfg.ny = std::stoi(argv[++i]);
        } else if (arg == "--steps") {
            need_value(arg);
            cfg.steps = std::stoi(argv[++i]);
        } else if (arg == "--snapshot-every") {
            need_value(arg);
            cfg.snapshot_every = std::stoi(argv[++i]);
        } else if (arg == "--dx") {
            need_value(arg);
            cfg.dx = std::stod(argv[++i]);
        } else if (arg == "--dy") {
            need_value(arg);
            cfg.dy = std::stod(argv[++i]);
        } else if (arg == "--dt") {
            need_value(arg);
            cfg.dt = std::stod(argv[++i]);
        } else if (arg == "--c") {
            need_value(arg);
            cfg.c = std::stod(argv[++i]);
        } else if (arg == "--block-x") {
            need_value(arg);
            cfg.block_x = std::stoi(argv[++i]);
        } else if (arg == "--block-y") {
            need_value(arg);
            cfg.block_y = std::stoi(argv[++i]);
        } else if (arg == "--output-dir") {
            need_value(arg);
            cfg.output_dir = argv[++i];
        } else if (arg == "--no-snapshots") {
            cfg.write_snapshots = false;
        } else if (arg == "--quiet") {
            cfg.quiet = true;
        } else if (arg == "--help" || arg == "-h") {
            print_usage();
            std::exit(EXIT_SUCCESS);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }
    if (cfg.nx < 3 || cfg.ny < 3) {
        throw std::runtime_error("nx and ny must be >= 3");
    }
    if (cfg.snapshot_every <= 0) cfg.snapshot_every = cfg.steps;
    return cfg;
}

void ensure_dir(const fs::path& p) {
    if (!fs::exists(p)) fs::create_directories(p);
}

void initialize_u0(std::vector<double>& u0, int nx, int ny, double dx, double dy) {
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            double x = j * dx;
            double y = i * dy;
            bool boundary = (i == 0 || j == 0 || i == ny - 1 || j == nx - 1);
            u0[idx(i, j, nx)] = boundary ? 0.0 : std::sin(M_PI * x) * std::sin(M_PI * y);
        }
    }
}

void initialize_u1(std::vector<double>& u1,
                   const std::vector<double>& u0,
                   int nx,
                   int ny,
                   double lambda2) {
    std::fill(u1.begin(), u1.end(), 0.0);
    for (int i = 1; i < ny - 1; ++i) {
        for (int j = 1; j < nx - 1; ++j) {
            int id = idx(i, j, nx);
            double lap = u0[idx(i + 1, j, nx)] + u0[idx(i - 1, j, nx)] +
                         u0[idx(i, j + 1, nx)] + u0[idx(i, j - 1, nx)] -
                         4.0 * u0[id];
            // zero initial velocity: u^1 = u^0 + 0.5 * lambda^2 * Lap(u^0)
            u1[id] = u0[id] + 0.5 * lambda2 * lap;
        }
    }
}

void write_snapshot_csv(const fs::path& path,
                        const std::vector<double>& field,
                        int nx,
                        int ny) {
    std::ofstream out(path);
    out << std::setprecision(16);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            out << field[idx(i, j, nx)];
            if (j + 1 < nx) out << ',';
        }
        out << '\n';
    }
}

void write_manifest_header(std::ofstream& manifest) {
    manifest << "step,time,csv_path\n";
}

__global__ void wave2d_global_kernel(const double* __restrict__ u_prev,
                                     const double* __restrict__ u_curr,
                                     double* __restrict__ u_next,
                                     int nx,
                                     int ny,
                                     double lambda2) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ny || j >= nx) return;
    int id = idx(i, j, nx);

    if (i == 0 || j == 0 || i == ny - 1 || j == nx - 1) {
        u_next[id] = 0.0;
        return;
    }

    double center = u_curr[id];
    double lap = u_curr[idx(i + 1, j, nx)] + u_curr[idx(i - 1, j, nx)] +
                 u_curr[idx(i, j + 1, nx)] + u_curr[idx(i, j - 1, nx)] -
                 4.0 * center;
    u_next[id] = 2.0 * center - u_prev[id] + lambda2 * lap;
}

__global__ void wave2d_shared_kernel(const double* __restrict__ u_prev,
                                     const double* __restrict__ u_curr,
                                     double* __restrict__ u_next,
                                     int nx,
                                     int ny,
                                     double lambda2) {
    extern __shared__ double tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int block_w = blockDim.x;
    const int block_h = blockDim.y;
    const int tile_w = block_w + 2;

    const int j = blockIdx.x * block_w + tx;
    const int i = blockIdx.y * block_h + ty;

    const int local_x = tx + 1;
    const int local_y = ty + 1;
    const int tile_idx = local_y * tile_w + local_x;

    double center = 0.0;
    if (i < ny && j < nx) center = u_curr[idx(i, j, nx)];
    tile[tile_idx] = center;

    if (tx == 0) {
        tile[local_y * tile_w + 0] = (j > 0 && i < ny) ? u_curr[idx(i, j - 1, nx)] : 0.0;
    }
    if (tx == block_w - 1) {
        tile[local_y * tile_w + (tile_w - 1)] =
            (j + 1 < nx && i < ny) ? u_curr[idx(i, j + 1, nx)] : 0.0;
    }
    if (ty == 0) {
        tile[0 * tile_w + local_x] = (i > 0 && j < nx) ? u_curr[idx(i - 1, j, nx)] : 0.0;
    }
    if (ty == block_h - 1) {
        tile[(block_h + 1) * tile_w + local_x] =
            (i + 1 < ny && j < nx) ? u_curr[idx(i + 1, j, nx)] : 0.0;
    }

    if (tx == 0 && ty == 0) {
        tile[0] = (i > 0 && j > 0) ? u_curr[idx(i - 1, j - 1, nx)] : 0.0;
    }
    if (tx == block_w - 1 && ty == 0) {
        tile[tile_w - 1] = (i > 0 && j + 1 < nx) ? u_curr[idx(i - 1, j + 1, nx)] : 0.0;
    }
    if (tx == 0 && ty == block_h - 1) {
        tile[(block_h + 1) * tile_w] =
            (i + 1 < ny && j > 0) ? u_curr[idx(i + 1, j - 1, nx)] : 0.0;
    }
    if (tx == block_w - 1 && ty == block_h - 1) {
        tile[(block_h + 1) * tile_w + (tile_w - 1)] =
            (i + 1 < ny && j + 1 < nx) ? u_curr[idx(i + 1, j + 1, nx)] : 0.0;
    }

    __syncthreads();

    if (i >= ny || j >= nx) return;
    int gid = idx(i, j, nx);

    if (i == 0 || j == 0 || i == ny - 1 || j == nx - 1) {
        u_next[gid] = 0.0;
        return;
    }

    double lap = tile[(local_y - 1) * tile_w + local_x] +
                 tile[(local_y + 1) * tile_w + local_x] +
                 tile[local_y * tile_w + (local_x - 1)] +
                 tile[local_y * tile_w + (local_x + 1)] -
                 4.0 * tile[tile_idx];
    u_next[gid] = 2.0 * tile[tile_idx] - u_prev[gid] + lambda2 * lap;
}

__global__ void combine_spmv_update(const double* __restrict__ u_prev,
                                    const double* __restrict__ u_curr,
                                    const double* __restrict__ lap_u,
                                    double* __restrict__ u_next,
                                    int n,
                                    double lambda2) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        u_next[gid] = 2.0 * u_curr[gid] - u_prev[gid] + lambda2 * lap_u[gid];
    }
}

struct CsrLaplacian {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
    std::vector<double> values;
};

CsrLaplacian build_laplacian_csr(int nx, int ny) {
    const int n = nx * ny;
    CsrLaplacian csr;
    csr.rows = n;
    csr.cols = n;
    csr.row_offsets.resize(n + 1, 0);

    std::vector<int> cols;
    std::vector<double> vals;
    cols.reserve(static_cast<size_t>(n) * 5);
    vals.reserve(static_cast<size_t>(n) * 5);

    int nnz = 0;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            int row = idx(i, j, nx);
            csr.row_offsets[row] = nnz;

            if (i == 0 || j == 0 || i == ny - 1 || j == nx - 1) {
                cols.push_back(row);
                vals.push_back(0.0);
                nnz += 1;
                continue;
            }

            cols.push_back(idx(i - 1, j, nx)); vals.push_back(1.0);
            cols.push_back(idx(i, j - 1, nx)); vals.push_back(1.0);
            cols.push_back(row);               vals.push_back(-4.0);
            cols.push_back(idx(i, j + 1, nx)); vals.push_back(1.0);
            cols.push_back(idx(i + 1, j, nx)); vals.push_back(1.0);
            nnz += 5;
        }
    }
    csr.row_offsets[n] = nnz;
    csr.col_indices = std::move(cols);
    csr.values = std::move(vals);
    csr.nnz = nnz;
    return csr;
}

RunSummary run_solver(const Config& cfg) {
    if (std::abs(cfg.dx - cfg.dy) > 1e-12) {
        throw std::runtime_error("This implementation assumes dx == dy for lambda^2.");
    }

    const double lambda = cfg.c * cfg.dt / cfg.dx;
    const double lambda2 = lambda * lambda;
    if (lambda > 1.0 / std::sqrt(2.0) + 1e-12) {
        std::cerr << "Warning: CFL condition may be violated. lambda = " << lambda
                  << ", stability limit ~ " << (1.0 / std::sqrt(2.0)) << std::endl;
    }

    const int nx = cfg.nx;
    const int ny = cfg.ny;
    const int n = nx * ny;

    fs::path out_dir(cfg.output_dir);
    ensure_dir(out_dir);
    ensure_dir(out_dir / "snapshots");

    std::vector<double> h_u0(n, 0.0), h_u1(n, 0.0), h_latest(n, 0.0);
    initialize_u0(h_u0, nx, ny, cfg.dx, cfg.dy);
    initialize_u1(h_u1, h_u0, nx, ny, lambda2);

    double *d_prev = nullptr, *d_curr = nullptr, *d_next = nullptr, *d_lap = nullptr;
    CUDA_CHECK(cudaMalloc(&d_prev, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_curr, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_next, n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_prev, h_u0.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_curr, h_u1.data(), n * sizeof(double), cudaMemcpyHostToDevice));

    // cuSPARSE objects (used only for method == cusparse)
    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t matA = nullptr;
    cusparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    void* dBuffer = nullptr;
    size_t bufferSize = 0;
    int *d_row_offsets = nullptr, *d_col_indices = nullptr;
    double *d_values = nullptr;
    CsrLaplacian csr;

    if (cfg.method == "cusparse") {
        CUDA_CHECK(cudaMalloc(&d_lap, n * sizeof(double)));
        csr = build_laplacian_csr(nx, ny);
        CUDA_CHECK(cudaMalloc(&d_row_offsets, (csr.rows + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_col_indices, csr.nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_values, csr.nnz * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_row_offsets, csr.row_offsets.data(), (csr.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_col_indices, csr.col_indices.data(), csr.nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_values, csr.values.data(), csr.nnz * sizeof(double), cudaMemcpyHostToDevice));

        CUSPARSE_CHECK(cusparseCreate(&handle));
        CUSPARSE_CHECK(cusparseCreateCsr(&matA,
                                         csr.rows,
                                         csr.cols,
                                         csr.nnz,
                                         d_row_offsets,
                                         d_col_indices,
                                         d_values,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_32I,
                                         CUSPARSE_INDEX_BASE_ZERO,
                                         CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecX, n, d_curr, CUDA_R_64F));
        CUSPARSE_CHECK(cusparseCreateDnVec(&vecY, n, d_lap, CUDA_R_64F));

        double alpha = 1.0;
        double beta = 0.0;
        CUSPARSE_CHECK(cusparseSpMV_bufferSize(handle,
                                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                               &alpha,
                                               matA,
                                               vecX,
                                               &beta,
                                               vecY,
                                               CUDA_R_64F,
                                               CUSPARSE_SPMV_ALG_DEFAULT,
                                               &bufferSize));
        CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    }

    cudaEvent_t sim_start, sim_stop, step_start, step_stop;
    CUDA_CHECK(cudaEventCreate(&sim_start));
    CUDA_CHECK(cudaEventCreate(&sim_stop));
    CUDA_CHECK(cudaEventCreate(&step_start));
    CUDA_CHECK(cudaEventCreate(&step_stop));

    std::ofstream manifest(out_dir / "snapshots" / (cfg.method + "_manifest.csv"));
    write_manifest_header(manifest);

    auto dump_snapshot = [&](int step, double time_value, const double* d_ptr) {
        if (!cfg.write_snapshots) return;
        CUDA_CHECK(cudaMemcpy(h_latest.data(), d_ptr, n * sizeof(double), cudaMemcpyDeviceToHost));
        fs::path csv_path = out_dir / "snapshots" /
                            (cfg.method + std::string("_step_") + std::to_string(step) + ".csv");
        write_snapshot_csv(csv_path, h_latest, nx, ny);
        manifest << step << ',' << std::setprecision(16) << time_value << ','
                 << csv_path.filename().string() << '\n';
    };

    // Save initial states for visualization.
    dump_snapshot(0, 0.0, d_prev);
    dump_snapshot(1, cfg.dt, d_curr);

    TimingStats timing;
    float total_kernel_ms = 0.0f;
    float total_library_ms = 0.0f;

    dim3 block(cfg.block_x, cfg.block_y);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    size_t shmem_bytes = static_cast<size_t>(cfg.block_x + 2) * static_cast<size_t>(cfg.block_y + 2) * sizeof(double);

    CUDA_CHECK(cudaEventRecord(sim_start));
    for (int step = 2; step <= cfg.steps; ++step) {
        CUDA_CHECK(cudaEventRecord(step_start));

        if (cfg.method == "global") {
            wave2d_global_kernel<<<grid, block>>>(d_prev, d_curr, d_next, nx, ny, lambda2);
            CUDA_CHECK(cudaGetLastError());
        } else if (cfg.method == "shared") {
            wave2d_shared_kernel<<<grid, block, shmem_bytes>>>(d_prev, d_curr, d_next, nx, ny, lambda2);
            CUDA_CHECK(cudaGetLastError());
        } else if (cfg.method == "cusparse") {
            float lib_ms = 0.0f;
            cudaEvent_t lib_start, lib_stop;
            CUDA_CHECK(cudaEventCreate(&lib_start));
            CUDA_CHECK(cudaEventCreate(&lib_stop));
            CUDA_CHECK(cudaEventRecord(lib_start));

            CUSPARSE_CHECK(cusparseDnVecSetValues(vecX, d_curr));
            CUSPARSE_CHECK(cusparseDnVecSetValues(vecY, d_lap));
            double alpha = 1.0;
            double beta = 0.0;
            CUSPARSE_CHECK(cusparseSpMV(handle,
                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                        &alpha,
                                        matA,
                                        vecX,
                                        &beta,
                                        vecY,
                                        CUDA_R_64F,
                                        CUSPARSE_SPMV_ALG_DEFAULT,
                                        dBuffer));
            CUDA_CHECK(cudaEventRecord(lib_stop));
            CUDA_CHECK(cudaEventSynchronize(lib_stop));
            CUDA_CHECK(cudaEventElapsedTime(&lib_ms, lib_start, lib_stop));
            total_library_ms += lib_ms;
            CUDA_CHECK(cudaEventDestroy(lib_start));
            CUDA_CHECK(cudaEventDestroy(lib_stop));

            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            combine_spmv_update<<<blocks, threads>>>(d_prev, d_curr, d_lap, d_next, n, lambda2);
            CUDA_CHECK(cudaGetLastError());
        } else {
            throw std::runtime_error("Unknown method: " + cfg.method);
        }

        CUDA_CHECK(cudaEventRecord(step_stop));
        CUDA_CHECK(cudaEventSynchronize(step_stop));
        float step_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&step_ms, step_start, step_stop));
        total_kernel_ms += step_ms;

        std::swap(d_prev, d_curr);
        std::swap(d_curr, d_next);

        if (step % cfg.snapshot_every == 0 || step == cfg.steps) {
            dump_snapshot(step, step * cfg.dt, d_curr);
        }
    }
    CUDA_CHECK(cudaEventRecord(sim_stop));
    CUDA_CHECK(cudaEventSynchronize(sim_stop));
    CUDA_CHECK(cudaEventElapsedTime(&timing.total_sim_ms, sim_start, sim_stop));

    timing.total_kernel_ms = total_kernel_ms;
    timing.total_library_ms = total_library_ms;
    int measured_steps = std::max(1, cfg.steps - 1);
    timing.avg_kernel_ms = total_kernel_ms / measured_steps;
    timing.avg_library_ms = total_library_ms / measured_steps;

    const size_t interior = static_cast<size_t>(std::max(0, nx - 2)) * static_cast<size_t>(std::max(0, ny - 2));
    const double bytes_per_step = static_cast<double>(interior) * 48.0; // 5 reads + 1 write in double precision
    double effective_bandwidth_gbps = 0.0;
    if (timing.avg_kernel_ms > 0.0f) {
        effective_bandwidth_gbps = (bytes_per_step / (timing.avg_kernel_ms * 1e-3)) / 1e9;
    }

    if (!cfg.quiet) {
        std::cout << std::fixed << std::setprecision(6)
                  << "method=" << cfg.method
                  << " nx=" << nx
                  << " ny=" << ny
                  << " steps=" << cfg.steps
                  << " lambda=" << lambda
                  << " avg_kernel_ms=" << timing.avg_kernel_ms
                  << " total_sim_ms=" << timing.total_sim_ms
                  << " avg_library_ms=" << timing.avg_library_ms
                  << " eff_bw_gbps=" << effective_bandwidth_gbps
                  << std::endl;
    }

    std::ofstream metrics(out_dir / (cfg.method + std::string("_metrics.csv")));
    metrics << "method,nx,ny,steps,dx,dy,dt,c,block_x,block_y,lambda,avg_kernel_ms,total_kernel_ms,total_sim_ms,avg_library_ms,total_library_ms,interior_points,effective_bandwidth_gbps\n";
    metrics << cfg.method << ',' << nx << ',' << ny << ',' << cfg.steps << ','
            << cfg.dx << ',' << cfg.dy << ',' << cfg.dt << ',' << cfg.c << ','
            << cfg.block_x << ',' << cfg.block_y << ',' << lambda << ','
            << timing.avg_kernel_ms << ',' << timing.total_kernel_ms << ','
            << timing.total_sim_ms << ',' << timing.avg_library_ms << ','
            << timing.total_library_ms << ',' << interior << ','
            << effective_bandwidth_gbps << '\n';

    manifest.close();
    metrics.close();

    CUDA_CHECK(cudaFree(d_prev));
    CUDA_CHECK(cudaFree(d_curr));
    CUDA_CHECK(cudaFree(d_next));
    if (d_lap) CUDA_CHECK(cudaFree(d_lap));
    if (d_row_offsets) CUDA_CHECK(cudaFree(d_row_offsets));
    if (d_col_indices) CUDA_CHECK(cudaFree(d_col_indices));
    if (d_values) CUDA_CHECK(cudaFree(d_values));
    if (dBuffer) CUDA_CHECK(cudaFree(dBuffer));
    if (matA) CUSPARSE_CHECK(cusparseDestroySpMat(matA));
    if (vecX) CUSPARSE_CHECK(cusparseDestroyDnVec(vecX));
    if (vecY) CUSPARSE_CHECK(cusparseDestroyDnVec(vecY));
    if (handle) CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(sim_start));
    CUDA_CHECK(cudaEventDestroy(sim_stop));
    CUDA_CHECK(cudaEventDestroy(step_start));
    CUDA_CHECK(cudaEventDestroy(step_stop));

    RunSummary summary;
    summary.cfg = cfg;
    summary.timing = timing;
    summary.effective_bandwidth_gbps = effective_bandwidth_gbps;
    summary.interior_points = interior;
    summary.snapshot_manifest = (out_dir / "snapshots" / (cfg.method + "_manifest.csv")).string();
    return summary;
}

int main(int argc, char** argv) {
    try {
        Config cfg = parse_args(argc, argv);
        run_solver(cfg);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}
