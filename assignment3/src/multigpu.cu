#include "multigpu.cuh"
#include "pipeline.cuh"
#include "pgm_io.cuh"
#include "cuda_check.cuh"

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

#include <nvtx3/nvToolsExt.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>

// Per-image streams let H2D blur Sobel and D2H overlap across different images.
// After histogramKernel on streams[i] we sync that stream then run thrust exclusive_scan on the host.
// Section 2.5 allows a CPU prefix sum. Small 256 element scan stays cheap versus device contention.
static void process_batch_on_device(
    std::vector<ImageEntry>& sub_batch,
    int device_id,
    int stop_after_stage)
{
    CUDA_CHECK(cudaSetDevice(device_id));

    if (sub_batch.empty())
        return;

    const int W = sub_batch[0].width;
    const int H = sub_batch[0].height;
    const size_t img_bytes = (size_t)W * H * sizeof(uint8_t);
    const size_t hist_bytes = 256 * sizeof(unsigned int);
    const size_t cdf_bytes = 256 * sizeof(float);
    const int n_images = static_cast<int>(sub_batch.size());

    std::vector<uint8_t*> d_in(n_images, nullptr);
    std::vector<uint8_t*> d_blur(n_images, nullptr);
    std::vector<uint8_t*> d_edges(n_images, nullptr);
    std::vector<uint8_t*> d_out(n_images, nullptr);
    std::vector<unsigned int*> d_hist(n_images, nullptr);
    std::vector<float*> d_cdf(n_images, nullptr);

    for (int i = 0; i < n_images; i++) {
        CUDA_CHECK(cudaMalloc(&d_in[i], img_bytes));
        CUDA_CHECK(cudaMalloc(&d_blur[i], img_bytes));
        CUDA_CHECK(cudaMalloc(&d_edges[i], img_bytes));
        CUDA_CHECK(cudaMalloc(&d_out[i], img_bytes));
        CUDA_CHECK(cudaMalloc(&d_hist[i], hist_bytes));
        CUDA_CHECK(cudaMalloc(&d_cdf[i], cdf_bytes));
    }

    nvtxRangePushA("A3_GPU_batch");

    {
        std::vector<cudaStream_t> streams(static_cast<size_t>(n_images));
        for (int i = 0; i < n_images; i++)
            CUDA_CHECK(cudaStreamCreate(&streams[i]));

        const dim3 block(TILE_W, TILE_H);
        const dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H);
        const dim3 b16(16, 16);
        const dim3 g16((W + 15) / 16, (H + 15) / 16);

        for (int i = 0; i < n_images; i++) {
            CUDA_CHECK(cudaMemcpyAsync(
                d_in[i], sub_batch[i].host_in, img_bytes,
                cudaMemcpyHostToDevice, streams[i]));

            gaussianBlurKernel<<<grid, block, 0, streams[i]>>>(d_in[i], d_blur[i], W, H);
            CUDA_KERNEL_CHECK();

            if (stop_after_stage <= 1) {
                CUDA_CHECK(cudaMemcpyAsync(
                    sub_batch[i].host_out, d_blur[i], img_bytes,
                    cudaMemcpyDeviceToHost, streams[i]));
                continue;
            }

            sobelKernel<<<grid, block, 0, streams[i]>>>(d_blur[i], d_edges[i], W, H);
            CUDA_KERNEL_CHECK();

            if (stop_after_stage <= 2) {
                CUDA_CHECK(cudaMemcpyAsync(
                    sub_batch[i].host_out, d_edges[i], img_bytes,
                    cudaMemcpyDeviceToHost, streams[i]));
                continue;
            }

            CUDA_CHECK(cudaMemsetAsync(d_hist[i], 0, hist_bytes, streams[i]));
            histogramKernel<<<g16, b16, 0, streams[i]>>>(d_edges[i], d_hist[i], W, H);
            CUDA_KERNEL_CHECK();

            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

            unsigned int h_hist[256];
            CUDA_CHECK(cudaMemcpy(h_hist, d_hist[i], hist_bytes, cudaMemcpyDeviceToHost));

            thrust::host_vector<unsigned int> hv(256);
            for (int k = 0; k < 256; k++)
                hv[k] = h_hist[k];
            thrust::host_vector<unsigned int> excl(256);
            thrust::exclusive_scan(hv.begin(), hv.end(), excl.begin());

            float h_cdf[256];
            for (int k = 0; k < 256; k++)
                h_cdf[k] = static_cast<float>(excl[k]);

            float cdf_min = 0.f;
            for (int b = 0; b < 256; b++) {
                if (h_cdf[b] > 0.f) {
                    cdf_min = h_cdf[b];
                    break;
                }
            }

            // Host to device on the same stream as equalizeKernel avoids racing the default stream.
            CUDA_CHECK(cudaMemcpyAsync(
                d_cdf[i], h_cdf, cdf_bytes, cudaMemcpyHostToDevice, streams[i]));

            equalizeKernel<<<g16, b16, 0, streams[i]>>>(
                d_edges[i], d_out[i], d_cdf[i], cdf_min, W, H);
            CUDA_KERNEL_CHECK();

            CUDA_CHECK(cudaMemcpyAsync(
                sub_batch[i].host_out, d_out[i], img_bytes,
                cudaMemcpyDeviceToHost, streams[i]));
        }

        for (int i = 0; i < n_images; i++)
            CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        for (int i = 0; i < n_images; i++)
            CUDA_CHECK(cudaStreamDestroy(streams[i]));
    }

    nvtxRangePop();

    for (int i = 0; i < n_images; i++)
        pgm_save(sub_batch[i].output_path, sub_batch[i].host_out, W, H);

    for (int i = 0; i < n_images; i++) {
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_blur[i]));
        CUDA_CHECK(cudaFree(d_edges[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaFree(d_hist[i]));
        CUDA_CHECK(cudaFree(d_cdf[i]));
    }
}

void run_pipeline_multigpu(std::vector<ImageEntry>& batch, int stop_after_stage)
{
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    if (num_gpus < 2) {
        fprintf(stderr, "[multigpu] Warning. Found %d GPU(s). Using device 0 only.\n", num_gpus);
        process_batch_on_device(batch, 0, stop_after_stage);
        return;
    }

    // Section 3.4.2. Split so GPU 0 and GPU 1 get nearly equal counts when N is odd.
    const size_t mid = (batch.size() + 1) / 2;
    std::vector<ImageEntry> sub0(batch.begin(), batch.begin() + static_cast<std::ptrdiff_t>(mid));
    std::vector<ImageEntry> sub1(batch.begin() + static_cast<std::ptrdiff_t>(mid), batch.end());

    // GPU 0 takes the first half. GPU 1 takes the rest. Host threads keep contexts separate.
    std::thread t0([&]() {
        process_batch_on_device(sub0, 0, stop_after_stage);
    });
    std::thread t1([&]() {
        if (!sub1.empty())
            process_batch_on_device(sub1, 1, stop_after_stage);
    });
    t0.join();
    t1.join();
}

void run_pipeline_singlegpu(std::vector<ImageEntry>& batch, int stop_after_stage)
{
    process_batch_on_device(batch, 0, stop_after_stage);
}
