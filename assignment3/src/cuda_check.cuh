#pragma once

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Matches the course error-checking handout (typos there are fixed here).
// Wrap every CUDA runtime call that returns cudaError_t.
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err = (call);                                                        \
        if (err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA Error: %s (at %s:%d)\n", cudaGetErrorString(err),      \
                    __FILE__, __LINE__);                                                 \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    } while (0)

// Call after each <<<>>> kernel launch (kernels do not return cudaError_t).
#define CUDA_KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
