# SC4064 Assignment 2 Report Template

## 1. Objective

This project implements and evaluates a GPU-based solver for the 2D wave equation using three approaches:

- custom CUDA global-memory stencil
- custom CUDA shared-memory tiled stencil
- cuSPARSE CSR SpMV formulation

The focus is performance analysis rather than numerical refinement.

## 2. Methodology

### 2.1 Numerical scheme

Use the finite-difference update:

`u^(n+1)_(i,j) = 2u^n_(i,j) - u^(n-1)_(i,j) + lambda^2 * (u^n_(i+1,j) + u^n_(i-1,j) + u^n_(i,j+1) + u^n_(i,j-1) - 4u^n_(i,j))`

with `lambda = c * dt / dx`, fixed `dx = dy = 0.01`, `dt = 0.005`, `c = 1.0`.

### 2.2 Implementations

**A1: Global-memory stencil**  
Describe direct neighbor loads from global memory and no shared-memory optimization.

**A2: Shared-memory tiled stencil**  
Describe 2D thread blocks, shared-memory tile, halo loading, and why `16x16` was chosen.

**Part B: cuSPARSE**  
Describe reformulation as `u^(n+1) = 2u^n - u^(n-1) + lambda^2 L u^n`, CSR Laplacian construction, and `cusparseSpMV`.

## 3. Visualization

Include selected outputs:

- 2D heatmap
- 3D surface plot
- optional animation screenshots

Briefly comment that the wave amplitude evolves smoothly while boundaries remain fixed at zero.

## 4. Performance Results

### 4.1 Timing

Include a table like this:

| Method | Grid size | Avg kernel time / step (ms) | Total simulation time (ms) | Avg library time (ms) |
|---|---:|---:|---:|---:|
| Global | | | | |
| Shared | | | | |
| cuSPARSE | | | | |

### 4.2 Effective memory bandwidth

Use:

`Bandwidth = Total bytes transferred / Kernel time`

For the stencil, assume `48 bytes` per interior update in double precision.

| Method | Grid size | Effective bandwidth (GB/s) |
|---|---:|---:|
| Global | | |
| Shared | | |
| cuSPARSE | | |

### 4.3 Scaling study

Use fixed `dx`, `dy`, `dt` and vary the physical domain length `L = 1, 2, 4, 8`.

Plot:

- average kernel time vs total grid points
- bandwidth vs total grid points
- total runtime vs total grid points

## 5. Discussion

Address the required questions directly.

### 5.1 How does runtime scale with total grid points?

State whether runtime is approximately linear and explain any deviations.

### 5.2 Does performance scale linearly with problem size?

Discuss launch overhead, cache effects, and saturation of memory bandwidth.

### 5.3 Is the kernel memory-bound or compute-bound?

Argue using the stencil structure:

- low arithmetic intensity
- repeated global memory traffic
- shared-memory version improves data reuse

This usually suggests the stencil is **memory-bound**.

### 5.4 How does block size affect performance?

Discuss:

- occupancy
- halo overhead
- shared-memory footprint
- memory coalescing

## 6. Conclusion

Summarize which implementation performed best and why. A common conclusion is:

- shared-memory stencil outperforms naive global-memory stencil due to reduced global memory traffic
- cuSPARSE is convenient and modular, but may not beat the custom stencil for this structured 5-point operator because of sparse-format overhead
