# SC4064 GPU Programming Assignment 2 — 2D Wave Equation Solver

This package includes all core files needed to satisfy the project requirements in Sections 3.1 to 3.10:

- **Part A1**: custom CUDA global-memory stencil
- **Part A2**: custom CUDA shared-memory tiled stencil with halo loading
- **Part B**: cuSPARSE implementation using CSR Laplacian + `cusparseSpMV`
- **Visualization**: snapshot export, 2D heatmaps, 3D surface plots, optional animation
- **Timing**: kernel time per timestep, total simulation time, cuSPARSE call time
- **Bandwidth study**: effective memory bandwidth using the 48-byte/update stencil model
- **Scaling study**: fixed `dx`, `dy`, `dt`; enlarged physical domain sizes `L = 1, 2, 4, 8`
- **Submission support**: Makefile, SLURM script, analysis scripts

## File Overview

- `wave2d_solver.cu`  
  Single CUDA source file containing all implementations in separate code paths/functions.

- `Makefile`  
  Builds the solver with `nvcc` and links cuSPARSE.

- `job_submission.slurm`  
  Example cluster submission script.

- `run_experiments.py`  
  Runs the required scaling study and collects metrics into one summary CSV.

- `visualize_wave.py`  
  Generates heatmaps, surface plots, and optional animation from exported snapshots.

- `analyze_results.py`  
  Produces performance plots from the scaling-study CSV.

- `report_template.md`  
  Concise write-up structure for the final <=3 page report.

## Numerical Model

The solver implements the finite-difference scheme:

`u^(n+1)_(i,j) = 2u^n_(i,j) - u^(n-1)_(i,j) + lambda^2 * (u^n_(i+1,j) + u^n_(i-1,j) + u^n_(i,j+1) + u^n_(i,j-1) - 4u^n_(i,j))`

with

`lambda = c * dt / dx`

and zero Dirichlet boundary conditions.

Initial conditions used:

- `u(x, y, 0) = sin(pi x) sin(pi y)`
- `du/dt(x, y, 0) = 0`

The first step is initialized with:

`u^1 = u^0 + 0.5 * lambda^2 * Lap(u^0)`

## Build

```bash
make ARCH=sm_80
```

Adjust `ARCH` if your GPU uses another compute capability.

## Basic Runs

### A1: Global-memory stencil

```bash
./wave2d_solver --method global --output-dir output/global
```

### A2: Shared-memory tiled stencil

```bash
./wave2d_solver --method shared --block-x 16 --block-y 16 --output-dir output/shared
```

### Part B: cuSPARSE CSR Laplacian

```bash
./wave2d_solver --method cusparse --output-dir output/cusparse
```

## Important Arguments

```bash
--nx N --ny N
--steps N
--dx 0.01 --dy 0.01 --dt 0.005 --c 1.0
--block-x 16 --block-y 16
--snapshot-every 100
--output-dir results/run_name
--no-snapshots
```

## Output Files

Each run writes:

- `*_metrics.csv` — scalar timing and bandwidth metrics
- `snapshots/*_manifest.csv` — snapshot index for visualization
- `snapshots/*.csv` — selected wave-field frames

Metrics recorded include:

- average kernel time per timestep
- total kernel time
- total simulation time
- average library call time
- total library call time
- effective memory bandwidth

## Bandwidth Formula Used

For the 5-point stencil in double precision:

- 5 reads
- 1 write
- `6 * 8 = 48 bytes` per grid update

Effective bandwidth is computed as:

`Bandwidth = Total bytes transferred / Kernel time`

The code uses **interior points only** for the bandwidth estimate because only those points perform the actual stencil update.

## Scaling Study

The assignment specifies a **performance study**, not a resolution study.

This package supports the required scaling strategy:

- keep `dx`, `dy`, `dt` fixed
- increase physical domain size `L`
- automatically convert `L` to `nx`, `ny`

Run:

```bash
python3 run_experiments.py --binary ./wave2d_solver --out-root results/scaling --steps 400
```

This generates:

- `results/scaling/summary_metrics.csv`

Then analyze:

```bash
python3 analyze_results.py --summary results/scaling/summary_metrics.csv
```

This generates plots for:

- kernel time vs total grid points
- bandwidth vs total grid points
- total simulation time vs total grid points

## Visualization

After a run with snapshots enabled:

```bash
python3 visualize_wave.py --manifest output/shared/snapshots/shared_manifest.csv --make-animation
```

This generates:

- `heatmap_step_*.png`
- `surface_step_*.png`
- `wave_animation.mp4` or fallback GIF

## Suggested Block-Size Justification

A reasonable report justification for `16 x 16` is:

- it uses 2D thread blocks as required
- 256 threads per block is a common occupancy-friendly choice
- shared-memory tile size becomes `(16 + 2) x (16 + 2)`, which is small enough to fit comfortably in shared memory
- it balances parallelism, halo overhead, and scheduling efficiency

You can also test `8x8`, `16x16`, and `32x8` to answer the block-size analysis question.

## Suggested Workflow For Submission

1. `make`
2. Run `global`, `shared`, and `cusparse`
3. Generate visualization images
4. Run the scaling study
5. Generate analysis plots
6. Write the final report using `report_template.md`
7. Zip the code and scripts, and submit the report separately as PDF

## Notes

- The cuSPARSE version uses a CSR Laplacian matrix and `cusparseSpMV`
- Boundary rows are stored as zero rows so boundary values remain zero after the combine step
- The shared-memory kernel explicitly performs halo loading
- The code warns if the CFL condition may be violated
