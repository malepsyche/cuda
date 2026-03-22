#!/bin/bash
#SBATCH --job-name=wave2d_gpu
#SBATCH --output=logs/wave2d_%j.out
#SBATCH --error=logs/wave2d_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
mkdir -p logs results

module purge
module load cuda || true

make clean
make ARCH=sm_80

./wave2d_solver --method global   --steps 400 --output-dir results/global   --snapshot-every 100
./wave2d_solver --method shared   --steps 400 --block-x 16 --block-y 16 --output-dir results/shared   --snapshot-every 100
./wave2d_solver --method cusparse --steps 400 --output-dir results/cusparse --snapshot-every 100

python3 visualize_wave.py --manifest results/global/snapshots/global_manifest.csv --make-animation
python3 visualize_wave.py --manifest results/shared/snapshots/shared_manifest.csv --make-animation
python3 visualize_wave.py --manifest results/cusparse/snapshots/cusparse_manifest.csv --make-animation

python3 run_experiments.py --binary ./wave2d_solver --out-root results/scaling --steps 400 --with-snapshots
python3 analyze_results.py --summary results/scaling/summary_metrics.csv
