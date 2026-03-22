#!/usr/bin/env bash
set -e

cmake -S . -B build
cmake --build build -j

echo "Running vector_addition"
./build/vector_addition

echo "Running matrix_addition"
./build/matrix_addition

echo "Running matrix_multiplication_untiled"
./build/matrix_multiplication_untiled

echo "Running wave2d_naive"
./build/wave2d_naive

echo "Running wave2d_tiled"
./build/wave2d_tiled