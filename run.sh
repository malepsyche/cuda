#!/usr/bin/env bash
set -e

cmake -S . -B build
cmake --build build -j

# echo "Running vector_addition"
# ./build/vector_addition

# echo "Running matrix_addition"
# ./build/matrix_addition

# echo "Running matrix_multiplication_untiled"
# ./build/matrix_multiplication_untiled

nvcc main.cu -o wave2d -lcusparse

rm -f results.csv

for L in 1 2 4 8
do
    ./wave2d $L
done

python visualize.py