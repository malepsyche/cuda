cmake_minimum_required(VERSION 3.18)
project(cuda_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# IMPORTANT: don't use "native" on this cluster
# Pick a conservative set, or just one if you want faster builds.
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

add_executable(vector_addition vector_addition.cu)
add_executable(matrix_addition matrix_addition.cu)
add_executable(matrix_multiplication_untiled matrix_multiplication_untiled.cu)

# If you really want --use_fast_math, set it per-target (not globally)
target_compile_options(vector_addition PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
target_compile_options(matrix_addition PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
target_compile_options(matrix_multiplication_untiled PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)