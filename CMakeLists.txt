cmake_minimum_required(VERSION 3.18)
project(cuda_bench LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Automatically detect your GPU architecture
set(CMAKE_CUDA_ARCHITECTURES 80)

# Optional CUDA flags
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:
    --use_fast_math
>)

find_package(CUDAToolkit REQUIRED)

# Create executables
add_executable(vector_addition vector_addition.cu)
add_executable(matrix_addition matrix_addition.cu)
add_executable(matrix_multiplication_untiled matrix_multiplication_untiled.cu)

# Link CUDA runtime
target_link_libraries(vector_addition PRIVATE CUDA::cudart)
target_link_libraries(matrix_addition PRIVATE CUDA::cudart)
target_link_libraries(matrix_multiplication_untiled PRIVATE CUDA::cudart)

# Enable separable compilation (safe default)
set_target_properties(vector_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_addition PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(matrix_multiplication_untiled PROPERTIES CUDA_SEPARABLE_COMPILATION ON)