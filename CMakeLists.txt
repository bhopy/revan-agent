cmake_minimum_required(VERSION 3.18)
project(RevanAgent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_executable(revan-agent src/revan_agent.cpp)

# Headers from bin_lib/include (llama.h, ggml.h, etc.)
target_include_directories(revan-agent PRIVATE ${CMAKE_SOURCE_DIR}/bin_lib/include)

# Import libs generated from pre-built DLLs (b8325, CUDA 12.4 runtime)
target_link_directories(revan-agent PRIVATE ${CMAKE_SOURCE_DIR}/bin_lib/lib)

target_link_libraries(revan-agent PRIVATE
    llama
    ggml
    ggml-base
    ggml-cuda
    CUDA::cudart
    CUDA::cublas
    CUDA::cublasLt
    CUDA::cuda_driver
)

if(WIN32)
    target_link_libraries(revan-agent PRIVATE ws2_32 pathcch)
endif()
