cmake_minimum_required(VERSION 3.18)
project(gpu_compression CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA settings
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_60")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/compression.cu
)

# Create the main library
add_library(compression STATIC ${SOURCES})
set_target_properties(compression PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Create the test executable
add_executable(compression_test tests/test_compression.cu)
target_link_libraries(compression_test compression)
