# CMake cho CLion (song song với Makefile — dùng cái nào cũng được)
#
#   cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
#   cmake --build cmake-build-release -j
#
# Tuỳ chọn:
#   -DENABLE_CUDA=OFF                 chỉ build bản tuần tự (máy không có CUDA)
#   -DCUDA_ARCHS="61;75"              kiến trúc GPU (61 = GTX 1080/1080 Ti, 75 = RTX 2060)
#   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc   nếu CLion không tự tìm thấy nvcc
cmake_minimum_required(VERSION 3.18)
project(HPC_20261B_Group9_DistributedSGD LANGUAGES C)

option(ENABLE_CUDA "Build CUDA version (cnn_cuda)" ON)
set(CUDA_ARCHS "61;75" CACHE STRING "CUDA architectures (sm_XX)")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_EXTENSIONS ON)            # gnu11 (clock_gettime)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
# KHÔNG dùng -march=native: binary phải chạy được trên CPU của server
set(CMAKE_C_FLAGS_RELEASE "-O3")

# ---- phần C dùng chung (CPU reference) ----
add_library(nn_common STATIC src/net_desc.c src/data.c src/cpu_nn.c)
target_include_directories(nn_common PUBLIC src)
target_compile_options(nn_common PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_libraries(nn_common PUBLIC m)

# ---- bản tuần tự ----
add_executable(cnn_seq src/main_seq.c)
target_link_libraries(cnn_seq PRIVATE nn_common)
target_compile_options(cnn_seq PRIVATE -Wall -Wextra -Wno-unused-parameter)
target_link_options(cnn_seq PRIVATE -static-libgcc)

# ---- bản CUDA ----
if(ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

        add_executable(cnn_cuda src/gpu_kernels.cu src/gpu_train.cu src/main_cuda.cu)
        # sm_61 + sm_75 (SASS) + PTX của 75 cho GPU mới hơn
        set(_archs "")
        foreach(a ${CUDA_ARCHS})
            list(APPEND _archs "${a}-real")
        endforeach()
        list(GET CUDA_ARCHS -1 _last)
        list(APPEND _archs "${_last}-virtual")
        set_target_properties(cnn_cuda PROPERTIES
                CUDA_ARCHITECTURES "${_archs}"
                CUDA_RUNTIME_LIBRARY Static            # cudart tĩnh -> server chỉ cần driver
                LINKER_LANGUAGE CUDA)
        target_compile_options(cnn_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -Xcompiler=-Wall>)
        target_link_libraries(cnn_cuda PRIVATE nn_common)
        target_link_options(cnn_cuda PRIVATE -static-libgcc -static-libstdc++)
        message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION}: building cnn_cuda for ${_archs}")
    else()
        message(WARNING "Không tìm thấy nvcc -> chỉ build cnn_seq. "
                "Đặt -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc hoặc -DENABLE_CUDA=OFF")
    endif()
endif()