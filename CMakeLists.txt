cmake_minimum_required(VERSION 3.20)

project(CudaBifur LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Найти CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Создать динамическую библиотеку
add_library(CudaLib SHARED 
    cudaLibrary.cu
    cudaMacros.cu
    hostLibrary.cu
)

# Указать директории для include-файлов
target_include_directories(CudaLib PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}  # Добавляет текущую директорию для поиска .h/.cuh файлов
)

# Линкуем с библиотекой cudart
target_link_libraries(CudaLib PRIVATE CUDA::cudart)

# Если необходимо, экспортируем символы для Windows
target_compile_definitions(CudaLib PRIVATE CUDALIB_EXPORTS)

# Использование CUDA_API макроса для правильного экспорта/импорта функций