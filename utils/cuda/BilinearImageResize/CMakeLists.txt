cmake_minimum_required(VERSION 3.5.1)
project(BilinearImageResize)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

set(SOURCE_FILES main.cpp converter.cpp resizeCPU.cpp)

include_directories(./include)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64/)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA)

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_86,code=sm_86)

file( GLOB  cu  *.cu)
cuda_add_executable(BilinearImageResize ${SOURCE_FILES} main.cpp resizeGPU.cu)

target_link_libraries(BilinearImageResize
cudnn 
cublas 
cudart_static 
cudart
opencv_core 
opencv_imgproc 
opencv_imgcodecs 
opencv_highgui 
)
