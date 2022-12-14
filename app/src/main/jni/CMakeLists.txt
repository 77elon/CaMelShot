project(CaMelShot)
cmake_minimum_required(VERSION 3.6)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/${CMAKE_ANDROID_ARCH_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
add_library(yolov5ncnn SHARED yolov5ncnn_jni.cpp)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../../../../sdk/native/jni)
find_package(OpenCV REQUIRED)
add_library(OpenCV SHARED IMPORTED)

set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp -static-openmp -frtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -static-openmp -frtti")

target_link_libraries(
        yolov5ncnn
        ncnn
        jnigraphics
        ${OpenCV_LIBS}
)
