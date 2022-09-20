cmake_minimum_required(VERSION 3.6)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/../native/jni)
find_package(OpenCV REQUIRED)
# dummy target to bring libc++_shared.so into packages
add_library(opencv_jni_shared STATIC dummy.cpp)
