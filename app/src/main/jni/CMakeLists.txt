project(yolov5ncnn)

cmake_minimum_required(VERSION 3.4.1)

#yolo
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
add_library(yolov5ncnn SHARED yolov5ncnn_jni.cpp)
##
#opencv
set(pathPROJECT C:/Users/SonChanHyeok/AndroidStudioProjects/myYoloV5)
set(pathOPENCV ${pathPROJECT}/sdk)
set(pathLIBOPENCV_JAVA ${pathOPENCV}/native/libs/${ANDROID_ABI}/libopencv_java4.so)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
include_directories(${pathOPENCV}/native/jni/include)

add_library(lib_opencv SHARED IMPORTED)
set_target_properties(lib_opencv PROPERTIES IMPORTED_LOCATION ${pathLIBOPENCV_JAVA})

target_link_libraries(yolov5ncnn
    ncnn

    jnigraphics

    lib_opencv

)
