# Install script for directory: /home/storagelab/ncnn/src

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/storagelab/ncnn/build-android-x86_64/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/storagelab/ncnn/build-android-x86_64/src/libncnn.a")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/ncnn" TYPE FILE FILES
    "/home/storagelab/ncnn/src/allocator.h"
    "/home/storagelab/ncnn/src/benchmark.h"
    "/home/storagelab/ncnn/src/blob.h"
    "/home/storagelab/ncnn/src/c_api.h"
    "/home/storagelab/ncnn/src/command.h"
    "/home/storagelab/ncnn/src/cpu.h"
    "/home/storagelab/ncnn/src/datareader.h"
    "/home/storagelab/ncnn/src/gpu.h"
    "/home/storagelab/ncnn/src/layer.h"
    "/home/storagelab/ncnn/src/layer_shader_type.h"
    "/home/storagelab/ncnn/src/layer_type.h"
    "/home/storagelab/ncnn/src/mat.h"
    "/home/storagelab/ncnn/src/modelbin.h"
    "/home/storagelab/ncnn/src/net.h"
    "/home/storagelab/ncnn/src/option.h"
    "/home/storagelab/ncnn/src/paramdict.h"
    "/home/storagelab/ncnn/src/pipeline.h"
    "/home/storagelab/ncnn/src/pipelinecache.h"
    "/home/storagelab/ncnn/src/simpleocv.h"
    "/home/storagelab/ncnn/src/simpleomp.h"
    "/home/storagelab/ncnn/src/simplestl.h"
    "/home/storagelab/ncnn/src/vulkan_header_fix.h"
    "/home/storagelab/ncnn/build-android-x86_64/src/ncnn_export.h"
    "/home/storagelab/ncnn/build-android-x86_64/src/layer_shader_type_enum.h"
    "/home/storagelab/ncnn/build-android-x86_64/src/layer_type_enum.h"
    "/home/storagelab/ncnn/build-android-x86_64/src/platform.h"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake"
         "/home/storagelab/ncnn/build-android-x86_64/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn/ncnn.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/storagelab/ncnn/build-android-x86_64/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/storagelab/ncnn/build-android-x86_64/src/CMakeFiles/Export/lib/cmake/ncnn/ncnn-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/ncnn" TYPE FILE FILES "/home/storagelab/ncnn/build-android-x86_64/src/ncnnConfig.cmake")
endif()

