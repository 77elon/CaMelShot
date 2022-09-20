LOCAL_PATH := $(call my-dir)
include $(CLEAR_VARS)

#opencv library
OPENCVROOT:= D:\OpenCV-android-sdk\sdk
OPENCV_CAMERA_MODULES:=on
OPENCV_INSTALL_MODULES:=on
OPENCV_LIB_TYPE:=SHARED
include ${OPENCVROOT}\native\jni\OpenCV.mk


LOCAL_MODULE    := opencv-lib
LOCAL_SRC_FILES := pca_test.cpp
LOCAL_LDLIBS += -llog

include $(BUILD_SHARED_LIBRARY)
