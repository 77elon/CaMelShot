#include <jni.h>
#include <string>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_dhnns_tflite_NativeLib_stringFromJNI(
        JNIEnv* env,
        jobject /* this */) {
    std::string hello = "Hello from C++";
    return env->NewStringUTF(hello.c_str());
}