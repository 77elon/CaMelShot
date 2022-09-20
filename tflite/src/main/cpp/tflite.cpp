#include <jni.h>
#include <iostream>
#include <string>
#include <memory>
#include <utility>

#include "opencv2/opencv.hpp"

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/interpreter_builder.h"

//Native pwd is '/'
const char* pwd = "";

//'building', 'light house', 'mountain', 'street'
const std::vector<std::string> labels = {"building", "light house", "mountain", "street", "statue"};

void object_detection(cv::Mat &src, std::string tfmodel)
{

    cv::Mat tensor;
    cv::resize(src, tensor, {300, 300});

    // Load the model
    std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(tfmodel.c_str());

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);

    // Resize input tensors, if desired.
    interpreter->AllocateTensors();

    // Fill `input`.
    auto* input = interpreter->typed_input_tensor<unsigned char>(0);
    std::memcpy(input, tensor.data, 300*300*3);

    interpreter->Invoke();

    auto rects = interpreter->typed_output_tensor<float>(0);
    auto classes = interpreter->typed_output_tensor<float>(1);
    auto scores = interpreter->typed_output_tensor<float>(2);
    auto numDetect = interpreter->typed_output_tensor<float>(3);


    const auto size = interpreter->output_tensor(0)->dims->size;
    for(int i=0; i<size; ++i){
        cv::Point2d tr{rects[i*4+1] * src.cols, rects[i*4] * src.rows};

        cv::rectangle(src,
                      cv::Point2d{rects[i*4+1] * src.cols, rects[i*4] * src.rows},
                      cv::Point2d{rects[i*4+3] * src.cols, rects[i*4+2] * src.rows},
                      {255,0,0}, 2);

        char buf[10];
        std::sprintf(buf, "(%.1f%%)",scores[i]*100);
        cv::putText(src, labels[std::floor(classes[i]+1.5)] + std::string(buf),
                    cv::Point2d{rects[i*4+1] * src.cols, rects[i*4] * src.rows - 4},
                    cv::FONT_ITALIC, 0.6, {0,0,0}, 2);
    }
}

extern "C"
{
    JNIEXPORT void JNICALL
    Java_com_dhnns_opencv_1test_CVFunc_flite_1test(JNIEnv *env, jclass clazz, jlong src_) {
        // TODO: implement flite_test()
        cv::Mat &src = *(cv::Mat *) src_;
        object_detection(src, "best-fp16.tflite");
    }
}
