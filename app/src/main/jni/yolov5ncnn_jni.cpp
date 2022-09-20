// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>
#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "layer.h"
#include "net.h"
#include "benchmark.h"

static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;

static ncnn::Net yolov5;

struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].w * faceobjects[i].h;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}

extern "C" {

// FIXME DeleteGlobalRef is missing for objCls
static jclass objCls = NULL;
static jmethodID constructortorId;
static jfieldID xId;
static jfieldID yId;
static jfieldID wId;
static jfieldID hId;
static jfieldID labelId;
static jfieldID probId;

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_yolov5ncnn_YoloV5Ncnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;
    opt.use_packing_layout = true;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    yolov5.opt = opt;

    // init param
    {
        int ret = yolov5.load_param(mgr, "best.ncnn.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "load_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = yolov5.load_model(mgr, "best.ncnn.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    // init jni glue
    jclass localObjCls = env->FindClass("com/tencent/yolov5ncnn/YoloV5Ncnn$Obj");
    objCls = reinterpret_cast<jclass>(env->NewGlobalRef(localObjCls));

    constructortorId = env->GetMethodID(objCls, "<init>", "(Lcom/tencent/yolov5ncnn/YoloV5Ncnn;)V");

    xId = env->GetFieldID(objCls, "x", "F");
    yId = env->GetFieldID(objCls, "y", "F");
    wId = env->GetFieldID(objCls, "w", "F");
    hId = env->GetFieldID(objCls, "h", "F");
    labelId = env->GetFieldID(objCls, "label", "Ljava/lang/String;");
    probId = env->GetFieldID(objCls, "prob", "F");

    return JNI_TRUE;
}
//TODO - detection
// public native Obj[] Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jobjectArray JNICALL Java_com_tencent_yolov5ncnn_YoloV5Ncnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return NULL;
        //return env->NewStringUTF("no vulkan capable gpu");
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    const int width = info.width;
    const int height = info.height;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
    const int target_size = 640;
    // letterbox pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_android_bitmap_resize(env, bitmap, ncnn::Mat::PIXEL_RGB, w, h);

    // pad to target_size rectangle
    // yolov5/utils/datasets.py letterbox
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    // yolov5
    std::vector<Object> objects;
    {
        const float prob_threshold = 0.25f;
        const float nms_threshold = 0.45f;

        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in_pad.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = yolov5.create_extractor();

        ex.set_vulkan_compute(use_gpu);

        ex.input("in0", in_pad);

        std::vector<Object> proposals;

        // anchor setting from yolov5/models/yolov5s.yaml
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "+++++++++++++++++++++++++");

        // stride 8
        {
            ncnn::Mat out;
            ex.extract("out0", out);

            ncnn::Mat anchors(6);
            anchors[0] = 10.f;
            anchors[1] = 13.f;
            anchors[2] = 16.f;
            anchors[3] = 30.f;
            anchors[4] = 33.f;
            anchors[5] = 23.f;

            std::vector<Object> objects8;
            generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        // stride 16
        {
            ncnn::Mat out;
            ex.extract("out1", out);

            ncnn::Mat anchors(6);
            anchors[0] = 30.f;
            anchors[1] = 61.f;
            anchors[2] = 62.f;
            anchors[3] = 45.f;
            anchors[4] = 59.f;
            anchors[5] = 119.f;

            std::vector<Object> objects16;
            generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;
            ex.extract("out2", out);

            ncnn::Mat anchors(6);
            anchors[0] = 116.f;
            anchors[1] = 90.f;
            anchors[2] = 156.f;
            anchors[3] = 198.f;
            anchors[4] = 373.f;
            anchors[5] = 326.f;

            std::vector<Object> objects32;
            generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].x - (wpad / 2)) / scale;
            float y0 = (objects[i].y - (hpad / 2)) / scale;
            float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;
            objects[i].w = x1 - x0;
            objects[i].h = y1 - y0;
        }
    }

    // objects to Obj[]
    static const char* class_names[] = {
            "building", "light house", "mailbox", "mountain", "sculpture", "street", "street_light", "telephone_box"
    };

    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%d objs detected", objects.size());

    for (size_t i=0; i<objects.size(); i++)
    {

        jobject jObj = env->NewObject(objCls, constructortorId, thiz);
        //__android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "Execute1");

        env->SetFloatField(jObj, xId, objects[i].x);
        env->SetFloatField(jObj, yId, objects[i].y);
        env->SetFloatField(jObj, wId, objects[i].w);
        env->SetFloatField(jObj, hId, objects[i].h);
        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
        env->SetFloatField(jObj, probId, objects[i].prob);
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "\nx: %.2f\ny: %.2f\nw: %.2f\nh: %.2f", objects[i].x, objects[i].y, objects[i].w, objects[i].h);
        env->SetObjectArrayElement(jObjArray, i, jObj);
    }

    double elasped = ncnn::get_current_time() - start_time;
    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%.2fms   detect", elasped);

    return jObjArray;
}
//MYJOB - detect with matrix
//JNIEXPORT jobjectArray JNICALL Java_com_tencent_yolov5ncnn_YoloV5Ncnn_DetectMat(JNIEnv* env, jobject thiz, jobject mat, jboolean use_gpu)
//{
//    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
//    {
//        return NULL;
//        //return env->NewStringUTF("no vulkan capable gpu");
//    }
//    ncnn::Mat& in = *(ncnn::Mat*)mat;
//
//    double start_time = ncnn::get_current_time();
//
//    const int width = in.w;
//    const int height = in.h;
//
//    // ncnn from bitmap
//    const int target_size = 640;
//
//    // letterbox pad to multiple of 32
//
//    int w = width;
//    int h = height;
//    float scale = 1.f;
//    if (w > h)
//    {
//        scale = (float)target_size / w;
//        w = target_size;
//        h = h * scale;
//    }
//    else
//    {
//        scale = (float)target_size / h;
//        h = target_size;
//        w = w * scale;
//    }
//
//    // pad to target_size rectangle
//    // yolov5/utils/datasets.py letterbox
//    int wpad = (w + 31) / 32 * 32 - w;
//    int hpad = (h + 31) / 32 * 32 - h;
//    ncnn::Mat in_pad;
//    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
//
//    // yolov5
//    std::vector<Object> objects;
//    {
//        const float prob_threshold = 0.25f;
//        const float nms_threshold = 0.45f;
//
//        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
//        in_pad.substract_mean_normalize(0, norm_vals);
//
//        ncnn::Extractor ex = yolov5.create_extractor();
//
//        ex.set_vulkan_compute(use_gpu);
//
//        ex.input("in0", in_pad);
//
//        std::vector<Object> proposals;
//
//        // anchor setting from yolov5/models/yolov5s.yaml
//
//        // stride 8
//        {
//            ncnn::Mat out;
//            ex.extract("out0", out);
//
//            ncnn::Mat anchors(6);
//            anchors[0] = 10.f;
//            anchors[1] = 13.f;
//            anchors[2] = 16.f;
//            anchors[3] = 30.f;
//            anchors[4] = 33.f;
//            anchors[5] = 23.f;
//
//            std::vector<Object> objects8;
//            generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);
//
//            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
//        }
//
//        // stride 16
//        {
//            ncnn::Mat out;
//            ex.extract("out1", out);
//
//            ncnn::Mat anchors(6);
//            anchors[0] = 30.f;
//            anchors[1] = 61.f;
//            anchors[2] = 62.f;
//            anchors[3] = 45.f;
//            anchors[4] = 59.f;
//            anchors[5] = 119.f;
//
//            std::vector<Object> objects16;
//            generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);
//
//            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
//        }
//
//        // stride 32
//        {
//            ncnn::Mat out;
//            ex.extract("out2", out);
//
//            ncnn::Mat anchors(6);
//            anchors[0] = 116.f;
//            anchors[1] = 90.f;
//            anchors[2] = 156.f;
//            anchors[3] = 198.f;
//            anchors[4] = 373.f;
//            anchors[5] = 326.f;
//
//            std::vector<Object> objects32;
//            generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);
//
//            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
//        }
//
//        // sort all proposals by score from highest to lowest
//        qsort_descent_inplace(proposals);
//
//        // apply nms with nms_threshold
//        std::vector<int> picked;
//        nms_sorted_bboxes(proposals, picked, nms_threshold);
//
//        int count = picked.size();
//
//        objects.resize(count);
//        for (int i = 0; i < count; i++)
//        {
//            objects[i] = proposals[picked[i]];
//
//            // adjust offset to original unpadded
//            float x0 = (objects[i].x - (wpad / 2)) / scale;
//            float y0 = (objects[i].y - (hpad / 2)) / scale;
//            float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
//            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;
//
//            // clip
//            x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
//            y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
//            x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
//            y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);
//
//            objects[i].x = x0;
//            objects[i].y = y0;
//            objects[i].w = x1 - x0;
//            objects[i].h = y1 - y0;
//        }
//    }
//
//    // objects to Obj[]
//    static const char* class_names[] = {
//            "building", "light house", "mailbox", "mountain", "sculpture", "street", "street_light", "telephone_box"
//    };
//
//    jobjectArray jObjArray = env->NewObjectArray(objects.size(), objCls, NULL);
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%d objs detected", objects.size());
//    for (size_t i=0; i<objects.size(); i++)
//    {
//        jobject jObj = env->NewObject(objCls, constructortorId, thiz);
//
//        env->SetFloatField(jObj, xId, objects[i].x);
//        env->SetFloatField(jObj, yId, objects[i].y);
//        env->SetFloatField(jObj, wId, objects[i].w);
//        env->SetFloatField(jObj, hId, objects[i].h);
//        env->SetObjectField(jObj, labelId, env->NewStringUTF(class_names[objects[i].label]));
//        env->SetFloatField(jObj, probId, objects[i].prob);
//        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "\nx: %.2f\ny: %.2f\nw: %.2f\nh: %.2f", objects[i].x, objects[i].y, objects[i].w, objects[i].h);
//        env->SetObjectArrayElement(jObjArray, i, jObj);
//    }
//
//    double elasped = ncnn::get_current_time() - start_time;
//    __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn", "%.2fms   detect", elasped);
//
//    return jObjArray;
//}

}

using namespace cv;
using namespace std;

void drawAxis(Mat&, Point, Point, Scalar, const float);
double getOrientation(const vector<Point> &, Mat&);

void drawAxis(Mat& img, Point p, Point q, Scalar colour, const float scale = 0.2)
{
    double angle = atan2( (double) p.y - q.y, (double) p.x - q.x ); // angle in radians
    double hypotenuse = sqrt( (double) (p.y - q.y) * (p.y - q.y) + (p.x - q.x) * (p.x - q.x));
    // Here we lengthen the arrow by a factor of scale
    q.x = (int) (p.x - scale * hypotenuse * cos(angle));
    q.y = (int) (p.y - scale * hypotenuse * sin(angle));
    line(img, p, q, colour, 1, LINE_AA);
    // create the arrow hooks
    p.x = (int) (q.x + 9 * cos(angle + CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle + CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
    p.x = (int) (q.x + 9 * cos(angle - CV_PI / 4));
    p.y = (int) (q.y + 9 * sin(angle - CV_PI / 4));
    line(img, p, q, colour, 1, LINE_AA);
}
double getOrientation(const vector<Point> &pts, Mat &img)
{
    //Construct a buffer used by the pca analysis
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64F);
    for (int i = 0; i < data_pts.rows; i++)
    {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }
    //Perform PCA analysis
    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);
    //Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                       static_cast<int>(pca_analysis.mean.at<double>(0, 1)));
    //Store the eigenvalues and eigenvectors
    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);
    for (int i = 0; i < 2; i++)
    {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }
    // Draw the principal components
    //circle(img, cntr, 3, Scalar(255, 0, 255), 2);
    Point p1 = cntr + 0.02 * Point(static_cast<int>(eigen_vecs[0].x * eigen_val[0]), static_cast<int>(eigen_vecs[0].y * eigen_val[0]));
    Point p2 = cntr - 0.02 * Point(static_cast<int>(eigen_vecs[1].x * eigen_val[1]), static_cast<int>(eigen_vecs[1].y * eigen_val[1]));
    //green
    //drawAxis(img, cntr, p1, Scalar(0, 255, 0), 1);
    //yellow
    //drawAxis(img, cntr, p2, Scalar(255, 255, 0), 5);

    int angle1 = atan2(eigen_vecs[0].y, eigen_vecs[0].x) * 180.0 / M_PI;
    String label = to_string(angle1 - 90) + " degrees";
    putText(img, label, Point(cntr.x, cntr.y), FONT_HERSHEY_SIMPLEX, 1, Scalar (0, 0, 0), 2, LINE_AA);
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;
}

extern "C" {

JNIEXPORT void JNICALL Java_com_tencent_yolov5ncnn_OpenCV_DetectMat
        (JNIEnv *, jclass, jlong mat, jboolean) {
    // TODO: implement DetectMat()
    Mat &src = *(Mat *) mat;

    Mat gray;
    cvtColor(src, gray, COLOR_BGR2GRAY);
    equalizeHist(gray ,gray);

    // Convert image to binary
    Mat bw;
    adaptiveThreshold(gray, bw, 200, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 2);

    Mat Blur;
    cv::Size B_Size(5, 5);
    GaussianBlur(gray, Blur, B_Size, 0, 0);

    Mat Thres;
    threshold(Blur, Thres, 127, 255, THRESH_BINARY | THRESH_OTSU);
    //adaptiveThreshold(Blur, Thres, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 11);

    Mat Outline;
    Canny(Thres, Outline, 0, 250, 3, true);

    cv::Size S_Size(3, 3);
    Mat kernel = getStructuringElement(MORPH_RECT, S_Size);
    Mat closed;
    morphologyEx(Outline, closed, MORPH_CLOSE, kernel);

    vector<vector<Point> > contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


    // Find all the contours in the thresholded image
    //vector<vector<Point> > contours;
    //findContours(bw, contours, RETR_LIST, CHAIN_APPROX_NONE);
    for (size_t i = 0; i < contours.size(); i++)
    {
        // Calculate the area of each contour
        double area = contourArea(contours[i]);
        // Ignore contours that are too small or too large
        if (area < 1e2 || 1e5 < area) continue;
        // Draw each contour only for visualisation purposes
        drawContours(src, contours, -1, Scalar(255, 255, 255), 3);
        // Find the orientation of each shape
        getOrientation(contours[i], src);
    }

}

}
