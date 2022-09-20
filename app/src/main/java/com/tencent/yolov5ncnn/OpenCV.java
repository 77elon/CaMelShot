package com.tencent.yolov5ncnn;

public class OpenCV {
    public native static void DetectMat(long mat, boolean use_gpu, YoloV5Ncnn.Obj[] objs);


}
