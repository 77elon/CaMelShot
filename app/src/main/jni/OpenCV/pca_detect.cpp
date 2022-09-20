#include <jni.h>
#include <unistd.h>
#include <cmath>
#include <android/log.h>

#include <iostream>
#include <string>

#include "opencv2/core/mat.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#define TH 3

using namespace std;
using namespace cv;

jsize ary_len;
Scalar Outline_Clr;

// yolov5ncnn Obj의 jObject Type Definition을 위해 추가하였음.
struct Object {
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
};

void set_outline_clr(const int &selection)
{
    if (selection == 0) {
        Outline_Clr = Scalar(255, 255, 255);
    } else if (selection == 1) {
        Outline_Clr = Scalar(0, 255, 0);
    } else if (selection == 2) {
        Outline_Clr = Scalar(255, 0, 0);
    } else if (selection == 3) {
        Outline_Clr = Scalar(0, 0, 255);
    }
}

double getOrientation(const vector<Point> &pts, Mat &img) {
    //findContours 함수에서 전달된 픽셀만큼의 (정확하게는 cv.findcontour 함수에서 전달된 좌표들에 대한 정보가 저장된다.) 처리 공간을 선언한다.
    int sz = static_cast<int>(pts.size());
    Mat data_pts = Mat(sz, 2, CV_64F);

    for (int i = 0; i < data_pts.rows; i++) {
        data_pts.at<double>(i, 0) = pts[i].x;
        data_pts.at<double>(i, 1) = pts[i].y;
    }

    // PCA는 연관성 분석이다.
    // 평균 값을 지나면서 연관도를 가지고 있는 것, Point가 각 vector(2-dim), value다.
    // Plane의 중심 좌표를 기준으로 직선들의 분포를 분석하여, 객체를 분석해주는 기법이다.
    // 직선 근사 분석 기법 + 차원 축소를 통한 평균 값을 지나는 선형 직선의 특성을 분석하는 것.
    // 평균 위치를 지나면서 PCA로 나온 제 1 주성분 벡터(Gradient Vector)와 수직인 2주성분 hypotenuse(빗변의) 방향을 구한다.
    // 이후 차원 축소된 경사 벡터로 부터 추출되는 것이 바로 eigenvector, eigenvalues 이며, 이를 통해서 Gradient Vector Direction을 구할 수 있게 된다.

    // 이를 통해 탐색된 객체 외곽선의 추세, 그리고 고유 벡터를 기반한 Gradient Vector를 구할 수 있다.
    // 실제 계산된 사각형에서의 각도 계산이 각 Gradient Vector (in 4-line, 2-dim)의 평균 값을 구해서 출력되는 것으로 추측된다.

    PCA pca_analysis(data_pts, Mat(), PCA::DATA_AS_ROW);

    // Store the center of the object
    Point cntr = Point(static_cast<int>(pca_analysis.mean.at<double>(0, 0)),
                       static_cast<int>(pca_analysis.mean.at<double>(0, 1)));

    vector<Point2d> eigen_vecs(2);
    vector<double> eigen_val(2);

    // 좌표 별 상관 분석이 완료된 고유 벡터와 값을 저장해주는 과정
    for (int i = 0; i < 2; i++) {
        eigen_vecs[i] = Point2d(pca_analysis.eigenvectors.at<double>(i, 0),
                                pca_analysis.eigenvectors.at<double>(i, 1));
        eigen_val[i] = pca_analysis.eigenvalues.at<double>(i);
    }

    // 2차원 2개의 y, x 좌표 고유 벡터간의 절대 각을 구함으로 나타나는 angle(radian)을 통해 degree 단위로 변환해준다.
    int angle1 = atan2(eigen_vecs[0].y, eigen_vecs[0].x) * 180.0 / M_PI;
    String label = to_string(angle1 - 90) + " degrees";
    //putText(img, label, Point(cntr.x, cntr.y), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2, LINE_AA);

    // 미래 사용을 위한 리턴... 현재 미사용.
    double angle = atan2(eigen_vecs[0].y, eigen_vecs[0].x); // orientation in radians
    return angle;
}

void draw_outline(const Mat &outCopy, vector<vector<Point>> & contours)
{
    if(ary_len != 0)
    {
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            //if (area < 1e3 || 1e7 < area) continue;
            if (area < 1e4) continue;
            drawContours(outCopy, contours, i, Outline_Clr, TH);
            //getOrientation(contours[i], outCopy);
        }
    }
    else
    {
        if (!contours.empty()) {
            // python과 다르게 Vector Point Type ret 되므로, k-v 기반의 max 함수를 사용할 수 없다.
            // 이를 위해 sort operator를 재정의하였으며, 가장 큰 객체에 대해서만 외곽선 검출이 수행된다.
            sort(contours.begin(), contours.end(),
                 [](const vector<Point> &c1, const vector<Point> &c2) {
                     return contourArea(c1, false) > contourArea(c2, false);
                 });

            drawContours(outCopy, contours, 0, Outline_Clr, TH);
            //getOrientation(contours[0], src);
        }
    }
}

void outline_detection(const Mat &imgCopy, const Mat &outCopy)
{
    // for debug
    //imgCopy.copyTo(outCopy, mask);

    Mat gray;
    cvtColor(imgCopy, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);
    //Mat thresh;
    //adaptiveThreshold(gray, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 15, 31);
    //threshold(gray, thresh, 100, 255, THRESH_BINARY | THRESH_OTSU);
    Mat outline;
    Canny(gray, outline, 140, 180, 5, true);
    //Canny(thresh, outline, 120, 180, 3, true);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat morphology;
    morphologyEx(outline, morphology, MORPH_DILATE, kernel, Point(-1, -1), 5);
    vector<vector<Point> > contours;
    findContours(morphology, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    draw_outline(outCopy, contours);

}

void outline_test(const Mat &imgCopy, const Mat &outCopy)
{
    Mat gray;
    cvtColor(imgCopy, gray, COLOR_BGR2GRAY);
    equalizeHist(gray ,gray);

    Mat Thres;
    threshold(gray, Thres, 127, 255, THRESH_BINARY | THRESH_OTSU);
    //adaptiveThreshold(Blur, Thres, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 3, 11);

    Mat outline;
    Canny(Thres, outline, 0, 250, 3, true);

    cv::Size S_Size(3, 3);
    Mat kernel = getStructuringElement(MORPH_RECT, S_Size);
    Mat closed;
    morphologyEx(outline, closed, MORPH_CLOSE, kernel, Point(-1, -1), 1);
    vector<vector<Point> > contours;
    findContours(closed, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for (size_t i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if (area < 1e4) continue;
        drawContours(outCopy, contours, -1, Scalar(255, 255, 255), TH);
        //getOrientation(contours[i], outCopy);
    }

}

extern "C"
JNIEXPORT void JNICALL
Java_com_SSRL_CaMelShot_DetectEdge_DetectMat(JNIEnv *env, jclass clazz, jlong in_mat,
                                             jlong out_mat,
                                             jboolean use_gpu, jobjectArray objs,
                                             jint colorSelect) {

    set_outline_clr(colorSelect);

    Mat &src = *(Mat *) in_mat;
    Mat &out = *(Mat *) out_mat;
    //Mat mask = Mat::zeros(src.size(), src.type());

    ary_len = env->GetArrayLength(objs);


    for (int i = 0; i < ary_len; i++) {
        Object points;
        ary_len = 0;
        jobject obj = env->GetObjectArrayElement(objs, i);
        jclass local_class = env->GetObjectClass(obj);

        jfieldID xid = env->GetFieldID(local_class, "x", "F");
        jfieldID yid = env->GetFieldID(local_class, "y", "F");
        jfieldID hid = env->GetFieldID(local_class, "h", "F");
        jfieldID wid = env->GetFieldID(local_class, "w", "F");

        points.x = (int) env->GetFloatField(obj, xid);
        points.y = (int) env->GetFloatField(obj, yid);
        points.h = (int) env->GetFloatField(obj, hid);
        points.w = (int) env->GetFloatField(obj, wid);

        // smaller than 10%... -> ret
        if(points.w * points.h < (src.rows * src.cols) * 0.1)
        {
            ary_len = 0;
            break;
        }

        //MYJOB - set roi
        Mat imgCopy(src, Rect(points.x, points.y, points.w, points.h));
        Mat outCopy(out, Rect(points.x, points.y, points.w, points.h));

        //rectangle(mask, Point(points.x, points.y), Point(points.x + points.w, points.y + points.h), Scalar(1, 1, 1, 1), FILLED);
        outline_detection(imgCopy, outCopy);
    }

    if(ary_len == 0)
    {
        outline_detection(src, out);
    }

    //outline_test(src, out);
}
