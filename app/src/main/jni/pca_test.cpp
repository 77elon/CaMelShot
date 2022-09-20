//
// Created by daeha on 2022-04-02.
//

#include "pca_test.h"
#include <jni.h>

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


    JNIEXPORT void JNICALL Java_com_dhnns_opencv_1test_CVFunc_ConvertRGBtoGray (JNIEnv *, jclass, jlong matAddrInput, jlong matAddrResult) {
        Mat &matInput = *(Mat *) matAddrInput;
        Mat &matResult = *(Mat *) matAddrResult;

        cvtColor(matInput, matResult, COLOR_RGBA2GRAY);
    }

    JNIEXPORT void JNICALL
    Java_com_dhnns_opencv_1test_CVFunc_PCACompute(JNIEnv *env, jclass clazz, jlong src_) {
        // TODO: implement PCACompute()
        Mat &src = *(Mat *) src_;

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
