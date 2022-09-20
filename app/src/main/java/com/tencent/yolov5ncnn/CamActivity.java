package com.tencent.yolov5ncnn;

import static android.Manifest.permission.CAMERA;

import static org.opencv.imgproc.Imgproc.rectangle;

import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvException;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgproc.Imgproc;

import java.util.Collections;
import java.util.List;


public class CamActivity extends MainActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    //MYJOB - permission
    private static final int CAMERA_PERMISSION_CODE = 200;
    //MYJOB yolov5
    private YoloV5Ncnn yolov5ncnn = new YoloV5Ncnn();
    //MYJOB - opencv
    private int m_Camidx = 0;
    private CameraBridgeViewBase m_CameraView;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        //get opencv camera
        m_CameraView = (CameraBridgeViewBase)findViewById(R.id.activity_surface_view);
        m_CameraView.setVisibility(SurfaceView.VISIBLE);
        m_CameraView.setCvCameraViewListener(CamActivity.this);
        m_CameraView.setCameraIndex(m_Camidx);
        //MYJOB Load yolo
        boolean ret_init = yolov5ncnn.Init(getAssets());
        if (!ret_init)
        {
            Log.e("MainActivity", "yolov5ncnn Init failed");
        }
    }

    @Override
    protected void onStart() {
        super.onStart();
        boolean _Permission = true;
        Log.d("opencv Cam", "onStart :: Permission Granted?");
        if(Build.VERSION.SDK_INT >= Build.VERSION_CODES.M){//최소 버전보다 버전이 높은지 확인
            if(checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{CAMERA}, CAMERA_PERMISSION_CODE);
                _Permission = false;
            }
        }
        //
        if(_Permission){
            onCameraPermissionGranted();
        }
    }

    private void onCameraPermissionGranted() {
        List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
        if (cameraViews == null) {
            return;
        }
        for (CameraBridgeViewBase cameraBridgeViewBase: cameraViews) {
            if (cameraBridgeViewBase != null) {
                cameraBridgeViewBase.setCameraPermissionGranted();
            }
        }
    }

    private List<? extends CameraBridgeViewBase> getCameraViewList() {
        return Collections.singletonList(m_CameraView);
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()) {
            Log.d("opencv Cam", "onResum :: OpenCV library found inside package. Using it!");
            m_LoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private BaseLoaderCallback m_LoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    m_CameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onPause()
    {
        super.onPause();
        if (m_CameraView != null)
            m_CameraView.disableView();
    }
    @Override
    public void onDestroy() {
        super.onDestroy();

        if (m_CameraView != null)
            m_CameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        YoloV5Ncnn.Obj[] objects;
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat matInput = inputFrame.rgba();
        //MYJOB - detecting with yolo
        //Mat imageBlob = Dnn.blobFromImage(matInput, 0, new Size(640, 640), new Scalar(0, 0, 0),false,false);
        Bitmap bitmap =  Bitmap.createBitmap(matInput.cols(), matInput.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(matInput, bitmap);
        YoloV5Ncnn.Obj[] objects = yolov5ncnn.Detect(bitmap, true);
        // input objects.length, point ary
        if(!objects.equals(null))
        {
            OpenCV.DetectMat(matInput.getNativeObjAddr(), true, objects);
            showObjects(matInput, objects);
        }
        //return imageBlob;

        return matInput;
    }
    //MYJOB - privated showObject by using opencv
    private void showObjects(Mat mat, YoloV5Ncnn.Obj[] objects) {
        for (int i = 0; i < objects.length; i++){
            Point pt1 = new Point(objects[i].x, objects[i].y);
            Point pt2 = new Point(objects[i].x + objects[i].w, objects[i].y + objects[i].h);
            rectangle(mat, pt1, pt2, new Scalar(255, 0,0), 3);
        }
    }
}
