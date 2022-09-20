package com.SSRL.CaMelShot;

import static android.Manifest.permission.CAMERA;
import static android.hardware.Camera.Parameters.FLASH_MODE_OFF;
import static android.hardware.Camera.Parameters.FLASH_MODE_TORCH;
import static org.opencv.imgproc.Imgproc.rectangle;
import static java.lang.Math.abs;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.graphics.drawable.Drawable;
import android.hardware.Camera;
import android.hardware.SensorManager;
import android.location.Address;
import android.location.Geocoder;
import android.location.Location;
import android.location.LocationManager;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.AsyncTask;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.util.Log;
import android.view.MotionEvent;
import android.view.OrientationEventListener;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.ActionBar;

import com.bumptech.glide.Glide;
import com.bumptech.glide.request.RequestOptions;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;

public class CamActivity extends MainActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    //MYJOB - permission
    private static final int CAMERA_PERMISSION_CODE = 200;
    //MYJOB yolov5
    private final DetectEdge yolov5ncnn = new DetectEdge();
    private CameraBridgeViewBase m_CameraView;
    private Activity activity;
    private static JavaCameraView mCameraPreview;
    private static final int CAMERA_PERMISSION_REQUEST_CODE = 200;
    private double touch_interval_X = 0; // X 터치 간격
    private double touch_interval_Y = 0; // Y 터치 간격
    private int zoom_in_count = 0; // 줌 인 카운트
    private int zoom_out_count = 0; // 줌 아웃 카운트
    private int touch_zoom = 0; // 줌 크기
    private int flash_count = 0;
    private SurfaceHolder mHolder;
    public static int degreePhone;
    private Camera.CameraInfo mCameraInfo;
    public static int degrees;
    static ImageView imageView;
    static ImageView imageViewAuto;
    ImageButton btnThumnail;
    File[] imageFiles;
    double latitude;
    double longitude;
    double altitude;
    public static int colorSelect = 0;
    private int cnt = 0;
    static DetectEdge.Obj[] objects;
    private GpsTracker gpsTracker;
    private static final int GPS_ENABLE_REQUEST_CODE = 2001;
    private static final int PERMISSIONS_REQUEST_CODE = 100;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN,
                WindowManager.LayoutParams.FLAG_FULLSCREEN);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON,
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_camera);

        ImageButton btnCapture = findViewById(R.id.btnCapture);
        ImageButton btnFlash = findViewById(R.id.btnFlash);
        ImageButton btnReverse = findViewById(R.id.btnReverse);
        btnThumnail = findViewById(R.id.btnThumnail);
        Button btnLineOnOff = findViewById(R.id.btnLineOnOff);
        Button btnLineColor = findViewById(R.id.btnLineColor);
        Button btnAuto = findViewById(R.id.btnAuto);
        Button btnRotate = findViewById(R.id.btnRotate);
        //get opencv camera
        imageView = findViewById(R.id.imageViewCamera);
        imageViewAuto = findViewById(R.id.imageViewCameraAuto);
        /*
        ActionBar bar = getSupportActionBar();
        bar.hide();
*/
        //btnFlash.setBackgroundResource(R.drawable.flash_off);
        //btnFlash.setScaleType(ImageView.ScaleType.CENTER_CROP);
        //arg0: 기울기
        // 0˚ (portrait)
        // 90˚
        // 180˚
        // 270˚ (landscape)
        OrientationEventListener orientEventListener = new OrientationEventListener(this,
                SensorManager.SENSOR_DELAY_NORMAL) {

            @Override
            public void onOrientationChanged(int arg0) {
                degrees = arg0;

                // 수정 필요...
                //imageViewAuto.setRotation(imageViewAuto.getRotation() - (arg0));
                imageViewAuto.setRotation(360 - (arg0-270));
                //arg0: 기울기
                // 0˚ (portrait)
                if (arg0 >= 315 || arg0 < 45) {
                    degreePhone = 0;
                }
                // 90˚
                else if (arg0 >= 45 && arg0 < 135) {
                    degreePhone = 270;
                }
                // 180˚
                else if (arg0 >= 135 && arg0 < 225) {
                    degreePhone = 180;
                }
                // 270˚ (landscape)
                else if (arg0 >= 225 && arg0 < 315) {
                    degreePhone = 90;
                }
                if (arg0 == 0 || arg0 == 360 || arg0 == 90 || arg0 == 180 || arg0 == 270) {
                    btnThumnail.setRotation(180 - (arg0 + 270));
                }
            }
        };
        camDraw();
        //핸들러 활성화
        orientEventListener.enable();
        //인식 오류 시, Toast 메시지 출력
        if (!orientEventListener.canDetectOrientation()) {
            finish();
        }

        m_CameraView = findViewById(R.id.activity_surface_view);
        m_CameraView.setVisibility(SurfaceView.VISIBLE);
        m_CameraView.setCvCameraViewListener(CamActivity.this);
        //MYJOB - opencv
        int m_Camidx = 0;
        m_CameraView.setCameraIndex(m_Camidx);
        activity = this;
        thumnail();

        btnCapture.setOnClickListener(v -> {
            JavaCameraView.params = JavaCameraView.mCamera.getParameters();
            if (flash_count == 1) { // 플래시 켜기1
                JavaCameraView.params.setFlashMode(FLASH_MODE_TORCH);
                JavaCameraView.mCamera.setParameters(JavaCameraView.params);
            }
            JavaCameraView.mCamera.takePicture(shutterCallback, rawCallback, jpegCallback);
            JavaCameraView.params.setFlashMode(FLASH_MODE_OFF);
            JavaCameraView.mCamera.setParameters(JavaCameraView.params);
            Handler handler = new Handler();
            handler.postDelayed(this::thumnail, 3000);
            gpsTracker = new GpsTracker(this);

            double latitude = gpsTracker.getLatitude();
            double longitude = gpsTracker.getLongitude();
            String address = getCurrentAddress(latitude, longitude);
            Toast.makeText(getApplicationContext(), address,Toast.LENGTH_SHORT);
        });

        btnReverse.setOnClickListener(v -> {
            switchCamera();
            thumnail();
        });

        btnLineOnOff.setOnClickListener(v -> {
            if (imageView.getVisibility() == View.VISIBLE) {
                imageView.setVisibility(View.INVISIBLE);
                imageViewAuto.setVisibility(View.INVISIBLE);
            } else {
                imageView.setVisibility(View.VISIBLE);
                imageViewAuto.setVisibility(View.INVISIBLE);
            }
        });

        btnLineColor.setOnClickListener(v -> {
            //imageView.setImageBitmap(null);
            lineColor();
        });

        btnFlash.setOnClickListener(v -> {
            if (flash_count == 1) {
                btnFlash.setImageResource(R.drawable.flash_off);
                Log.v("Flash_test", "on");
            } else {
                btnFlash.setImageResource(R.drawable.flash_on);
                Log.v("Flash_test", "off");
            }
            flashOnOff();
        });

        btnThumnail.setOnClickListener(v -> {
            try {
                Intent gallery = getPackageManager().getLaunchIntentForPackage("com.sec.android.gallery3d");
                //Intent galleryIntent = new Intent(Intent.ACTION_VIEW, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                gallery.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                startActivity(gallery);
            } catch (Exception e) {
                thumnail();
            }
        });

        btnRotate.setOnClickListener(v -> {
            if(imageViewAuto.getVisibility() == View.VISIBLE)
            {
                imageViewAuto.setRotation((imageViewAuto.getRotation() + 90) % 360);
            }
            else
            {
                imageView.setRotation((imageView.getRotation() + 90) % 360);
            }
        });
        // Switched
        btnAuto.setOnClickListener(v -> {
            if(imageViewAuto.getVisibility() == View.VISIBLE)
            {
                imageViewAuto.setVisibility(View.INVISIBLE);
                imageView.setVisibility(View.VISIBLE);
            }
            else
            {
                imageViewAuto.setVisibility(View.VISIBLE);
                imageView.setVisibility(View.INVISIBLE);
            }
        });

        //MYJOB Load yolo
        boolean ret_init = yolov5ncnn.Init(getAssets());
        if (!ret_init) {
            Log.e("MainActivity", "yolov5ncnn Init failed");
        }

    }

    @Override
    protected void onStart() {
        super.onStart();
        boolean _Permission = true;
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {//최소 버전보다 버전이 높은지 확인
            if (checkSelfPermission(CAMERA) != PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{CAMERA, Manifest.permission.ACCESS_MEDIA_LOCATION,
                        Manifest.permission.WRITE_EXTERNAL_STORAGE, Manifest.permission.ACCESS_COARSE_LOCATION, Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.INTERNET}, CAMERA_PERMISSION_REQUEST_CODE);
                _Permission = false;
            }
        }
        //
        if (_Permission) {
            onCameraPermissionGranted();
        }
    }

    private void onCameraPermissionGranted() {
        List<? extends CameraBridgeViewBase> cameraViews = getCameraViewList();
        for (CameraBridgeViewBase cameraBridgeViewBase : cameraViews) {
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
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    private final BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                m_CameraView.enableView();
            } else {
                super.onManagerConnected(status);
            }
            mCameraInfo = new Camera.CameraInfo();
            SurfaceView mSurfaceView = m_CameraView;
            mHolder = mSurfaceView.getHolder();
        }
    };

    @Override
    public void onPause() {
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
        DetectEdge.Obj[] objects;
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat matInput = inputFrame.rgba();
        int wid = matInput.cols();
        int hei = matInput.rows();

        Log.d("myDebug", String.valueOf(wid) + "--" + String.valueOf(hei));

        //MYJOB - detecting with yolo
        if(cnt++ == 0)
            objects = yolov5ncnn.DetectYolo(matInput.getNativeObjAddr(), false);
        else if(cnt >= 5) cnt = 0;
        showObjects(matInput, objects);
        //

        if ( matInput == null )
            matInput = new Mat(matInput.rows(), matInput.cols(), matInput.type());

        //ConvertRGBtoGray(matInput.getNativeObjAddr(), matInput.getNativeObjAddr());
        if(mCameraPreview.lensId == 98) {
            Core.flip(matInput, matInput, 1);
        }
        return matInput;
    }

    //MYJOB - privated showObject for mat format detecting
    private void showObjects(Mat mat, DetectEdge.Obj[] objects) {
        for (DetectEdge.Obj object : objects) {
            Point pt1 = new Point(object.x, object.y);
            Point pt2 = new Point(object.x + object.w, object.y + object.h);
            rectangle(mat, pt1, pt2, new Scalar(255, 0, 0), 3);
            //set text box
            {
                String text = object.label + " = " + String.format("%.1f", object.prob * 100) + "%";
                float x = object.x;
                float y = object.y;
                Imgproc.putText(mat, text, new Point(x, y), 1, 3, new Scalar(255, 0, 0), 2, -1);
            }
        }
    }

    public boolean onTouchEvent(MotionEvent event) {
        switch (event.getAction() & MotionEvent.ACTION_MASK) {
            case MotionEvent.ACTION_DOWN: // 싱글 터치
                //Camera.Parameters params = mCameraPreview.mCamera.getParameters();
                // Error
                mCameraPreview.params = JavaCameraView.mCamera.getParameters();
                // 오토 포커스 설정
                mCameraPreview.mCamera.autoFocus((success, camera) -> {});
                break;
            case MotionEvent.ACTION_MOVE: // 터치 후 이동 시
                //Camera.Parameters params = mCameraPreview.mCamera.getParameters();
                JavaCameraView.params = JavaCameraView.mCamera.getParameters();
                //Parameters params = mCameraPreview.mCamera.getParameters();
                if (event.getPointerCount() == 2) { // 터치 손가락 2개일 때
                    double now_interval_X = abs(event.getX(0) - event.getX(1)); // 두 손가락 X좌표 차이 절대값
                    double now_interval_Y = abs(event.getY(0) - event.getY(1)); // 두 손가락 Y좌표 차이 절대값
                    if (touch_interval_X < now_interval_X && touch_interval_Y < now_interval_Y) { // 이전 값과 비교
                        // 여기에 확대기능에 대한 코드를 정의 하면됩니다. (두 손가락을 벌렸을 때 분기점입니다.)
                        zoom_in_count++;
                        if (zoom_in_count > 5) { // 카운트를 세는 이유 : 너무 많은 호출을 줄이기 위해
                            zoom_in_count = 0;
                            touch_zoom += 5;
                            JavaCameraView.params = JavaCameraView.mCamera.getParameters();
                            activity.setProgress(touch_zoom / 6);
                            if (JavaCameraView.params.getMaxZoom() < touch_zoom) {
                                touch_zoom = JavaCameraView.params.getMaxZoom();
                            }
                            JavaCameraView.params.setZoom(touch_zoom);
                            JavaCameraView.mCamera.setParameters(JavaCameraView.params);
                        }
                    }
                    if (touch_interval_X > now_interval_X && touch_interval_Y > now_interval_Y) {
                        // 여기에 축소기능에 대한 코드를 정의 하면됩니다. (두 손가락 사이를 좁혔을 때 분기점입니다.)
                        zoom_out_count++;
                        if (zoom_out_count > 5) {
                            zoom_out_count = 0;
                            touch_zoom -= 10;
                            JavaCameraView.params = JavaCameraView.mCamera.getParameters();
                            activity.setProgress(touch_zoom / 6);
                            if (0 > touch_zoom)
                                touch_zoom = 0;
                            JavaCameraView.params.setZoom(touch_zoom);
                            JavaCameraView.mCamera.setParameters(JavaCameraView.params);
                        }
                    }
                    touch_interval_X = abs(event.getX(0) - event.getX(1));
                    touch_interval_Y = abs(event.getY(0) - event.getY(1));
                }
                break;
            case MotionEvent.ACTION_POINTER_DOWN: // 여러개 터치했을 때
                break;
            case MotionEvent.ACTION_UP: // 터치 뗐을 때
                break;
        }
        return super.onTouchEvent(event);
    }

    final Camera.ShutterCallback shutterCallback = () -> { };

    final Camera.PictureCallback rawCallback = (data, camera) -> { };

    public static int calculatePreviewOrientation(Camera.CameraInfo info, int rotation) {
        int degrees = 0;
        int result;
        if (info.facing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
            result = (degrees + degrees) % 360;
            result = (360 - result) % 360;  // compensate the mirror
        } else {  // back-facing
            result = (info.orientation - degrees + 360) % 360;
        }

        return result;
    }

    final Camera.PictureCallback jpegCallback = new Camera.PictureCallback() {
        public void onPictureTaken(byte[] data, Camera camera) {

            //이미지의 너비와 높이 결정
            int w = camera.getParameters().getPictureSize().width;
            int h = camera.getParameters().getPictureSize().height;
            int mDisplayOrientation = activity.getWindowManager().getDefaultDisplay().getRotation();
            int orientation = calculatePreviewOrientation(mCameraInfo, mDisplayOrientation);
            camera.setDisplayOrientation(orientation);
            //byte array를 bitmap으로 변환
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inPreferredConfig = Bitmap.Config.ARGB_8888;
            Bitmap bitmap = BitmapFactory.decodeByteArray(data, 0, data.length, options);

            //이미지를 디바이스 방향으로 회전
            Matrix matrix = new Matrix();
            if (JavaCameraView.lensId == 99) {
                switch (degreePhone) {
                    case 0:
                        matrix.postRotate(90);
                        break;
                    case 90:
                        matrix.postRotate(0);
                        break;
                    case 180:
                        matrix.postRotate(270);
                        break;
                    case 270:
                        matrix.postRotate(180);
                        break;
                }
            } else {
                switch (degreePhone) {
                    case 0:
                        matrix.postRotate(270);
                        break;
                    case 90:
                        matrix.postRotate(0);
                        break;
                    case 180:
                        matrix.postRotate(90);
                        break;
                    case 270:
                        matrix.postRotate(180);
                        break;
                }
            }
            bitmap = Bitmap.createBitmap(bitmap, 0, 0, w, h, matrix, true);

            //bitmap을 byte array로 변환
            ByteArrayOutputStream stream = new ByteArrayOutputStream();
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, stream);
            byte[] currentData = stream.toByteArray();

            //파일로 저장
            new SaveImageTask().execute(currentData);
        }
    };

    private class SaveImageTask extends AsyncTask<byte[], Void, Void> {

        // Problem ignored
        @SuppressLint({"WrongThread", "MissingPermission", "NewApi"})
        @Override
        protected Void doInBackground(byte[]... data) {
            FileOutputStream outStream = null;

            double latitude = gpsTracker.getLatitude();
            double longitude = gpsTracker.getLongitude();
            String address = getCurrentAddress(latitude, longitude);
            address = address.replace("\n", "");
            Log.d("@@@@@@@@@@123123", address);

            try {
                File path = new File("/storage/emulated/0/Pictures/CaMelShot");
                if (!path.exists()) {
                    path.mkdirs();
                }

                long now = System.currentTimeMillis();
                Date date = new Date(now);
                SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd-hhmmss");
                String getTime = dateFormat.format(date);

                address = address + "_" + getTime;

                String fileName = address+".jpg";
                Log.d("@@@@@@@@jpg", fileName);
                File outputFile = new File(path, fileName);

                outStream = new FileOutputStream(outputFile);
                outStream.write(data[0]);
                outStream.flush();
                outStream.close();

                JavaCameraView.mCamera.startPreview();
                // 갤러리에 반영
                Intent mediaScanIntent = new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE);
                mediaScanIntent.setData(Uri.fromFile(outputFile));
                sendBroadcast(mediaScanIntent);
                try {
                    m_CameraView.surfaceCreated(mHolder);
                    //
                } catch (Exception ignored) { }

                try {
                    ExifInterface exif = new ExifInterface("/storage/emulated/0/Pictures/CaMelShot/" + fileName);

                    exif.saveAttributes();
                } catch (IOException e) {
                    e.printStackTrace();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
            return null;
        }
    }

    public String getCurrentAddress( double latitude, double longitude) {

        //지오코더... GPS를 주소로 변환
        Geocoder geocoder = new Geocoder(this, Locale.getDefault());

        List<Address> addresses;
        try {

            addresses = geocoder.getFromLocation(
                    latitude,
                    longitude,
                    7);
            Log.d("@@@@@@@@@@@@", String.valueOf(addresses));
        } catch (IOException ioException) {
            //네트워크 문제
            Toast.makeText(this, "지오코더 서비스 사용불가", Toast.LENGTH_LONG).show();
            return "지오코더 서비스 사용불가";
        } catch (IllegalArgumentException illegalArgumentException) {
            Toast.makeText(this, "잘못된 GPS 좌표", Toast.LENGTH_LONG).show();
            return "잘못된 GPS 좌표";

        }



        if (addresses == null || addresses.size() == 0) {
            Toast.makeText(this, "주소 미발견", Toast.LENGTH_LONG).show();
            return "주소 미발견";

        }

        Address address = addresses.get(0);
        return address.getAddressLine(0).toString()+"\n";

    }


    //여기부터는 GPS 활성화를 위한 메소드들
    private void showDialogForLocationServiceSetting() {

        AlertDialog.Builder builder = new AlertDialog.Builder(this);
        builder.setTitle("위치 서비스 비활성화");
        builder.setMessage("앱을 사용하기 위해서는 위치 서비스가 필요합니다.\n"
                + "위치 설정을 수정하실래요?");
        builder.setCancelable(true);
        builder.setPositiveButton("설정", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int id) {
                Intent callGPSSettingIntent
                        = new Intent(android.provider.Settings.ACTION_LOCATION_SOURCE_SETTINGS);
                startActivityForResult(callGPSSettingIntent, GPS_ENABLE_REQUEST_CODE);
            }
        });
        builder.setNegativeButton("취소", new DialogInterface.OnClickListener() {
            @Override
            public void onClick(DialogInterface dialog, int id) {
                dialog.cancel();
            }
        });
        builder.create().show();
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);

        switch (requestCode) {

            case GPS_ENABLE_REQUEST_CODE:

                //사용자가 GPS 활성 시켰는지 검사
                if (checkLocationServicesStatus()) {
                    if (checkLocationServicesStatus()) {

                        Log.d("@@@", "onActivityResult : GPS 활성화 되있음");
                        return;
                    }
                }

                break;
        }
    }

    public boolean checkLocationServicesStatus() {
        LocationManager locationManager = (LocationManager) getSystemService(LOCATION_SERVICE);

        return locationManager.isProviderEnabled(LocationManager.GPS_PROVIDER)
                || locationManager.isProviderEnabled(LocationManager.NETWORK_PROVIDER);
    }

    public void flashOnOff() {
        flash_count++;
        JavaCameraView.params = JavaCameraView.mCamera.getParameters();
        if (flash_count > 1)
            flash_count = 0;
    }

    public void switchCamera() {
        switch (JavaCameraView.lensId) {
            case 99: {
                JavaCameraView.lensId = 98;
                break;
            }
            case 98: {
                JavaCameraView.lensId = 99;
                break;
            }
        }
        m_CameraView.surfaceChanged(mHolder, 0, 0, 0);
    }

    public void thumnail() {
        String imageFname;
        try {
            imageFiles = new File(Environment.getExternalStorageDirectory()
                    .getAbsolutePath() + "/Pictures/CaMelShot").listFiles();
            imageFname = imageFiles != null ? imageFiles[imageFiles.length - 1].toString() : null;
            btnThumnail.setScaleType(ImageView.ScaleType.CENTER_INSIDE);
            //Glide.with(this).load(Drawable.createFromPath(imageFname)).into(btnThumnail);
            Glide.with(this).load(Drawable.createFromPath(imageFname)).apply(new RequestOptions().circleCrop()).into(btnThumnail);

            //btnThumnail.setImageDrawable(Drawable.createFromPath(imageFname));
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    public void lineColor() {
        switch (colorSelect) {
            case 0:
                colorSelect = 1;
                break;
            case 1:
                colorSelect = 2;
                break;
            case 2:
                colorSelect = 3;
                break;
            case 3:
                colorSelect = 0;
                break;
        }
        camDraw();
    }
}