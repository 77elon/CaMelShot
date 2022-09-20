package com.SSRL.CaMelShot;

import static org.opencv.core.CvType.CV_8UC4;

import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.FileNotFoundException;
import java.io.IOException;

public class MainActivity extends AppCompatActivity
{
    static {
        System.loadLibrary("opencv_jni_shared");
        System.loadLibrary("opencv_java4");
        System.loadLibrary("yolov5ncnn");
    }
    private static final int SELECT_IMAGE = 1;
    private ImageView imageView;
    public static Bitmap bitmap = null;
    private static Bitmap yourSelectedImage = null;
    private static Bitmap copySelectedImage = null;
    private static final DetectEdge yolov5ncnn = new DetectEdge();
    //MYJOB
    public static Context mContext;
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        //setContentView(R.layout.main);

        boolean ret_init = yolov5ncnn.Init(getAssets());
        if (!ret_init)
        {
            Log.e("MainActivity", "yolov5ncnn Init failed");
        }

        imageView = findViewById(R.id.imageView);

        Button buttonImage = findViewById(R.id.buttonImage);
        buttonImage.setOnClickListener(arg0 -> {
            Intent i = new Intent(Intent.ACTION_PICK);
            i.setType("image/*");
            startActivityForResult(i, SELECT_IMAGE);
            yourSelectedImage = null;
            imageView.setImageBitmap(null);
        });
        //TODO - Just detection button (신경 X 안써도 되는 것)
        Button buttonDetect = findViewById(R.id.buttonDetect);
        buttonDetect.setOnClickListener(arg0 -> {
            if (yourSelectedImage == null)
                return;
            //MYJOB - 전체 화면 대상 contour 추출 버튼으로 바꿈
            //Mat in = new Mat();
            //Utils.bitmapToMat(yourSelectedImage, in);
            //DetectEdge.DetectMat_tmp(in.getNativeObjAddr(), true);
            //Utils.matToBitmap(in, yourSelectedImage);
            //아래는 기존 비트맵 obj 디텍팅
            DetectEdge.Obj[] objs;
            yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);
            imageView.setImageBitmap(yourSelectedImage);
            objs = DetectEdge.Detect(yourSelectedImage, false);
            showObjects(objs);
        });
        //Detect AND Edge Button
        //TODO - detection & edge button (사용자에게 이미지 받아와서 불러올 함수 onclick(지금은 리스너) -> imageProc으로 바꿔서 호출 바람)
        Button buttonDetectGPU = findViewById(R.id.buttonDetectGPU);
        buttonDetectGPU.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View arg0) {
                if (yourSelectedImage == null) {
                    return;
                }
                yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                int w = yourSelectedImage.getWidth();
                int h = yourSelectedImage.getHeight();
                copySelectedImage = yourSelectedImage;
                //image
                Mat in = new Mat();
                Utils.bitmapToMat(yourSelectedImage, in);
                //빈 공간
                Mat out = Mat.zeros(new Size(w, h), CV_8UC4);
                //객체 검출
                DetectEdge.Obj[] objects = yolov5ncnn.Detect(yourSelectedImage, false);
                if(!objects.equals(null))
                {
                    //이미지와 빈 캠버스, 검출된 객체 input
                    DetectEdge.DetectMat(in.getNativeObjAddr(), out.getNativeObjAddr(), false, objects, CamActivity.colorSelect);
                }
                copySelectedImage = Bitmap.createScaledBitmap(copySelectedImage, 1920, 1080, true);
                Utils.matToBitmap(out, yourSelectedImage);
                imageView.setImageBitmap(yourSelectedImage);
                imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
            }
        });

        Button btnUploadImage = findViewById(R.id.btnUploadImage);
        btnUploadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse("http://121.187.77.18/input.html"));
                startActivity(intent);
            }
        });

        Button btnDownloadImage = findViewById(R.id.btnDownloadImage);
        btnDownloadImage.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse("http://121.187.77.18/output.php"));
                startActivity(intent);
            }
        });

        //MYJOB start camera activity, clicking btn
        ImageButton buttonCam = findViewById(R.id.buttonCam);
        buttonCam.setOnClickListener(view -> {
            Intent intent = new Intent(MainActivity.this, CamActivity.class);
            startActivity(intent);
        });

        //MYJOB set context
        mContext = this;
    }

    public void showObjects(DetectEdge.Obj[] objects)
    {
        if (objects == null)
        {
            imageView.setImageBitmap(bitmap);
            return;
        }

        // draw objects on bitmap
        Bitmap rgba = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        final int[] colors = new int[] {
                Color.rgb( 54,  67, 244),
                Color.rgb( 99,  30, 233),
                Color.rgb(176,  39, 156),
                Color.rgb(183,  58, 103),
                Color.rgb(181,  81,  63),
                Color.rgb(243, 150,  33),
                Color.rgb(244, 169,   3),
                Color.rgb(212, 188,   0),
                Color.rgb(136, 150,   0),
                Color.rgb( 80, 175,  76),
                Color.rgb( 74, 195, 139),
                Color.rgb( 57, 220, 205),
                Color.rgb( 59, 235, 255),
                Color.rgb(  7, 193, 255),
                Color.rgb(  0, 152, 255),
                Color.rgb( 34,  87, 255),
                Color.rgb( 72,  85, 121),
                Color.rgb(158, 158, 158),
                Color.rgb(139, 125,  96)
        };

        Canvas canvas = new Canvas(rgba);

        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(4);

        Paint textbgpaint = new Paint();
        textbgpaint.setColor(Color.WHITE);
        textbgpaint.setStyle(Paint.Style.FILL);

        Paint textpaint = new Paint();
        textpaint.setColor(Color.BLACK);
        textpaint.setTextSize(26);
        textpaint.setTextAlign(Paint.Align.LEFT);

        for (int i = 0; i < objects.length; i++)
        {
            paint.setColor(colors[i % 19]);
            //TODO - 객체 상자 그리기
            canvas.drawRect(objects[i].x, objects[i].y, objects[i].x + objects[i].w, objects[i].y + objects[i].h, paint);

            // draw filled text inside image
            {
                String text = objects[i].label + " = " + String.format("%.1f", objects[i].prob * 100) + "%";

                float text_width = textpaint.measureText(text);
                float text_height = - textpaint.ascent() + textpaint.descent();

                float x = objects[i].x;
                float y = objects[i].y - text_height;
                if (y < 0)
                    y = 0;
                if (x + text_width > rgba.getWidth())
                    x = rgba.getWidth() - text_width;

                canvas.drawRect(x, y, x + text_width, y + text_height, textbgpaint);
                canvas.drawText(text, x, y - textpaint.ascent(), textpaint);
            }
        }

        imageView.setImageBitmap(rgba);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data)
    {
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && null != data) {
            Uri selectedImage = data.getData();
            try
            {
                if (requestCode == SELECT_IMAGE) {
                    bitmap = decodeUri(selectedImage);
                    yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);
                    imageView.setImageBitmap(bitmap);
                    //imageView.setImageBitmap(yourSelectedImage);
                }
            }
            catch (FileNotFoundException e)
            {
                Log.e("MainActivity", "FileNotFoundException");
            }
        }
    }

    private Bitmap decodeUri(Uri selectedImage) throws FileNotFoundException
    {
        // Decode image size
        BitmapFactory.Options o = new BitmapFactory.Options();
        o.inJustDecodeBounds = true;
        BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o);

        // The new size we want to scale to
        final int REQUIRED_SIZE = 640;

        // Find the correct scale value. It should be the power of 2.
        int width_tmp = o.outWidth, height_tmp = o.outHeight;
        int scale = 1;
        while (width_tmp / 2 >= REQUIRED_SIZE
                && height_tmp / 2 >= REQUIRED_SIZE) {
            width_tmp /= 2;
            height_tmp /= 2;
            scale *= 2;
        }

        // Decode with inSampleSize
        BitmapFactory.Options o2 = new BitmapFactory.Options();
        o2.inSampleSize = scale;
        Bitmap bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(selectedImage), null, o2);

        // Rotate according to EXIF
        int rotate = 0;
        try
        {
            ExifInterface exif = new ExifInterface(getContentResolver().openInputStream(selectedImage));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_270:
                    rotate = 270;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    rotate = 180;
                    break;
                case ExifInterface.ORIENTATION_ROTATE_90:
                    rotate = 90;
                    break;
            }
        }
        catch (IOException e)
        {
            Log.e("MainActivity", "ExifInterface IOException");
        }

        Matrix matrix = new Matrix();
        matrix.postRotate(rotate);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    public static void camDraw() {
        if (yourSelectedImage == null) {
            return;
        }
        yourSelectedImage = bitmap.copy(Bitmap.Config.ARGB_8888, true);
        //TODO
        //int w = yourSelectedImage.getWidth();
        //int h = yourSelectedImage.getHeight();
        int ori_w = yourSelectedImage.getWidth();
        int ori_h = yourSelectedImage.getHeight();

        double w, h;
        // 정확한 수정이 필요할 것.
        // 특히 비율 기반으로 출력해주는 Activity의 계산 정확도 부재와 출력 해상도와 입력 해상도가 (비율, 고정) 다름을 인지...
        //1800 * 1370
        if(ori_h > ori_w){
            h = 1920;
            w = ori_w * (1920.0f / ori_h);
        }
        else{
            w = 1920;
            h = ori_h * (1920.0f / ori_w);
        }
        //
        copySelectedImage = yourSelectedImage;
        //image
        Mat in = new Mat();
        Utils.bitmapToMat(copySelectedImage, in);
        Imgproc.resize(in, in, new Size((int)w, (int)h));
        Log.d("mySize1", ori_w + " " + ori_h);
        Log.d("mySize2", w + " " + h);
        Log.d("mySize3", in.cols() + " " + in.rows());
        //빈 공간
        //Mat out = Mat.zeros(in.size(), in.type());
        Mat out = Mat.zeros(new Size((int)w, (int)h), CV_8UC4);
        //객체 검출
        DetectEdge.Obj[] objects = yolov5ncnn.DetectYolo(in.getNativeObjAddr(), false);
        DetectEdge.DetectMat(in.getNativeObjAddr(), out.getNativeObjAddr(), false, objects, CamActivity.colorSelect);

        //Imgproc.resize(out, out, in.size());
        Imgproc.resize(out, out, new Size((int)w,(int)h));
        copySelectedImage = Bitmap.createScaledBitmap(copySelectedImage, (int)w, (int)h, true);
        Utils.matToBitmap(out, copySelectedImage);
        CamActivity.imageView.setImageBitmap(copySelectedImage);
        CamActivity.imageView.setScaleType(ImageView.ScaleType.CENTER);
        CamActivity.imageViewAuto.setImageBitmap(copySelectedImage);
        CamActivity.imageViewAuto.setScaleType(ImageView.ScaleType.CENTER);
    }
}
