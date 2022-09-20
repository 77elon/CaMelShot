package com.dhnns.tflite;

public class NativeLib {

    // Used to load the 'tflite' library on application startup.
    static {
        System.loadLibrary("tflite");
    }

    /**
     * A native method that is implemented by the 'tflite' native library,
     * which is packaged with this application.
     */
    public native String stringFromJNI();
}