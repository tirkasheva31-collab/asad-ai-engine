package com.asad.ai;

public final class NativeAI {
    static {
        System.loadLibrary("asad-ai");
    }

    private NativeAI() {}

    public static native boolean nativeLoadModel(String absoluteModelPath);
    public static native String nativeGenerate(String prompt, int maxTokens);
    public static native void nativeFree();
}
