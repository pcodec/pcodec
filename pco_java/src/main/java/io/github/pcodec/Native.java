package io.github.pcodec;

import io.github.pcodec.NumArray;
import io.questdb.jar.jni.JarJniLoader;

public class Native {
    static {
        JarJniLoader.loadLib(Native.class, "/io/github/pcodec", "pco_java");
    }

    public static native byte[] simpler_compress_i64(NumArray src, int level);
}
