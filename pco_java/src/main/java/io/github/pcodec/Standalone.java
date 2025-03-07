package io.github.pcodec;

import java.util.Optional;

import io.questdb.jar.jni.JarJniLoader;

public class Standalone {
    static {
        JarJniLoader.loadLib(Standalone.class, "/io/github/pcodec", "pco_java");
    }

    public static native byte[] simpler_compress(NumArray src, int level);

    public static native Optional<NumArray> simple_decompress(byte[] src);
}
