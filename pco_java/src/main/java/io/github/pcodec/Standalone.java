package io.github.pcodec;

import java.util.Optional;

import io.questdb.jar.jni.JarJniLoader;

/**
 * Contains functions and data structures for manipulating Pco standalone files.
 *
 * This format is easy to use and recommended for simple proofs of concept, but
 * the wrapped format might be better for interleaving in other file formats.
 */
public class Standalone {
    static {
        JarJniLoader.loadLib(Standalone.class, "/io/github/pcodec", "pco_java");
    }

    /**
     * Compresses an array of numbers into bytes.
     */
    public static native byte[] simple_compress(NumArray src, ChunkConfig config) throws IllegalArgumentException;

    /**
     * Decompresses bytes into an array of numbers.
     *
     * If the numbers are empty, Pco will be unable to infer the number type, so
     * this will return an empty optional.
     */
    public static native Optional<NumArray> simple_decompress(byte[] src) throws RuntimeException;
}
