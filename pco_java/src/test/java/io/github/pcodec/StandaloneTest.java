package io.github.pcodec;

import org.junit.Test;
import static org.junit.Assert.*;

public class StandaloneTest {
    @Test
    public void testIntRecovery() {
        int[] src = { 1, 2, 3 };
        NumArray numArray = NumArray.i32Array(src);
        byte[] compressed = Standalone.simple_compress(numArray, new ChunkConfig());
        NumArray recovered = Standalone.simple_decompress(compressed).get();
        assertArrayEquals(src, recovered.as_i32_array());
    }

    @Test
    public void testLongRecovery() {
        long[] src = { 1, 2, 3 };
        NumArray numArray = NumArray.i64Array(src);
        byte[] compressed = Standalone.simple_compress(numArray, new ChunkConfig());
        NumArray recovered = Standalone.simple_decompress(compressed).get();
        assertArrayEquals(src, recovered.as_i64_array());
    }

    @Test
    public void testF16Recovery() {
        short[] src = { 1, 2, 3 };
        NumArray numArray = NumArray.f16Array(src);
        byte[] compressed = Standalone.simple_compress(numArray, new ChunkConfig());
        NumArray recovered = Standalone.simple_decompress(compressed).get();
        assertArrayEquals(src, recovered.as_f16_array());
    }

    @Test
    public void testIllegalArgument() {
        short[] src = { 1, 2, 3 };
        NumArray numArray = NumArray.i16Array(src);
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Standalone.simple_compress(
                            numArray,
                            new ChunkConfig().withCompressionLevel(13));
                });
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Standalone.simple_compress(
                            numArray,
                            new ChunkConfig().withCompressionLevel(-1));
                });
    }
}
