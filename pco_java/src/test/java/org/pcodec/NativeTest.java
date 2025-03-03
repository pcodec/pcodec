import org.junit.Test;
import org.pcodec.Native;
import static org.junit.Assert.*;

public class NativeTest {
    @Test
    public void testNativeOps() {
        long[] src = {1, 2, 3};
        byte[] compressed = Native.simpler_compress_i64(src);
        System.out.println("compressed down to " + compressed.length);
    }
}
