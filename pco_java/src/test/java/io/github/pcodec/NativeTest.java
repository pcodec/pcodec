import org.junit.Test;
import io.github.pcodec.Native;
import io.github.pcodec.NumArray;
import static org.junit.Assert.*;

public class NativeTest {
    @Test
    public void testNativeOps() {
        long[] src = {1, 2, 3};
        NumArray numArray = NumArray.longArray(src);
        byte[] compressed = Native.simpler_compress_i64(numArray, 8);
        System.out.println("compressed down to " + compressed.length);
    }
}
