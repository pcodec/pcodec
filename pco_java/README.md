# Pcodec JVM API

Pcodec is a codec for numerical sequences. Example usage:

```java
import io.github.pcodec.ChunkConfig;
import io.github.pcodec.NumArray;
import io.github.pcodec.Standalone;

int[] src = { 1, 2, 3 };
NumArray numArray = NumArray.i32Array(src);
byte[] compressed = Standalone.simple_compress(numArray, new ChunkConfig());
// simple_decompress returns Optional.empty when there is no data,
// but in this example we know there is some.
NumArray recovered = Standalone.simple_decompress(compressed).get();
assertArrayEquals(src, recovered.as_i32_array());
```

For pcodec's uses, design, and benchmarks, [see the main repo](https://github.com/pcodec/pcodec).

Since Java does not natively support all data types, we transmute to the most appropriate Java integer type when necessary.
For instance, 32-bit unsigned integers are decompressed as `int`s, and 16-bit floats are decompressed as `short`s.
