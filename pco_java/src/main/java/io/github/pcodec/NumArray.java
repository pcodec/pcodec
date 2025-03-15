package io.github.pcodec;

/**
 * Represents an array of numbers and the corresponding data type.
 *
 * The underlying representation is e.g. a long[] for the i64 data type. For
 * the data types where Java has no built in data type, we use integers of the
 * corresponding precision. For instance, an f16 array is represented as a
 * short[]. Under the hood, Pco transmutes data into the correct data type.
 */
public class NumArray {
    public final Object nums;
    private final byte numberTypeByte;

    private NumArray(Object nums, byte numberTypeByte) {
        this.nums = nums;
        this.numberTypeByte = numberTypeByte;
    }

    private NumArray(Object nums, NumberType numberType) {
        this.nums = nums;
        this.numberTypeByte = numberType.byte_;
    }

    public NumberType numberType() {
        return NumberType.fromByte(numberTypeByte);
    }

    public static NumArray i16Array(short[] nums) {
        return new NumArray(nums, NumberType.I16);
    }

    public static NumArray i32Array(int[] nums) {
        return new NumArray(nums, NumberType.I32);
    }

    public static NumArray i64Array(long[] nums) {
        return new NumArray(nums, NumberType.I64);
    }

    public static NumArray u16Array(short[] nums) {
        return new NumArray(nums, NumberType.U16);
    }

    public static NumArray u32Array(int[] nums) {
        return new NumArray(nums, NumberType.U32);
    }

    public static NumArray u64Array(long[] nums) {
        return new NumArray(nums, NumberType.U64);
    }

    public static NumArray f16Array(short[] nums) {
        return new NumArray(nums, NumberType.F16);
    }

    public static NumArray f32Array(float[] nums) {
        return new NumArray(nums, NumberType.F32);
    }

    public static NumArray f64Array(double[] nums) {
        return new NumArray(nums, NumberType.F64);
    }

    private IllegalStateException invalidNumberType(NumberType numberType) {
        return new IllegalStateException("Cannot cast pco NumArray of " + this.numberType() + " to " + numberType);
    }

    public short[] as_i16_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.I16.byte_) {
            return (short[]) this.nums;
        }
        throw invalidNumberType(NumberType.I16);
    }

    public int[] as_i32_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.I32.byte_) {
            return (int[]) this.nums;
        }
        throw invalidNumberType(NumberType.I32);
    }

    public long[] as_i64_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.I64.byte_) {
            return (long[]) this.nums;
        }
        throw invalidNumberType(NumberType.I64);
    }

    public short[] as_u16_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.U16.byte_) {
            return (short[]) this.nums;
        }
        throw invalidNumberType(NumberType.U16);
    }

    public int[] as_u32_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.U32.byte_) {
            return (int[]) this.nums;
        }
        throw invalidNumberType(NumberType.U32);
    }

    public long[] as_u64_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.U64.byte_) {
            return (long[]) this.nums;
        }
        throw invalidNumberType(NumberType.U64);
    }

    public short[] as_f16_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.F16.byte_) {
            return (short[]) this.nums;
        }
        throw invalidNumberType(NumberType.F16);
    }

    public float[] as_f32_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.F32.byte_) {
            return (float[]) this.nums;
        }
        throw invalidNumberType(NumberType.F32);
    }

    public double[] as_f64_array() throws IllegalStateException {
        if (numberTypeByte == NumberType.F64.byte_) {
            return (double[]) this.nums;
        }
        throw invalidNumberType(NumberType.F64);
    }
}
