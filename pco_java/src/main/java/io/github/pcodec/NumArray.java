package io.github.pcodec;

public class NumArray {
    public Object nums;
    private int dtypeByte;

    private NumArray(Object nums, int dtypeByte) {
        this.nums = nums;
        this.dtypeByte = dtypeByte;
    }

    private NumArray(Object nums, NumberType dtype) {
        this.nums = nums;
        this.dtypeByte = dtype.dtypeByte;
    }

    public NumberType dtype() {
        return NumberType.fromDtypeByte(dtypeByte);
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

    public short[] as_i16_array() throws IllegalStateException {
        if (dtypeByte == NumberType.I16.dtypeByte) {
            return (short[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to I16");
    }

    public int[] as_i32_array() throws IllegalStateException {
        if (dtypeByte == NumberType.I32.dtypeByte) {
            return (int[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to I32");
    }

    public long[] as_i64_array() throws IllegalStateException {
        if (dtypeByte == NumberType.I64.dtypeByte) {
            return (long[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to I64");
    }

    public short[] as_u16_array() throws IllegalStateException {
        if (dtypeByte == NumberType.U16.dtypeByte) {
            return (short[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to U16");
    }

    public int[] as_u32_array() throws IllegalStateException {
        if (dtypeByte == NumberType.U32.dtypeByte) {
            return (int[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to U32");
    }

    public long[] as_u64_array() throws IllegalStateException {
        if (dtypeByte == NumberType.U64.dtypeByte) {
            return (long[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to U64");
    }

    public short[] as_f16_array() throws IllegalStateException {
        if (dtypeByte == NumberType.F16.dtypeByte) {
            return (short[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to F16");
    }

    public float[] as_f32_array() throws IllegalStateException {
        if (dtypeByte == NumberType.F32.dtypeByte) {
            return (float[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to F32");
    }

    public double[] as_f64_array() throws IllegalStateException {
        if (dtypeByte == NumberType.F64.dtypeByte) {
            return (double[]) this.nums;
        }
        throw new IllegalStateException("Cannot cast pco NumArray of " + this.dtype() + " to F64");
    }
}
