package io.github.pcodec;

public class NumArray {
    public Object nums;
    public int dtype;

    private NumArray(Object nums, int dtype) {
        this.nums = nums;
        this.dtype = dtype;
    }

    public static NumArray longArray(long[] nums) {
        return new NumArray(nums, 4);
    }
}
