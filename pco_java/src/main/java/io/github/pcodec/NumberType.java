package io.github.pcodec;

public enum NumberType {
    I16(8),
    I32(3),
    I64(4),
    F16(9),
    F32(5),
    F64(6),
    U16(7),
    U32(1),
    U64(2);

    public final int dtypeByte;

    NumberType(int dtypeByte) {
        this.dtypeByte = dtypeByte;
    }

    public static NumberType fromDtypeByte(int dtypeByte) {
        for (NumberType dtype : values()) {
            if (dtype.dtypeByte == dtypeByte) {
                return dtype;
            }
        }
        throw new IllegalArgumentException("Invalid dtype byte: " + dtypeByte);
    }
}
