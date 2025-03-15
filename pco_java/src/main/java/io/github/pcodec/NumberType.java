package io.github.pcodec;

/**
 * Specifies which type of Pco-supported number is being used.
 *
 * Each number type has a corresponding unique byte.
 */
public enum NumberType {
    I16((byte) 8),
    I32((byte) 3),
    I64((byte) 4),
    F16((byte) 9),
    F32((byte) 5),
    F64((byte) 6),
    U16((byte) 7),
    U32((byte) 1),
    U64((byte) 2);

    public final byte byte_;

    private NumberType(byte byte_) {
        this.byte_ = byte_;
    }

    public static NumberType fromByte(int byte_) {
        for (NumberType numberType : values()) {
            if (numberType.byte_ == byte_) {
                return numberType;
            }
        }
        throw new IllegalArgumentException("Invalid number type byte: " + byte_);
    }
}
