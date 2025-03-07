package io.github.pcodec;

/**
 * Specifies which type of Pco-supported number is being used.
 *
 * Each number type has a corresponding unique byte.
 */
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

    public final int byte_;

    NumberType(int byte_) {
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
