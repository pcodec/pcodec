use crate::constants::Bitlen;
use crate::data_types::{NumberLike, UnsignedLike};

macro_rules! impl_unsigned {
  ($t: ty, $float: ty, $signed: ty) => {
    impl UnsignedLike for $t {
      const ZERO: Self = 0;
      const ONE: Self = 1;
      const MID: Self = 1 << (Self::BITS - 1);
      const MAX: Self = Self::MAX;
      const BITS: Bitlen = Self::BITS as Bitlen;

      type Float = $float;

      #[inline]
      fn from_word(word: usize) -> Self {
        word as Self
      }

      #[inline]
      fn leading_zeros(self) -> Bitlen {
        self.leading_zeros() as Bitlen
      }

      #[inline]
      fn rshift_word(self, shift: Bitlen) -> usize {
        (self >> shift) as usize
      }

      #[inline]
      fn lshift_word(self, shift: Bitlen) -> usize {
        (self as usize) << shift
      }

      #[inline]
      fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
      }

      #[inline]
      fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
      }

      #[inline]
      fn to_int_float(self) -> Self::Float {
        <$signed>::from_unsigned(self) as Self::Float
      }

      #[inline]
      fn from_int_float(float: Self::Float) -> Self {
        (float as $signed).to_unsigned()
      }

      #[inline]
      fn to_float_bits(self) -> Self::Float {
        Self::Float::from_bits(self)
      }

      #[inline]
      fn from_float_bits(float: Self::Float) -> Self {
        float.to_bits()
      }
    }
  };
}

impl_unsigned!(u32, f32, i32);
impl_unsigned!(u64, f64, i64);

macro_rules! impl_unsigned_number {
  ($t: ty, $signed: ty, $float: ty, $header_byte: expr) => {
    impl NumberLike for $t {
      const HEADER_BYTE: u8 = $header_byte;
      const PHYSICAL_BITS: usize = Self::BITS as usize;

      type Unsigned = Self;

      #[inline]
      fn to_unsigned(self) -> Self::Unsigned {
        self
      }

      #[inline]
      fn from_unsigned(off: Self::Unsigned) -> Self {
        off
      }

      #[inline]
      fn transmute_to_unsigned_slice(slice: &mut [Self]) -> &mut [Self::Unsigned] {
        slice
      }

      #[inline]
      fn transmute_to_unsigned(self) -> Self::Unsigned {
        self
      }
    }
  };
}

impl_unsigned_number!(u32, i32, f32, 4);
impl_unsigned_number!(u64, i64, f64, 2);
