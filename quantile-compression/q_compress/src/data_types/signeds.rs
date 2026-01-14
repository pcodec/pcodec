use std::convert::TryInto;

use crate::data_types::{NumberLike, SignedLike};
use crate::errors::QCompressResult;

macro_rules! impl_signed {
  ($t: ty, $unsigned: ty, $header_byte: expr) => {
    impl SignedLike for $t {
      const ZERO: Self = 0;

      #[inline]
      fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
      }

      #[inline]
      fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
      }
    }

    impl NumberLike for $t {
      const HEADER_BYTE: u8 = $header_byte;
      const PHYSICAL_BITS: usize = Self::BITS as usize;

      type Signed = Self;
      type Unsigned = $unsigned;

      #[inline]
      fn to_signed(self) -> Self::Signed {
        self
      }

      #[inline]
      fn from_signed(signed: Self::Signed) -> Self {
        signed
      }

      #[inline]
      fn to_unsigned(self) -> Self::Unsigned {
        self.wrapping_sub(Self::MIN) as $unsigned
      }

      #[inline]
      fn from_unsigned(off: Self::Unsigned) -> Self {
        Self::MIN.wrapping_add(off as $t)
      }

      fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
      }

      fn from_bytes(bytes: &[u8]) -> QCompressResult<Self> {
        Ok(Self::from_be_bytes(
          bytes.try_into().unwrap(),
        ))
      }
    }
  };
}

impl_signed!(i16, u16, 13);
impl_signed!(i32, u32, 3);
impl_signed!(i64, u64, 1);
#[cfg(feature = "timestamps_96")]
impl_signed!(i128, u128, 10);
