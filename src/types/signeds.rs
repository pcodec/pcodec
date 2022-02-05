use std::cmp::Ordering;
use std::convert::TryInto;

use crate::types::NumberLike;
use crate::errors::QCompressResult;

macro_rules! impl_signed_number {
  ($t: ty, $unsigned: ty, $header_byte: expr) => {
    impl NumberLike for $t {
      const HEADER_BYTE: u8 = $header_byte;
      const PHYSICAL_BITS: usize = Self::BITS as usize;

      type Signed = Self;
      type Unsigned = $unsigned;

      fn to_signed(self) -> Self::Signed {
        self
      }

      fn from_signed(signed: Self::Signed) -> Self {
        signed
      }

      fn to_unsigned(self) -> Self::Unsigned {
        self.wrapping_sub(Self::MIN) as $unsigned
      }

      fn from_unsigned(off: Self::Unsigned) -> Self {
        Self::MIN.wrapping_add(off as $t)
      }

      fn num_eq(&self, other: &Self) -> bool {
        self.eq(other)
      }

      fn num_cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
      }

      fn to_bytes(self) -> Vec<u8> {
        self.to_be_bytes().to_vec()
      }

      fn from_bytes(bytes: Vec<u8>) -> QCompressResult<Self> {
        Ok(Self::from_be_bytes(bytes.try_into().unwrap()))
      }
    }
  }
}

impl_signed_number!(i8, u8, 10);
impl_signed_number!(i32, u32, 3);
impl_signed_number!(i64, u64, 1);
impl_signed_number!(i128, u128, 11);
