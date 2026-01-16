use std::fmt::Display;
use std::hash::Hash;
use std::ops::*;

use crate::constants::Bitlen;
use crate::data_types::signed::Signed;
use crate::data_types::Number;

pub trait LatentPriv:
  Add<Output = Self>
  + AddAssign
  + BitAnd<Output = Self>
  + BitOr<Output = Self>
  + BitAndAssign
  + BitOrAssign
  + Display
  + Div<Output = Self>
  + DivAssign
  + Hash
  + Mul<Output = Self>
  + MulAssign
  + Number<L = Self>
  + Ord
  + PartialOrd
  + Rem<Output = Self>
  + RemAssign
  + Send
  + Sync
  + Shl<Bitlen, Output = Self>
  + Shr<Bitlen, Output = Self>
  + Sub<Output = Self>
{
  const ZERO: Self;
  const ONE: Self;
  const MID: Self;
  const MAX: Self;
  const BITS: Bitlen;

  type Conv: Signed;

  /// Converts a `u32` into this type. Panics if the conversion is
  /// impossible.
  fn from_u32(x: u32) -> Self;

  /// Converts a `u64` into this type. Panics if the conversion is
  /// impossible.
  fn from_u64(x: u64) -> Self;

  fn leading_zeros(self) -> Bitlen;

  /// Converts the latent to a `u64`, truncating higher bits if necessary.
  fn to_u64(self) -> u64;

  fn from_conv(x: Self::Conv) -> Self;
  fn to_conv(self) -> Self::Conv;

  fn wrapping_add(self, other: Self) -> Self;
  fn wrapping_sub(self, other: Self) -> Self;

  fn toggle_center(self) -> Self {
    self.wrapping_add(Self::MID)
  }
}
