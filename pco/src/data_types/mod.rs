use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::{
  Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign, Neg,
  Rem, RemAssign, Shl, Shr, Sub, SubAssign,
};

pub use dynamic::{LatentType, NumberType};
pub use split_latents::SplitLatents;

use crate::constants::Bitlen;
use crate::describers::LatentDescriber;
use crate::errors::PcoResult;
use crate::metadata::dyn_latents::DynLatents;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{ChunkMeta, Mode};
use crate::ChunkConfig;

mod dynamic;
mod floats;
mod signeds;
mod split_latents;
mod unsigneds;

pub(crate) type ModeAndLatents = (Mode, SplitLatents);

/// This is used internally for compressing and decompressing with
/// float modes.
pub(crate) trait Float:
  Add<Output = Self>
  + AddAssign
  + Copy
  + Debug
  + Display
  + Mul<Output = Self>
  + Neg<Output = Self>
  + Number
  + PartialOrd
  + RemAssign
  + Send
  + Sync
  + Sub<Output = Self>
  + SubAssign
  + Div<Output = Self>
{
  /// Number of bits that aren't used for exponent or sign.
  /// E.g. for f32 this should be 23.
  const PRECISION_BITS: Bitlen;
  const ZERO: Self;
  const MAX_FOR_SAMPLING: Self;

  fn abs(self) -> Self;
  fn inv(self) -> Self;
  fn round(self) -> Self;
  /// This only needs to cover a small range (from 2^-BITS to 2^BITS) and might
  /// not be valid outside of it.
  fn exp2(power: i32) -> Self;
  fn from_f64(x: f64) -> Self;
  fn to_f64(self) -> f64;
  fn is_normal(self) -> bool;
  fn is_sign_positive_(&self) -> bool;
  /// Returns the float's exponent. For instance, for f32 this should be
  /// between -127 and +126.
  fn exponent(&self) -> i32;
  fn trailing_zeros(&self) -> u32;
  fn max(a: Self, b: Self) -> Self;
  fn min(a: Self, b: Self) -> Self;

  /// This should use something like [`f32::to_bits()`]
  fn to_latent_bits(self) -> Self::L;
  /// This should surjectively map the latent to the set of integers in its
  /// floating point type. E.g. 3.0, Inf, and NaN are int floats, but 3.5 is
  /// not.
  fn int_float_from_latent(l: Self::L) -> Self;
  /// This should be the inverse of `int_float_from_latent`.
  fn int_float_to_latent(self) -> Self::L;
  /// This should map from e.g. 7_u32 -> 7.0_f32
  fn from_latent_numerical(l: Self::L) -> Self;
}

// TODO in 1.0 seal Latent and Number
/// **unstable API** Trait for data types that behave like unsigned integers.
///
/// This is used extensively in `pco` to guarantee that bitwise
/// operations like `>>` and `|=` are available and that certain properties
/// hold.
/// Under the hood, when numbers are encoded or decoded, they go through their
/// corresponding `Latent` representation.
/// Metadata stores numbers as their latent representations.
pub trait Latent:
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

  /// Converts a `usize` into this type. Panics if the conversion is
  /// impossible.
  fn from_u64(x: u64) -> Self;

  fn leading_zeros(self) -> Bitlen;

  /// Converts the latent to a usize, truncating higher bits if necessary.
  fn to_u64(self) -> u64;

  fn wrapping_add(self, other: Self) -> Self;
  fn wrapping_sub(self, other: Self) -> Self;

  fn toggle_center(self) -> Self {
    self.wrapping_add(Self::MID)
  }
}

/// **unstable API** Trait for data types supported for compression/decompression.
///
/// If you have a new data type you would like to add to the library or,
/// these are the questions you need to answer:
/// * What is the corresponding latent type? This is probably the
///   smallest unsigned integer with enough bits to represent the number.
/// * How can I convert to this latent representation and back
///   in *a way that preserves ordering*? For instance, transmuting `f32` to `u32`
///   wouldn't preserve ordering and would cause pco to fail. In this example,
///   one needs to flip the sign bit and, if negative, the rest of the bits.
///
/// Custom data types (defined outside of pco) are not currently supported.
pub trait Number: Copy + Debug + Display + Default + PartialEq + Send + Sync + 'static {
  /// A number from 1-255 that corresponds to the number's data type.
  ///
  /// Each `Number` implementation should have a different `NUMBER_TYPE_BYTE`.
  /// This byte gets written into the file's header during compression, and
  /// if the wrong header byte shows up during decompression, the decompressor
  /// will return an error.
  ///
  /// To choose a header byte for a new data type, review all header bytes in
  /// the library and pick an unused one. For instance, as of writing, bytes
  /// 1 through 6 are used, so 7 would be a good choice for another
  /// `pco` data type implementation.
  const NUMBER_TYPE_BYTE: u8;

  /// The latent this type can convert between to do bitwise logic and such.
  type L: Latent;

  fn get_latent_describers(meta: &ChunkMeta) -> PerLatentVar<LatentDescriber>;

  fn mode_is_valid(mode: Mode) -> bool;
  /// Breaks the numbers into latent variables for better compression.
  ///
  /// Returns
  /// * mode: the [`Mode`] that will be stored alongside the data
  ///   for decompression
  /// * latents: a primary and optionally secondary latent variable, each of
  ///   which contains a latent per num in `nums`. Primary must be of the same
  ///   latent type as T.
  fn choose_mode_and_split_latents(
    nums: &[Self],
    config: &ChunkConfig,
  ) -> PcoResult<ModeAndLatents>;

  fn from_latent_ordered(l: Self::L) -> Self;
  fn to_latent_ordered(self) -> Self::L;
  fn join_latents(mode: Mode, primary: &mut [Self::L], secondary: Option<&DynLatents>);

  fn transmute_to_latents(slice: &mut [Self]) -> &mut [Self::L];
  fn transmute_to_latent(self) -> Self::L;
}

pub(crate) fn split_latents_classic<T: Number>(nums: &[T]) -> SplitLatents {
  let primary = DynLatents::new(nums.iter().map(|&x| x.to_latent_ordered()).collect()).unwrap();
  SplitLatents {
    primary,
    secondary: None,
  }
}
