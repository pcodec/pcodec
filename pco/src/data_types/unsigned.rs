use std::mem::MaybeUninit;

use crate::compression_intermediates::Bid;
use crate::constants::Bitlen;
use crate::data_types::{Latent, LatentPriv, Number, NumberPriv};
use crate::describers::LatentDescriber;
use crate::dyn_slices::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{ChunkMeta, Mode};
use crate::mode::{classic, dict, int_mult};
use crate::{describers, ChunkConfig, ModeSpec};

fn classic_bid<T: Number>() -> Bid<T> {
  Bid {
    mode: Mode::Classic,
    bits_saved_per_num: 0.0,
    split_fn: Box::new(|nums| classic::split_latents(nums)),
  }
}

fn int_mult_bid<T: Number>(base: T::L) -> Bid<T> {
  Bid {
    mode: Mode::int_mult(base),
    bits_saved_per_num: 0.0,
    split_fn: Box::new(move |nums| int_mult::split_latents(nums, base)),
  }
}

pub fn choose_mode_bids<T: Number>(nums: &[T], config: &ChunkConfig) -> PcoResult<Vec<Bid<T>>> {
  let bids = match config.mode_spec {
    ModeSpec::Auto => match int_mult::choose_base(nums) {
      // int mult is only bid when it's already known to be worthwhile, so we
      // don't offer classic alongside it
      Some(base) => vec![int_mult_bid(base)],
      None => vec![classic_bid()],
    },
    ModeSpec::Classic => vec![classic_bid()],
    ModeSpec::TryFloatMult(_) | ModeSpec::TryFloatQuant(_) => {
      return Err(PcoError::invalid_argument(
        "unable to use float mode for ints",
      ))
    }
    ModeSpec::TryIntMult(base_u64) => vec![int_mult_bid(T::L::from_u64(base_u64))],
    ModeSpec::TryDict => vec![dict::compute_bid(nums)],
  };

  Ok(bids)
}

pub fn join_latents<T: Number>(
  mode: &Mode,
  primary: DynLatentSlice,
  secondary: Option<DynLatentSlice>,
  dst: &mut [MaybeUninit<T>],
) -> PcoResult<()> {
  match mode {
    Mode::Classic => classic::join_latents(primary, dst),
    Mode::Dict(dict) => dict::join_latents(dict, primary, dst),
    Mode::IntMult(base) => int_mult::join_latents(*base, primary, secondary, dst),
    Mode::FloatMult(_) | Mode::FloatQuant(_) => {
      unreachable!("impossible mode for unsigned ints")
    }
  }
}

pub fn mode_is_valid<L: Latent>(mode: &Mode) -> bool {
  match mode {
    Mode::Classic | Mode::Dict(_) => true,
    Mode::FloatMult(_) | Mode::FloatQuant(_) => false,
    Mode::IntMult(base) => *base.downcast_ref::<L>().unwrap() > L::ZERO,
  }
}

macro_rules! impl_latent {
  ($t: ty, $conv: ty) => {
    impl LatentPriv for $t {
      const ZERO: Self = 0;
      const ONE: Self = 1;
      const MID: Self = 1 << (Self::BITS - 1);
      const MAX: Self = Self::MAX;
      const BITS: Bitlen = Self::BITS as Bitlen;

      type Conv = $conv;

      #[inline]
      fn from_u32(x: u32) -> Self {
        x as Self
      }

      #[inline]
      fn from_u64(x: u64) -> Self {
        x as Self
      }

      #[inline]
      fn leading_zeros(self) -> Bitlen {
        self.leading_zeros() as Bitlen
      }

      #[inline]
      fn to_u64(self) -> u64 {
        self as u64
      }

      #[inline]
      fn from_conv(x: Self::Conv) -> Self {
        x as Self
      }

      #[inline]
      fn to_conv(self) -> Self::Conv {
        self as Self::Conv
      }

      #[inline]
      fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
      }

      #[inline]
      fn wrapping_mul(self, other: Self) -> Self {
        self.wrapping_mul(other)
      }

      #[inline]
      fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
      }
    }
  };
}

impl_latent!(u8, i16);
impl_latent!(u16, i32);
impl_latent!(u32, i64);
// 64-bit convolutions can't safely be done in any efficient type without risk
// of overflow, so this i64 is a misnomer; we have runtime checks to prevent
// attempting this
impl_latent!(u64, i64);

macro_rules! impl_unsigned_number {
  ($t: ty, $header_byte: expr) => {
    impl NumberPriv for $t {
      const NUMBER_TYPE_BYTE: u8 = $header_byte;

      type L = Self;

      fn mode_is_valid(mode: &Mode) -> bool {
        mode_is_valid::<Self::L>(mode)
      }
      fn choose_mode_bids(nums: &[Self], config: &ChunkConfig) -> PcoResult<Vec<Bid<Self>>> {
        choose_mode_bids(nums, config)
      }

      #[inline]
      fn from_latent_ordered(l: Self::L) -> Self {
        l
      }
      #[inline]
      fn to_latent_ordered(self) -> Self::L {
        self
      }
      fn join_latents(
        mode: &Mode,
        primary: DynLatentSlice,
        secondary: Option<DynLatentSlice>,
        dst: &mut [MaybeUninit<Self>],
      ) -> PcoResult<()> {
        join_latents(mode, primary, secondary, dst)
      }
    }

    impl Number for $t {
      fn get_latent_describers(meta: &ChunkMeta) -> PerLatentVar<LatentDescriber> {
        describers::match_classic_mode::<Self>(meta, "")
          .or_else(|| describers::match_int_modes::<Self>(meta, false))
          .expect("invalid mode for unsigned type")
      }
    }
  };
}

impl_unsigned_number!(u32, 1);
impl_unsigned_number!(u64, 2);
impl_unsigned_number!(u16, 7);
impl_unsigned_number!(u8, 10);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::metadata::{DynLatents, Mode};

  #[test]
  fn test_mode_validation() {
    // CLASSIC
    assert!(u32::mode_is_valid(&Mode::Classic));

    // DICT
    assert!(u32::mode_is_valid(&Mode::Dict(
      DynLatents::new(vec![1_u32, 3])
    )));

    // INT MULT
    for base in [1_u32, 77, u32::MAX] {
      assert!(u32::mode_is_valid(&Mode::int_mult(base)))
    }
    assert!(!u32::mode_is_valid(&Mode::int_mult(0_u32)));

    // FLOAT
    assert!(!u32::mode_is_valid(&Mode::FloatQuant(3)));
  }
}
