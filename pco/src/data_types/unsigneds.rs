use super::ModeAndLatents;
use crate::constants::Bitlen;
use crate::data_types::{split_latents_classic, Latent, Number};
use crate::describers::LatentDescriber;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{ChunkMeta, DynLatent, DynLatents, Mode};
use crate::{describers, int_mult_utils, ChunkConfig, ModeSpec};

pub fn choose_mode_and_split_latents<T: Number>(
  nums: &[T],
  config: &ChunkConfig,
) -> PcoResult<ModeAndLatents> {
  match config.mode_spec {
    ModeSpec::Auto => {
      if let Some(base) = int_mult_utils::choose_base(nums) {
        let mode = Mode::int_mult(base);
        let latents = int_mult_utils::split_latents(nums, base);
        Ok((mode, latents))
      } else {
        Ok((Mode::Classic, split_latents_classic(nums)))
      }
    }

    ModeSpec::Classic => Ok((Mode::Classic, split_latents_classic(nums))),
    ModeSpec::TryFloatMult(_) | ModeSpec::TryFloatQuant(_) => Err(PcoError::invalid_argument(
      "unable to use float mode for ints",
    )),
    ModeSpec::TryIntMult(base_u64) => {
      let base = T::L::from_u64(base_u64);
      let mode = Mode::IntMult(DynLatent::new(base).unwrap());
      let latents = int_mult_utils::split_latents(nums, base);
      Ok((mode, latents))
    }
  }
}

pub fn mode_is_valid<L: Latent>(mode: Mode) -> bool {
  match mode {
    Mode::Classic => true,
    Mode::IntMult(base) => *base.downcast_ref::<L>().unwrap() > L::ZERO,
    _ => false,
  }
}

macro_rules! impl_latent {
  ($t: ty) => {
    impl Latent for $t {
      const ZERO: Self = 0;
      const ONE: Self = 1;
      const MID: Self = 1 << (Self::BITS - 1);
      const MAX: Self = Self::MAX;
      const BITS: Bitlen = Self::BITS as Bitlen;

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
      fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
      }

      #[inline]
      fn wrapping_sub(self, other: Self) -> Self {
        self.wrapping_sub(other)
      }
    }
  };
}

impl_latent!(u16);
impl_latent!(u32);
impl_latent!(u64);

macro_rules! impl_unsigned_number {
  ($t: ty, $header_byte: expr) => {
    impl Number for $t {
      const NUMBER_TYPE_BYTE: u8 = $header_byte;

      type L = Self;

      fn get_latent_describers(meta: &ChunkMeta) -> PerLatentVar<LatentDescriber> {
        describers::match_classic_mode::<Self>(meta, "")
          .or_else(|| describers::match_int_modes::<Self>(meta, false))
          .expect("invalid mode for unsigned type")
      }

      fn mode_is_valid(mode: Mode) -> bool {
        mode_is_valid::<Self::L>(mode)
      }
      fn choose_mode_and_split_latents(
        nums: &[Self],
        config: &ChunkConfig,
      ) -> PcoResult<ModeAndLatents> {
        choose_mode_and_split_latents(nums, config)
      }

      #[inline]
      fn from_latent_ordered(l: Self::L) -> Self {
        l
      }
      #[inline]
      fn to_latent_ordered(self) -> Self::L {
        self
      }
      fn join_latents(mode: Mode, primary: &mut [Self::L], secondary: Option<&DynLatents>) {
        match mode {
          Mode::Classic => (),
          Mode::IntMult(dyn_latent) => {
            let base = *dyn_latent.downcast_ref::<Self::L>().unwrap();
            int_mult_utils::join_latents(base, primary, secondary)
          }
          _ => unreachable!("impossible mode for unsigned ints"),
        }
      }

      fn transmute_to_latents(slice: &mut [Self]) -> &mut [Self::L] {
        slice
      }
      #[inline]
      fn transmute_to_latent(self) -> Self::L {
        self
      }
    }
  };
}

impl_unsigned_number!(u32, 1);
impl_unsigned_number!(u64, 2);
impl_unsigned_number!(u16, 7);

#[cfg(test)]
mod tests {
  use super::*;
  use crate::metadata::Mode;

  #[test]
  fn test_mode_validation() {
    // CLASSIC
    assert!(u32::mode_is_valid(Mode::Classic));

    // INT MULT
    for base in [1_u32, 77, u32::MAX] {
      assert!(u32::mode_is_valid(Mode::int_mult(base)))
    }
    assert!(!u32::mode_is_valid(Mode::int_mult(0_u32)));

    // FLOAT
    assert!(!u32::mode_is_valid(Mode::FloatQuant(3)));
  }
}
