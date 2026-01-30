use crate::constants::Bitlen;
use crate::data_types::{unsigned, ModeAndLatents, Number, NumberPriv};
use crate::describers::LatentDescriber;
use crate::dyn_slices::DynLatentSlice;
use crate::errors::PcoResult;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{ChunkMeta, Mode};
use crate::{describers, ChunkConfig};
use std::ops::*;

pub trait Signed: AddAssign + Copy + Ord + Shr<Bitlen, Output = Self> + Mul<Output = Self> {
  const ZERO: Self;
  const MAX: Self;
  const BITS: Bitlen;

  fn from_i64(x: i64) -> Self;
  fn to_f64(self) -> f64;
}

macro_rules! impl_signed {
  ($t: ty, $latent: ty, $header_byte: expr) => {
    impl NumberPriv for $t {
      const NUMBER_TYPE_BYTE: u8 = $header_byte;

      type L = $latent;

      fn mode_is_valid(mode: &Mode) -> bool {
        unsigned::mode_is_valid::<Self::L>(mode)
      }
      fn choose_mode_and_split_latents(
        nums: &[Self],
        config: &ChunkConfig,
      ) -> PcoResult<ModeAndLatents> {
        unsigned::choose_mode_and_split_latents(&nums, config)
      }

      #[inline]
      fn from_latent_ordered(l: Self::L) -> Self {
        (l as Self).wrapping_add(Self::MIN)
      }
      #[inline]
      fn to_latent_ordered(self) -> Self::L {
        self.wrapping_sub(Self::MIN) as $latent
      }
      fn join_latents(
        mode: &Mode,
        primary: DynLatentSlice,
        secondary: Option<DynLatentSlice>,
        dst: &mut [Self],
      ) -> PcoResult<()> {
        unsigned::join_latents(mode, primary, secondary, dst)
      }
    }

    impl Number for $t {
      fn get_latent_describers(meta: &ChunkMeta) -> PerLatentVar<LatentDescriber> {
        describers::match_classic_mode::<Self>(meta, "")
          .or_else(|| describers::match_int_modes::<Self::L>(meta, true))
          .expect("invalid mode for signed type")
      }
    }

    impl Signed for $t {
      const BITS: Bitlen = Self::BITS;
      const ZERO: Self = 0;
      const MAX: Self = Self::MAX;

      fn from_i64(x: i64) -> Self {
        x as Self
      }
      fn to_f64(self) -> f64 {
        self as f64
      }
    }
  };
}

impl_signed!(i32, u32, 3);
impl_signed!(i64, u64, 4);
impl_signed!(i16, u16, 8);
impl_signed!(i8, u8, 11);

#[cfg(test)]
mod tests {
  use crate::data_types::{LatentPriv, NumberPriv};

  #[test]
  fn test_ordering() {
    assert_eq!(i32::MIN.to_latent_ordered(), 0_u32);
    assert_eq!((-1_i32).to_latent_ordered(), u32::MID - 1);
    assert_eq!(0_i32.to_latent_ordered(), u32::MID);
    assert_eq!(i32::MAX.to_latent_ordered(), u32::MAX);
  }
}
