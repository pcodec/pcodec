use std::mem;

use crate::data_types::{unsigneds, ModeAndLatents, Number};
use crate::describers::LatentDescriber;
use crate::errors::PcoResult;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{ChunkMeta, DynLatents, Mode};
use crate::{describers, int_mult_utils, ChunkConfig};

macro_rules! impl_signed {
  ($t: ty, $latent: ty, $header_byte: expr) => {
    impl Number for $t {
      const NUMBER_TYPE_BYTE: u8 = $header_byte;

      type L = $latent;

      fn get_latent_describers(meta: &ChunkMeta) -> PerLatentVar<LatentDescriber> {
        describers::match_classic_mode::<Self>(meta, "")
          .or_else(|| describers::match_int_modes::<Self::L>(meta, true))
          .expect("invalid mode for signed type")
      }

      fn mode_is_valid(mode: Mode) -> bool {
        unsigneds::mode_is_valid::<Self::L>(mode)
      }
      fn choose_mode_and_split_latents(
        nums: &[Self],
        config: &ChunkConfig,
      ) -> PcoResult<ModeAndLatents> {
        unsigneds::choose_mode_and_split_latents(&nums, config)
      }

      #[inline]
      fn from_latent_ordered(l: Self::L) -> Self {
        (l as Self).wrapping_add(Self::MIN)
      }
      #[inline]
      fn to_latent_ordered(self) -> Self::L {
        self.wrapping_sub(Self::MIN) as $latent
      }
      fn join_latents(mode: Mode, primary: &mut [Self::L], secondary: Option<&DynLatents>) {
        match mode {
          Mode::Classic => (),
          Mode::IntMult(dyn_latent) => {
            let base = *dyn_latent.downcast_ref::<Self::L>().unwrap();
            int_mult_utils::join_latents(base, primary, secondary)
          }
          _ => unreachable!("impossible mode for signed ints"),
        }
      }

      fn transmute_to_latents(slice: &mut [Self]) -> &mut [Self::L] {
        unsafe { mem::transmute(slice) }
      }
      #[inline]
      fn transmute_to_latent(self) -> Self::L {
        unsafe { mem::transmute(self) }
      }
    }
  };
}

impl_signed!(i32, u32, 3);
impl_signed!(i64, u64, 4);
impl_signed!(i16, u16, 8);

#[cfg(test)]
mod tests {
  use crate::data_types::{Latent, Number};

  #[test]
  fn test_ordering() {
    assert_eq!(i32::MIN.to_latent_ordered(), 0_u32);
    assert_eq!((-1_i32).to_latent_ordered(), u32::MID - 1);
    assert_eq!(0_i32.to_latent_ordered(), u32::MID);
    assert_eq!(i32::MAX.to_latent_ordered(), u32::MAX);
  }
}
