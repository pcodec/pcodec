use std::ops::{Deref, DerefMut};

use crate::constants::Bitlen;
use crate::data_types::Latent;
use crate::dyn_slices::DynLatentSlice;
use crate::macros::define_latent_enum;
use crate::metadata::PerLatentVar;
use crate::FULL_BATCH_N;

// Struct to enforce alignment of the scratch arrays to 64 bytes. This can
// improve performance for SIMD operations. The primary goal here is to avoid
// regression by ensuring that the arrays stay "well-aligned", even if the
// surrounding code is changed.
#[derive(Clone, Debug)]
#[repr(align(64))]
pub struct ScratchArray<L: Latent>(pub [L; FULL_BATCH_N]);

impl<L: Latent> Deref for ScratchArray<L> {
  type Target = [L; FULL_BATCH_N];
  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
impl<L: Latent> DerefMut for ScratchArray<L> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

define_latent_enum!(
  #[derive()]
  pub union UniversalScratchArray(ScratchArray)
);

pub struct LatentDecompressorScratch {
  pub offset_bits_csum: ScratchArray<Bitlen>,
  pub offset_bits: ScratchArray<Bitlen>,
  pub latents: UniversalScratchArray,
}

impl LatentDecompressorScratch {
  pub fn latents<'a, L: Latent>(&'a mut self) -> DynLatentSlice<'a> {
    DynLatentSlice::new(self.latents.downcast_mut::<L>().unwrap().as_mut_slice())
  }
}

#[derive(Clone, Debug)]
pub struct DecompressorScratch {
  pub(crate) per_latent_var: Box<PerLatentVar<LatentDecompressorScratch>>,
}
