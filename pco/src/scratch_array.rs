use crate::{dyn_latent_slice::DynLatentSlice, macros::define_latent_enum};
use std::ops::{Deref, DerefMut};

use crate::{data_types::Latent, FULL_BATCH_N};

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

type Boxed<L> = Box<ScratchArray<L>>;

define_latent_enum!(
  #[derive(Clone, Debug)]
  pub DynScratchArray(Boxed)
);

impl DynScratchArray {
  pub fn slice<'a>(&'a self) -> DynLatentSlice<'a> {
    match self {
      Self::U16(inner) => DynLatentSlice::U16(&inner.0),
      Self::U32(inner) => DynLatentSlice::U32(&inner.0),
      Self::U64(inner) => DynLatentSlice::U64(&inner.0),
    }
  }
}
