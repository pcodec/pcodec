use std::fmt::Debug;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};

use crate::constants::Bitlen;
use crate::data_types::Latent;
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

impl<L: Latent> Default for ScratchArray<L> {
  fn default() -> Self {
    Self([L::ZERO; FULL_BATCH_N])
  }
}

// We never use the type contained in universal scratch, but the important thing
// is that it's large enough to be used as scratch by any latent type. We tried
// implementing with a union, but since we never reference the individual fields
// it was unnecessarily complicated.
#[derive(Clone, Debug)]
#[repr(align(64))]
pub struct UniversalScratchArray([MaybeUninit<u64>; FULL_BATCH_N]);

impl UniversalScratchArray {
  pub fn downcast_mut<L: Latent>(&mut self) -> &mut [MaybeUninit<L>; FULL_BATCH_N] {
    // Safety: we're returning maybeuninits
    unsafe { mem::transmute(&mut self.0) }
  }

  pub unsafe fn downcast_init<L: Latent>(&self) -> &[L; FULL_BATCH_N] {
    mem::transmute(&self.0)
  }

  pub unsafe fn downcast_init_mut<L: Latent>(&mut self) -> &mut [L; FULL_BATCH_N] {
    mem::transmute(&mut self.0)
  }
}

impl Default for UniversalScratchArray {
  fn default() -> Self {
    Self([MaybeUninit::uninit(); FULL_BATCH_N])
  }
}

#[derive(Clone, Debug, Default)]
pub struct LatentDecompressorScratch {
  pub only_bin_offset_bits: Option<Bitlen>, // used to check if this scratch is primed for some perf optimizations
  pub offset_bits_csum: ScratchArray<Bitlen>,
  pub offset_bits: ScratchArray<Bitlen>,
  pub latents: UniversalScratchArray,
}

#[derive(Clone, Debug, Default)]
pub struct DecompressorScratchInner {
  pub delta: LatentDecompressorScratch,
  pub primary: LatentDecompressorScratch,
  pub secondary: LatentDecompressorScratch,
}

/// A spare memory region used by Pco during decompression.
///
/// Power users who need multiple threads sharing chunks should use one scratch
/// per active [`PageDecompressor`][crate::wrapped::PageDecompressor].
/// This likely means one per thread.
#[derive(Clone, Debug, Default)]
pub struct DecompressorScratch(pub(crate) Box<DecompressorScratchInner>);
