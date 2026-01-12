use std::ops::{Deref, DerefMut};

use crate::dyn_latent_slice::DynLatentSlice;
use crate::macros::{define_latent_enum, match_latent_enum};
use crate::metadata::{ChunkLatentVarMeta, DynBins};
use crate::FULL_BATCH_N;
use crate::{
  ans::{self, Spec},
  constants::Bitlen,
  data_types::Latent,
  errors::PcoResult,
  metadata::{bins, delta_encoding::LatentVarDeltaEncoding, Bin},
  read_write_uint,
};

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

#[derive(Clone, Debug)]
pub struct ChunkLatentDecompressorScratch<L: Latent> {
  pub offset_bits_csum: ScratchArray<Bitlen>,
  pub offset_bits: ScratchArray<Bitlen>,
  pub latents: ScratchArray<L>,
}

impl<L: Latent> ChunkLatentDecompressorScratch<L> {
  #[inline]
  pub unsafe fn set(&mut self, i: usize, offset_bit_idx: Bitlen, offset_bits: Bitlen, lower: L) {
    *self.offset_bits_csum.get_unchecked_mut(i) = offset_bit_idx;
    *self.offset_bits.get_unchecked_mut(i) = offset_bits;
    *self.latents.get_unchecked_mut(i) = lower;
  }
}

#[derive(Clone, Debug)]
pub struct ChunkLatentDecompressor<L: Latent> {
  pub delta_encoding: LatentVarDeltaEncoding,
  pub bytes_per_offset: usize,
  pub state_lowers: Vec<L>,
  pub n_bins: usize,
  pub decoder: ans::Decoder,
  pub scratch: ChunkLatentDecompressorScratch<L>,
}

impl<L: Latent> ChunkLatentDecompressor<L> {
  pub fn new(
    ans_size_log: Bitlen,
    bins: &[Bin<L>],
    delta_encoding: LatentVarDeltaEncoding,
  ) -> PcoResult<Self> {
    let bytes_per_offset = read_write_uint::calc_max_bytes(bins::max_offset_bits(bins));
    let bin_offset_bits = bins.iter().map(|bin| bin.offset_bits).collect::<Vec<_>>();
    let weights = bins::weights(bins);
    let ans_spec = Spec::from_weights(ans_size_log, weights)?;
    let state_lowers = ans_spec
      .state_symbols
      .iter()
      .map(|&s| bins.get(s as usize).map_or(L::ZERO, |b| b.lower))
      .collect();
    let decoder = ans::Decoder::new(&ans_spec, &bin_offset_bits);

    let only_bin = if bins.len() == 1 { Some(bins[0]) } else { None };

    let mut offset_bits_csum_scratch = ScratchArray([0; FULL_BATCH_N]);
    let mut offset_bits_scratch = ScratchArray([0; FULL_BATCH_N]);
    let mut latents = ScratchArray([L::ZERO; FULL_BATCH_N]);

    if let Some(bin) = &only_bin {
      // we optimize performance by setting state once and never again
      let mut csum = 0;
      for i in 0..FULL_BATCH_N {
        offset_bits_scratch[i] = bin.offset_bits;
        offset_bits_csum_scratch[i] = csum;
        latents[i] = bin.lower;
        csum += bin.offset_bits;
      }
    }

    Ok(Self {
      bytes_per_offset,
      state_lowers,
      n_bins: bins.len(),
      decoder,
      delta_encoding,
      scratch: ChunkLatentDecompressorScratch {
        offset_bits_csum: offset_bits_csum_scratch,
        offset_bits: offset_bits_scratch,
        latents,
      },
    })
  }
}

define_latent_enum!(
  #[derive(Clone, Debug)]
  pub DynChunkLatentDecompressor(ChunkLatentDecompressor)
);

impl DynChunkLatentDecompressor {
  pub fn create(
    latent_var: &ChunkLatentVarMeta,
    delta_encoding: LatentVarDeltaEncoding,
  ) -> PcoResult<DynChunkLatentDecompressor> {
    let res = match_latent_enum!(
      &latent_var.bins,
      DynBins<L>(bins) => {
        let inner = ChunkLatentDecompressor::new(
          latent_var.ans_size_log,
          bins,
          delta_encoding,
        )?;
        DynChunkLatentDecompressor::new(inner)
      }
    );
    Ok(res)
  }

  pub fn latents<'a>(&'a mut self) -> DynLatentSlice<'a> {
    match self {
      Self::U16(inner) => DynLatentSlice::U16(&mut *inner.scratch.latents),
      Self::U32(inner) => DynLatentSlice::U32(&mut *inner.scratch.latents),
      Self::U64(inner) => DynLatentSlice::U64(&mut *inner.scratch.latents),
    }
  }
}
