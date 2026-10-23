use better_io::BetterBufRead;
use std::marker::PhantomData;

use crate::chunk_latent_decompressor::{
  DynChunkLatentDecompressor, DynChunkLatentDecompressorScratch,
};
use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::{ChunkMeta, LatentVarKey, PerLatentVar};
use crate::wrapped::PageDecompressor;

#[derive(Clone, Debug)]
pub struct ChunkDecompressorInner {
  pub(crate) meta: ChunkMeta,
  pub(crate) per_latent_var: PerLatentVar<DynChunkLatentDecompressor>,
}

impl ChunkDecompressorInner {
  fn new(meta: ChunkMeta) -> PcoResult<Self> {
    let per_latent_var = meta.per_latent_var.as_ref().map_result(|key, latent_var| {
      let delta_encoding = meta.delta_encoding.for_latent_var(key);
      DynChunkLatentDecompressor::create(latent_var, delta_encoding)
    })?;

    Ok(Self {
      meta,
      per_latent_var,
    })
  }

  pub fn n_latents_per_delta_state(&self) -> usize {
    self
      .meta
      .delta_encoding
      .for_latent_var(LatentVarKey::Primary)
      .n_latents_per_state()
  }
}

/// Working memory for decompressing pages of a shared `ChunkDecompressor`,
/// one per thread. It works for any chunk with the same latent variables and
/// latent types as the chunk that made it.
#[derive(Clone, Debug)]
pub struct DecompressorScratch {
  pub(crate) per_latent_var: PerLatentVar<DynChunkLatentDecompressorScratch>,
}

/// Holds metadata about a chunk and can produce page decompressors.
#[derive(Clone, Debug)]
pub struct ChunkDecompressor<T: Number> {
  pub(crate) inner: ChunkDecompressorInner,
  phantom: PhantomData<T>,
}

impl<T: Number> ChunkDecompressor<T> {
  pub(crate) fn new(meta: ChunkMeta) -> PcoResult<Self> {
    if !T::mode_is_valid(&meta.mode) {
      return Err(PcoError::corruption(format!(
        "invalid mode for {} number type: {:?}",
        std::any::type_name::<T>(),
        meta.mode
      )));
    }

    ChunkDecompressorInner::new(meta).map(|cd| Self {
      inner: cd,
      phantom: PhantomData,
    })
  }

  /// Returns pre-computed information about the chunk.
  pub fn meta(&self) -> &ChunkMeta {
    &self.inner.meta
  }

  /// Reads metadata for a page and returns a `PageDecompressor` and the
  /// remaining input.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// Even though this takes `&mut self`, the page decompressor only mutates the
  /// chunk decompressor's scratch buffers and has no effect on the
  /// decompression of later pages.
  pub fn page_decompressor<R: BetterBufRead>(
    &mut self,
    src: R,
    n: usize,
  ) -> PcoResult<PageDecompressor<'_, T, R>> {
    PageDecompressor::<T, R>::new(src, self, n)
  }

  pub fn new_scratch(&self) -> DecompressorScratch {
    let per_latent_var = self
      .inner
      .per_latent_var
      .as_ref()
      .map(|_key, cld| cld.new_scratch());
    DecompressorScratch { per_latent_var }
  }

  /// Returns whether the scratch has this chunk's latent variables and latent
  /// types.
  pub fn is_compatible_with_scratch(&self, scratch: &DecompressorScratch) -> bool {
    let chunk_types = self
      .meta()
      .per_latent_var
      .as_ref()
      .map(|_key, latent_var| latent_var.latent_type());
    let scratch_types = scratch
      .per_latent_var
      .as_ref()
      .map(|_key, scratch| scratch.latent_type());
    chunk_types == scratch_types
  }

  /// Like `page_decompressor`, but decompresses into the given scratch, so it
  /// only needs `&self` and threads can share this chunk decompressor.
  ///
  /// Will return an error if the scratch is not compatible with this chunk.
  pub fn page_decompressor_with_scratch<'a, R: BetterBufRead>(
    &'a self,
    scratch: &'a mut DecompressorScratch,
    src: R,
    n: usize,
  ) -> PcoResult<PageDecompressor<'a, T, R>> {
    if !self.is_compatible_with_scratch(scratch) {
      return Err(PcoError::invalid_argument(
        "scratch is not compatible with this chunk; create one for it with new_scratch",
      ));
    }
    PageDecompressor::<T, R>::new_with_scratch(src, self, scratch, n)
  }
}
