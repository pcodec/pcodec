use better_io::BetterBufRead;
use std::marker::PhantomData;

use crate::chunk_latent_decompressor::DynChunkLatentDecompressor;
use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::{ChunkMeta, LatentVarKey, PerLatentVar};
use crate::scratch::DecompressorScratch;
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

  pub fn new_scratch(&self) -> DecompressorScratch {
    let per_latent_var = self.per_latent_var.as_ref().map(|_key, dyn_cld| {
      let scratch = dyn_cld.new_scratch();
      Box::new(scratch)
    });

    DecompressorScratch { per_latent_var }
  }
}

/// Holds metadata about a chunk and can produce page decompressors.
#[derive(Clone, Debug)]
pub struct ChunkDecompressor<T: Number> {
  pub(crate) inner: ChunkDecompressorInner,
  // TODO in 2.0 consider removing owned scratch from ChunkDecompressor
  pub(crate) scratch: DecompressorScratch,
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

    let cd = ChunkDecompressorInner::new(meta)?;
    let scratch = cd.new_scratch();
    Ok(Self {
      inner: cd,
      scratch,
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
    PageDecompressor::<T, R>::new(src, &self.inner, n, &mut self.scratch)
  }

  /// Reads metadata for a page and returns a `PageDecompressor` and the
  /// remaining input.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// Even though this takes `&mut self`, the page decompressor only mutates the
  /// chunk decompressor's scratch buffers and has no effect on the
  /// decompression of later pages.
  pub fn page_decompressor_with_scratch<'a, R: BetterBufRead>(
    &'a mut self,
    src: R,
    n: usize,
    scratch: &'a mut DecompressorScratch,
  ) -> PcoResult<PageDecompressor<'a, T, R>> {
    PageDecompressor::<T, R>::new(src, &self.inner, n, scratch)
  }
}
