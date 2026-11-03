use better_io::BetterBufRead;
use std::marker::PhantomData;

use crate::chunk_latent_decompressor::DynChunkLatentDecompressor;
use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::{ChunkMeta, LatentVarKey, PerLatentVar};
use crate::scratch::{DecompressorScratch, DecompressorScratchInner};
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

  pub(crate) fn adapt_scratch(&self, scratch: &mut DecompressorScratchInner) {
    if let Some(cld) = self.per_latent_var.delta.as_ref() {
      cld.adapt_scratch(&mut scratch.delta);
    }
    self
      .per_latent_var
      .primary
      .adapt_scratch(&mut scratch.primary);
    if let Some(cld) = self.per_latent_var.secondary.as_ref() {
      cld.adapt_scratch(&mut scratch.secondary);
    }
  }
}

/// Holds metadata about a chunk and can produce page decompressors.
#[derive(Clone, Debug)]
pub struct ChunkDecompressor<T: Number> {
  pub(crate) inner: ChunkDecompressorInner,
  // TODO in 2.0 consider removing owned scratch from ChunkDecompressor
  pub(crate) scratch: Option<DecompressorScratch>,
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

    Ok(Self {
      inner: ChunkDecompressorInner::new(meta)?,
      scratch: None,
      phantom: PhantomData,
    })
  }

  /// Returns pre-computed information about the chunk.
  pub fn meta(&self) -> &ChunkMeta {
    &self.inner.meta
  }

  pub(crate) fn inner_and_scratch(
    &mut self,
  ) -> (
    &ChunkDecompressorInner,
    &mut DecompressorScratch,
  ) {
    let scratch = self.scratch.get_or_insert_with(|| {
      let mut scratch = DecompressorScratch::default();
      self.inner.adapt_scratch(&mut scratch.0);
      scratch
    });
    (&self.inner, scratch)
  }

  /// Reads metadata for a page and returns a `PageDecompressor` and the
  /// remaining input.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// Even though this takes `&mut self`, the page decompressor only mutates the
  /// chunk decompressor's internal scratch buffers and has no effect on the
  /// decompression of later pages.
  /// Use
  /// [`Self::page_decompressor_with_scratch`][Self::page_decompressor_with_scratch]
  /// if you need to multiple references to this chunk decompressor.
  pub fn page_decompressor<R: BetterBufRead>(
    &mut self,
    src: R,
    n: usize,
  ) -> PcoResult<PageDecompressor<'_, T, R>> {
    let (inner, scratch) = self.inner_and_scratch();
    PageDecompressor::<T, R>::new(src, inner, n, scratch)
  }

  /// Reads metadata for a page and returns a `PageDecompressor` and the
  /// remaining input, using external scratch space.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// The benefit of this over
  /// [`Self::page_decompressor`][Self::page_decompressor] is that it requires
  /// only an immutable reference to `self`, allowing the chunk decompressor to
  /// be used by multiple threads in parallel.
  /// Scratch can be created by
  /// [`DecompressorScratch::default`][crate::DecompressorScratch::default].
  pub fn page_decompressor_with_scratch<'a, R: BetterBufRead>(
    &'a self,
    src: R,
    n: usize,
    scratch: &'a mut DecompressorScratch,
  ) -> PcoResult<PageDecompressor<'a, T, R>> {
    self.inner.adapt_scratch(&mut scratch.0);
    PageDecompressor::<T, R>::new(src, &self.inner, n, scratch)
  }
}
