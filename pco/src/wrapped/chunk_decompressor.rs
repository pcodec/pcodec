use better_io::BetterBufRead;
use std::marker::PhantomData;

use crate::chunk_latent_decompressor::{ChunkLatentDecompressor, DynChunkLatentDecompressor};
use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::per_latent_var::PerLatentVarBuilder;
use crate::metadata::{ChunkMeta, PerLatentVar};
use crate::wrapped::PageDecompressor;

/// Holds metadata about a chunk and can produce page decompressors.
#[derive(Clone, Debug)]
pub struct ChunkDecompressor<T: Number> {
  pub(crate) meta: ChunkMeta,
  pub(crate) per_latent_var: PerLatentVar<DynChunkLatentDecompressor>,
  phantom: PhantomData<T>,
}

fn make_clds(meta: &ChunkMeta) -> PcoResult<PerLatentVar<DynChunkLatentDecompressor>> {
  let mut builder = PerLatentVarBuilder::default();
  for (key, latent_var) in meta.per_latent_var.as_ref().enumerated() {
    let delta_encoding = meta.delta_encoding.for_latent_var(key);
    let cld = DynChunkLatentDecompressor::create(latent_var, delta_encoding)?;
    builder.set(key, cld);
  }
  Ok(builder.into())
}

impl<T: Number> ChunkDecompressor<T> {
  pub(crate) fn new(meta: ChunkMeta) -> PcoResult<Self> {
    if !T::mode_is_valid(&meta.mode) {
      return Err(PcoError::corruption(format!(
        "invalid mode for {} data type: {:?}",
        std::any::type_name::<T>(),
        meta.mode
      )));
    }
    meta.validate_delta_encoding()?;

    let per_latent_var = make_clds(&meta)?;

    Ok(Self {
      meta,
      per_latent_var,
      phantom: PhantomData,
    })
  }

  /// Returns pre-computed information about the chunk.
  pub fn meta(&self) -> &ChunkMeta {
    &self.meta
  }

  /// Reads metadata for a page and returns a `PageDecompressor` and the
  /// remaining input.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  pub fn page_decompressor<R: BetterBufRead>(
    &self,
    src: R,
    n: usize,
  ) -> PcoResult<PageDecompressor<T, R>> {
    PageDecompressor::<T, R>::new(src, &self.meta, &self.per_latent_var, n)
  }
}
