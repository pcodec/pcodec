use better_io::BetterBufRead;
use std::cell::OnceCell;
use std::marker::PhantomData;

use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::{ChunkMeta, PerLatentVar};
use crate::page_latent_decompressor::DynChunkLatentVarConfig;
use crate::wrapped::page_decompressor::make_latent_var_configs;
use crate::wrapped::PageDecompressor;

/// Holds metadata about a chunk and can produce page decompressors.
#[derive(Clone, Debug)]
pub struct ChunkDecompressor<T: Number> {
  pub(crate) meta: ChunkMeta,
  latent_var_configs: OnceCell<PerLatentVar<DynChunkLatentVarConfig>>,
  phantom: PhantomData<T>,
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

    Ok(Self {
      meta,
      latent_var_configs: OnceCell::new(),
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
    // TODO error handling
    let latent_var_configs = self
      .latent_var_configs
      .get_or_init(|| make_latent_var_configs(&self.meta).unwrap());
    PageDecompressor::<T, R>::new(src, &self.meta, latent_var_configs, n)
  }
}
