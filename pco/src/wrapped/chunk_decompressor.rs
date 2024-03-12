use better_io::BetterBufRead;

use crate::data_types::NumberLike;
use crate::errors::PcoResult;
use crate::wrapped::PageDecompressor;
use crate::ChunkMeta;

// TODO in 0.2 make this only generic over UnsignedLike
/// Holds metadata about a chunk and can produce page decompressors.
#[derive(Clone, Debug)]
pub struct ChunkDecompressor<T: NumberLike> {
  pub(crate) meta: ChunkMeta<T::L>,
}

impl<T: NumberLike> From<ChunkMeta<T::L>> for ChunkDecompressor<T> {
  fn from(meta: ChunkMeta<T::L>) -> Self {
    Self { meta }
  }
}

impl<T: NumberLike> ChunkDecompressor<T> {
  /// Returns pre-computed information about the chunk.
  pub fn meta(&self) -> &ChunkMeta<T::L> {
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
    PageDecompressor::new(src, &self.meta, n)
  }
}
