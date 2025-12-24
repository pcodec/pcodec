use std::fmt::Debug;

use better_io::BetterBufRead;

use crate::bit_reader::BitReaderBuilder;
use crate::data_types::{LatentType, Number};
use crate::errors::PcoResult;
use crate::metadata::chunk::ChunkMeta;
use crate::metadata::format_version::FormatVersion;
use crate::wrapped::chunk_decompressor::ChunkDecompressor;

/// Top-level entry point for decompressing wrapped pco files.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct FileDecompressor {
  format_version: FormatVersion,
}

impl FileDecompressor {
  /// Reads a short header and returns a `FileDecompressor` and the remaining
  /// input.
  ///
  /// Will return an error if any version incompatibilities or
  /// insufficient data are found.
  pub fn new<R: BetterBufRead>(src: R) -> PcoResult<(Self, R)> {
    let mut reader_builder = BitReaderBuilder::new(src);
    let format_version = reader_builder.with_reader(
      FormatVersion::MAX_ENCODED_SIZE,
      FormatVersion::read_from,
    )?;
    Ok((
      Self { format_version },
      reader_builder.into_inner(),
    ))
  }

  pub fn format_version(&self) -> u8 {
    self.format_version.0
  }

  /// Reads a chunk's metadata and returns a `ChunkDecompressor` and the
  /// remaining input.
  ///
  /// Will return an error if version incompatibilities, corruptions, or
  /// insufficient data are found.
  pub fn chunk_decompressor<T: Number, R: BetterBufRead>(
    &self,
    src: R,
  ) -> PcoResult<(ChunkDecompressor<T>, R)> {
    let mut reader_builder = BitReaderBuilder::new(src);
    let latent_type = LatentType::new::<T::L>().unwrap();
    let chunk_meta = ChunkMeta::read_from::<R>(
      &mut reader_builder,
      &self.format_version,
      latent_type,
    )?;
    let cd = ChunkDecompressor::new(chunk_meta)?;
    Ok((cd, reader_builder.into_inner()))
  }
}
