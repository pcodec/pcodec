use std::io::Write;

use crate::bit_writer::BitWriter;
use crate::data_types::Number;
use crate::errors::PcoResult;
use crate::metadata::format_version::FormatVersion;
use crate::wrapped::chunk_compressor;
use crate::wrapped::chunk_compressor::ChunkCompressor;
use crate::ChunkConfig;

/// The top-level struct for compressing wrapped pco files.
///
/// Example of the lowest level API for writing a wrapped file:
/// ```
/// use pco::ChunkConfig;
/// use pco::wrapped::FileCompressor;
/// # use pco::errors::PcoResult;
///
/// # fn main() -> PcoResult<()> {
/// let mut compressed = Vec::new();
/// let file_compressor = FileCompressor::default();
/// // probably write some custom stuff here
/// file_compressor.write_header(&mut compressed)?;
/// // probably write more custom stuff here
/// for chunk in [vec![1, 2, 3], vec![4, 5]] {
///   let mut chunk_compressor = file_compressor.chunk_compressor::<i64>(
///     &chunk,
///     &ChunkConfig::default(),
///   )?;
///   chunk_compressor.write_meta(&mut compressed)?;
///   for page_idx in 0..chunk_compressor.n_per_page().len() {
///     // probably write more custom stuff here
///     chunk_compressor.write_page(page_idx, &mut compressed)?;
///   }
/// }
/// // probably write more custom stuff here
/// // now `compressed` is a complete file with 2 chunks
/// # Ok(())
/// # }
/// ```
///
/// The one requirement for a wrapping format is that it saves the count of
/// numbers in each page; this will be needed for decompression.
/// Otherwise, you may write anything else you like in your wrapping file!
#[derive(Clone, Debug, Default)]
pub struct FileCompressor {
  pub(crate) format_version: FormatVersion,
}

impl FileCompressor {
  /// Writes a short header to the destination.
  ///
  /// Will return an error if the provided `Write` errors.
  pub fn write_header<W: Write>(&self, dst: W) -> PcoResult<W> {
    let mut writer = BitWriter::new(dst, FormatVersion::MAX_ENCODED_SIZE);
    self.format_version.write_to(&mut writer)?;
    writer.flush()?;
    Ok(writer.into_inner())
  }

  /// Creates a `ChunkCompressor` that can be used to write chunk metadata
  /// and create page compressors.
  ///
  /// Will return an error if any arguments provided are invalid.
  ///
  /// Although this doesn't write anything yet, it does the bulk of
  /// compute necessary for the compression.
  pub fn chunk_compressor<T: Number>(
    &self,
    nums: &[T],
    config: &ChunkConfig,
  ) -> PcoResult<ChunkCompressor> {
    chunk_compressor::new(nums, config)
  }
}
