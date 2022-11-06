use crate::{ChunkMetadata, CompressorConfig, Flags};
use crate::base_compressor::BaseCompressor;
use crate::chunk_metadata::ChunkSpec;
use crate::data_types::NumberLike;
use crate::errors::QCompressResult;
use crate::mode::Standalone;

/// Converts vectors of numbers into compressed bytes.
///
/// All `Compressor` methods leave its state unchanged if they return an error.
/// You can configure behavior like compression level by instantiating with
/// [`.from_config()`][Compressor::from_config]
///
/// You can use the compressor at a file or chunk level.
/// ```
/// use q_compress::standalone::Compressor;
///
/// let my_nums = vec![1, 2, 3];
///
/// // FILE LEVEL
/// let mut compressor = Compressor::<i32>::default();
/// let bytes = compressor.simple_compress(&my_nums);
///
/// // CHUNK LEVEL
/// let mut compressor = Compressor::<i32>::default();
/// compressor.header().expect("header");
/// compressor.chunk(&my_nums).expect("chunk");
/// compressor.footer().expect("footer");
/// let bytes = compressor.drain_bytes();
/// ```
/// Note that in practice we would need larger chunks than this to
/// achieve good compression, preferably containing 3k-10M numbers.
#[derive(Clone, Debug, Default)]
pub struct Compressor<T: NumberLike>(BaseCompressor<T, Standalone>);

const DEFAULT_CHUNK_SIZE: usize = 1000000;

impl<T: NumberLike> Compressor<T> {
  /// Creates a new compressor, given a [`CompressorConfig`].
  /// Internally, the compressor builds [`Flags`] as well as an internal
  /// configuration that doesn't show up in the output file.
  /// You can inspect the flags it chooses with [`.flags()`][Self::flags].
  pub fn from_config(config: CompressorConfig) -> Self {
    Self(BaseCompressor::<T, Standalone>::from_config(config))
  }
  /// Returns a reference to the compressor's flags.
  pub fn flags(&self) -> &Flags {
    &self.0.flags
  }

  /// Writes out a header using the compressor's data type and flags.
  /// Will return an error if the compressor has already written the header.
  ///
  /// Each .qco file must start with such a header, which contains:
  /// * a 4-byte magic header for "qco!" in ascii,
  /// * a byte for the data type (e.g. `i64` has byte 1 and `f64` has byte
  /// 5), and
  /// * bytes for the flags used to compress.
  pub fn header(&mut self) -> QCompressResult<()> {
    self.0.header()
  }

  /// Writes out a chunk of data representing the provided numbers.
  /// Will return an error if the compressor has not yet written the header
  /// or already written the footer.
  ///
  /// Each chunk contains a [`ChunkMetadata`] section followed by the chunk body.
  /// The chunk body encodes the numbers passed in here.
  pub fn chunk(&mut self, nums: &[T]) -> QCompressResult<ChunkMetadata<T>> {
    let pre_meta_bit_idx = self.0.writer.bit_size();
    let mut meta = self.0.chunk_metadata_internal(nums, &ChunkSpec::default())?;
    let post_meta_byte_idx = self.0.writer.byte_size();

    self.0.data_page_internal()?;

    meta.compressed_body_size = self.0.writer.byte_size() - post_meta_byte_idx;
    meta.update_write_compressed_body_size(&mut self.0.writer, pre_meta_bit_idx);
    Ok(meta)
  }

  /// Writes out a single footer byte indicating that the .qco file has ended.
  /// Will return an error if the compressor has not yet written the header
  /// or already written the footer.
  pub fn footer(&mut self) -> QCompressResult<()> {
    self.0.footer()
  }

  /// Takes in a slice of numbers and returns compressed bytes.
  pub fn simple_compress(&mut self, nums: &[T]) -> Vec<u8> {
    // The following unwraps are safe because the writer will be byte-aligned
    // after each step and ensure each chunk has appropriate size.
    self.header().unwrap();
    nums.chunks(DEFAULT_CHUNK_SIZE)
      .for_each(|chunk| {
        self.chunk(chunk).unwrap();
      });

    self.footer().unwrap();
    self.drain_bytes()
  }

  /// Returns all bytes produced by the compressor so far that have not yet
  /// been read.
  ///
  /// In the future we may implement a method to write to a `std::io::Write` or
  /// implement `Compressor` as `std::io::Read`, TBD.
  pub fn drain_bytes(&mut self) -> Vec<u8> {
    self.0.writer.drain_bytes()
  }

  /// Returns the number of bytes produced by the compressor so far that have
  /// not yet been read.
  pub fn byte_size(&mut self) -> usize {
    self.0.writer.byte_size()
  }
}
