use better_io::BetterBufRead;

use crate::bit_reader::{BitReader, BitReaderBuilder};
use crate::constants::Bitlen;
use crate::data_types::{Number, NumberType};
use crate::errors::{PcoError, PcoResult};
use crate::metadata::ChunkMeta;
use crate::progress::Progress;
use crate::standalone::constants::*;
use crate::standalone::NumberTypeOrTermination;
use crate::{bit_reader, wrapped};

unsafe fn read_varint(reader: &mut BitReader) -> PcoResult<u64> {
  let power = 1 + reader.read_uint::<Bitlen>(BITS_TO_ENCODE_VARINT_POWER);
  let res = reader.read_uint(power);
  reader.drain_empty_byte("standalone size hint")?;
  Ok(res)
}

unsafe fn read_uniform_type(reader: &mut BitReader) -> PcoResult<Option<NumberType>> {
  let byte = reader.read_aligned_bytes(1)?[0];
  if byte == MAGIC_TERMINATION_BYTE {
    return Ok(None);
  }

  match NumberType::from_descriminant(byte) {
    Some(number_type) => Ok(Some(number_type)),
    None => Err(PcoError::corruption(format!(
      "unknown data type byte: {}",
      byte
    ))),
  }
}

/// Top-level entry point for decompressing standalone .pco files.
///
/// Example of the lowest level API for reading a .pco file:
/// ```
/// use pco::FULL_BATCH_N;
/// use pco::standalone::{FileDecompressor, MaybeChunkDecompressor};
/// # use pco::errors::PcoResult;
///
/// # fn main() -> PcoResult<()> {
/// let compressed = vec![112, 99, 111, 33, 0, 0]; // the minimal .pco file, for the sake of example
/// let mut nums = vec![0; FULL_BATCH_N];
/// let (file_decompressor, mut src) = FileDecompressor::new(compressed.as_slice())?;
/// while let MaybeChunkDecompressor::Some(mut chunk_decompressor) = file_decompressor.chunk_decompressor::<i64, _>(src)? {
///   let mut finished_chunk = false;
///   while !finished_chunk {
///     let progress = chunk_decompressor.decompress(&mut nums)?;
///     // Do something with &nums[0..progress.n_processed]
///     finished_chunk = progress.finished;
///   }
///   src = chunk_decompressor.into_src();
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct FileDecompressor {
  uniform_type: Option<NumberType>,
  n_hint: usize,
  inner: wrapped::FileDecompressor,
}

/// The outcome of starting a new chunk of a standalone file.
#[allow(clippy::large_enum_variant)]
pub enum MaybeChunkDecompressor<T: Number, R: BetterBufRead> {
  /// We get a `ChunkDecompressor` when there is another chunk as evidenced
  /// by the data type byte.
  Some(ChunkDecompressor<T, R>),
  /// We are at the end of the pco data if we encounter a null byte instead of
  /// a data type byte.
  EndOfData(R),
}

impl FileDecompressor {
  /// Reads a short header and returns a `FileDecompressor` and the
  /// remaining input.
  ///
  /// Will return an error if any corruptions, version incompatibilities, or
  /// insufficient data are found.
  pub fn new<R: BetterBufRead>(mut src: R) -> PcoResult<(Self, R)> {
    bit_reader::ensure_buf_read_capacity(&mut src, STANDALONE_HEADER_PADDING);
    let mut reader_builder = BitReaderBuilder::new(src, STANDALONE_HEADER_PADDING, 0);
    // Do this part first so we check for insufficient data before returning a
    // confusing corruption error.
    let header = reader_builder
      .with_reader(|reader| Ok(reader.read_aligned_bytes(MAGIC_HEADER.len())?.to_vec()))?;
    if header != MAGIC_HEADER {
      return Err(PcoError::corruption(format!(
        "magic header does not match {:?}; instead found {:?}",
        MAGIC_HEADER, header,
      )));
    }

    let (standalone_version, uniform_number_type, n_hint) =
      reader_builder.with_reader(|reader| unsafe {
        let standalone_version = reader.read_usize(BITS_TO_ENCODE_STANDALONE_VERSION);
        if standalone_version < 2 {
          // These versions only had wrapped version; we need to rewind so they can
          // reuse it.
          reader.bits_past_byte -= BITS_TO_ENCODE_STANDALONE_VERSION;
          return Ok((standalone_version, None, 0));
        }

        let uniform_type = if standalone_version >= 3 {
          read_uniform_type(reader)?
        } else {
          None
        };

        let n_hint = read_varint(reader)? as usize;

        Ok((standalone_version, uniform_type, n_hint))
      })?;

    if standalone_version > CURRENT_STANDALONE_VERSION {
      return Err(PcoError::compatibility(format!(
        "file's standalone version ({}) exceeds max supported ({}); consider upgrading pco",
        standalone_version, CURRENT_STANDALONE_VERSION,
      )));
    }

    let (inner, rest) = wrapped::FileDecompressor::new(reader_builder.into_inner())?;
    Ok((
      Self {
        inner,
        uniform_type: uniform_number_type,
        n_hint,
      },
      rest,
    ))
  }

  pub fn format_version(&self) -> u8 {
    self.inner.format_version()
  }

  pub fn uniform_type(&self) -> Option<NumberType> {
    self.uniform_type
  }

  pub fn n_hint(&self) -> usize {
    self.n_hint
  }

  /// Peeks at what's next in the file, returning whether it's a termination
  /// or chunk with some data type.
  ///
  /// If a uniform number type for the file exists, it will be used instead.
  /// Will return an error if there is insufficient data.
  pub fn peek_number_type_or_termination(&self, src: &[u8]) -> PcoResult<NumberTypeOrTermination> {
    if let Some(uniform_type) = self.uniform_type {
      return Ok(NumberTypeOrTermination::Known(uniform_type));
    }

    match src.first() {
      Some(&byte) => Ok(NumberTypeOrTermination::from(byte)),
      None => Err(PcoError::insufficient_data(
        "unable to peek data type from empty bytes",
      )),
    }
  }

  /// Reads a chunk's metadata and returns either a `ChunkDecompressor` or
  /// the rest of the source if at the end of the pco file.
  ///
  /// Will return an error if corruptions or insufficient
  /// data are found.
  pub fn chunk_decompressor<T: Number, R: BetterBufRead>(
    &self,
    mut src: R,
  ) -> PcoResult<MaybeChunkDecompressor<T, R>> {
    bit_reader::ensure_buf_read_capacity(&mut src, STANDALONE_CHUNK_PREAMBLE_PADDING);
    let mut reader_builder = BitReaderBuilder::new(src, STANDALONE_CHUNK_PREAMBLE_PADDING, 0);
    let type_or_termination_byte =
      reader_builder.with_reader(|reader| Ok(reader.read_aligned_bytes(1)?[0]))?;
    if type_or_termination_byte == MAGIC_TERMINATION_BYTE {
      return Ok(MaybeChunkDecompressor::EndOfData(
        reader_builder.into_inner(),
      ));
    }

    if let Some(uniform_type) = self.uniform_type() {
      if uniform_type as u8 != type_or_termination_byte {
        return Err(PcoError::corruption(format!(
          "chunk's number type of {} does not match file's uniform number type of {:?}",
          type_or_termination_byte, uniform_type,
        )));
      }
    }
    if type_or_termination_byte != T::NUMBER_TYPE_BYTE {
      // This is most likely user error, but since we can't be certain
      // of that, we call it a corruption.
      return Err(PcoError::corruption(format!(
        "requested chunk decompression with {:?} does not match chunk's number type of {:?}",
        NumberType::from_descriminant(T::NUMBER_TYPE_BYTE),
        type_or_termination_byte,
      )));
    }

    let n = reader_builder
      .with_reader(|reader| unsafe { Ok(reader.read_usize(BITS_TO_ENCODE_N_ENTRIES) + 1) })?;
    let src = reader_builder.into_inner();
    let (inner_cd, src) = self.inner.chunk_decompressor::<T, R>(src)?;
    let inner_pd = inner_cd.page_decompressor(src, n)?;

    let res = ChunkDecompressor {
      inner_cd,
      inner_pd,
      n,
      n_processed: 0,
    };
    Ok(MaybeChunkDecompressor::Some(res))
  }

  /// Takes in compressed bytes (after the header, at the start of the chunks)
  /// and returns a vector of numbers.
  ///
  /// Will return an error if there are any compatibility, corruption,
  /// or insufficient data issues.
  ///
  /// This function exists (in addition to the [standalone
  /// functions][crate::standalone]) because the user may want to peek at the
  /// dtype, allowing them to know which type `<T>` to use here. There is no
  /// analagous file compressor method because the user always knows the dtype
  /// during compression.
  pub fn simple_decompress<T: Number>(&self, mut src: &[u8]) -> PcoResult<Vec<T>> {
    let mut res = Vec::with_capacity(self.n_hint());
    while let MaybeChunkDecompressor::Some(mut chunk_decompressor) = self.chunk_decompressor(src)? {
      chunk_decompressor.decompress_remaining_extend(&mut res)?;
      src = chunk_decompressor.into_src();
    }
    Ok(res)
  }
}

/// Holds metadata about a chunk and supports decompression.
pub struct ChunkDecompressor<T: Number, R: BetterBufRead> {
  inner_cd: wrapped::ChunkDecompressor<T>,
  inner_pd: wrapped::PageDecompressor<T, R>,
  n: usize,
  n_processed: usize,
}

impl<T: Number, R: BetterBufRead> ChunkDecompressor<T, R> {
  /// Returns pre-computed information about the chunk.
  pub fn meta(&self) -> &ChunkMeta {
    &self.inner_cd.meta
  }

  /// Returns the count of numbers in the chunk.
  pub fn n(&self) -> usize {
    self.n
  }

  /// Reads the next decompressed numbers into the destination, returning
  /// progress into the chunk and advancing along the compressed data.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// `dst` must have length either a multiple of 256 or be at least the count
  /// of numbers remaining in the chunk.
  pub fn decompress(&mut self, dst: &mut [T]) -> PcoResult<Progress> {
    let progress = self.inner_pd.decompress(dst)?;

    self.n_processed += progress.n_processed;

    Ok(progress)
  }

  /// Returns the rest of the compressed data source.
  pub fn into_src(self) -> R {
    self.inner_pd.into_src()
  }

  // a helper for some internal things
  pub(crate) fn decompress_remaining_extend(&mut self, dst: &mut Vec<T>) -> PcoResult<()> {
    let initial_len = dst.len();
    let remaining = self.n - self.n_processed;
    dst.reserve(remaining);
    unsafe {
      dst.set_len(initial_len + remaining);
    }
    let progress = self.decompress(&mut dst[initial_len..])?;
    assert!(progress.finished);
    Ok(())
  }
}
