use crate::bit_writer::BitWriter;
use crate::chunk_config::PagingSpec;
use crate::data_types::{NumberLike, UnsignedLike};
use crate::errors::PcoResult;
use crate::standalone::constants::{BITS_TO_ENCODE_COMPRESSED_PAGE_SIZE, BITS_TO_ENCODE_N_ENTRIES, MAGIC_HEADER, MAGIC_TERMINATION_BYTE};
use crate::{bit_reader, wrapped, ChunkConfig, ChunkMetadata, bit_writer};
use crate::constants::MINIMAL_PADDING_BYTES;

pub struct FileCompressor(wrapped::FileCompressor);

impl FileCompressor {
  pub fn new() -> Self {
    Self(wrapped::FileCompressor::new())
  }

  pub fn header_size_hint(&self) -> usize {
    MAGIC_HEADER.len() + self.0.header_size_hint()
  }

  pub fn write_header(&self, dst: &mut [u8]) -> PcoResult<usize> {
    let mut extension = bit_reader::make_extension_for(dst, 0);
    let mut writer = BitWriter::new(dst, &mut extension);
    writer.write_aligned_bytes(&MAGIC_HEADER)?;
    let mut consumed = writer.bytes_consumed()?;
    consumed += self.0.write_header(&mut dst[consumed..])?;
    Ok(consumed)
  }

  pub fn chunk_compressor<T: NumberLike>(
    &self,
    nums: &[T],
    config: &ChunkConfig,
  ) -> PcoResult<ChunkCompressor<T::Unsigned>> {
    let mut config = config.clone();
    config.paging_spec = PagingSpec::ExactPageSizes(vec![nums.len()]);

    Ok(ChunkCompressor {
      inner: self.0.chunk_compressor(nums, &config)?,
      dtype_byte: T::DTYPE_BYTE,
    })
  }

  pub fn footer_size_hint(&self) -> usize {
    1
  }

  pub fn write_footer(&self, dst: &mut [u8]) -> PcoResult<usize> {
    let mut extension = bit_reader::make_extension_for(dst, 0);
    let mut writer = BitWriter::new(dst, &mut extension);
    writer.write_aligned_bytes(&[MAGIC_TERMINATION_BYTE])?;
    writer.bytes_consumed()
  }
}

pub struct ChunkCompressor<U: UnsignedLike> {
  inner: wrapped::ChunkCompressor<U>,
  dtype_byte: u8,
}

impl<U: UnsignedLike> ChunkCompressor<U> {
  pub fn chunk_meta(&self) -> &ChunkMetadata<U> {
    self.inner.chunk_meta()
  }

  pub fn chunk_size_hint(&self) -> usize {
    1 + self.inner.chunk_meta_size_hint() + self.inner.page_size_hint(0)
  }

  pub fn write_chunk(&self, dst: &mut [u8]) -> PcoResult<usize> {
    let mut ext = bit_reader::make_extension_for(dst, MINIMAL_PADDING_BYTES);
    let mut writer = BitWriter::new(dst, &mut ext);
    writer.ensure_padded(MINIMAL_PADDING_BYTES)?;
    writer.write_aligned_bytes(&[self.dtype_byte])?;
    writer.write_usize(self.inner.page_sizes()[0] - 1, BITS_TO_ENCODE_N_ENTRIES);
    let byte_idx_to_write_page_size = writer.aligned_dst_byte_idx()?;
    writer.write_usize(0, BITS_TO_ENCODE_COMPRESSED_PAGE_SIZE); // to be filled in later

    let mut consumed = writer.bytes_consumed()?;
    consumed += self.inner.write_chunk_meta(&mut dst[consumed..])?;

    let pre_page_consumed = consumed;
    consumed += self.inner.write_page(0, &mut dst[consumed..])?;

    // go back and fill in the compressed page size we omitted before
    ext.fill(0);
    let page_size = consumed - pre_page_consumed;
    bit_writer::write_uint_to::<_, 0>(page_size, byte_idx_to_write_page_size, 0, BITS_TO_ENCODE_COMPRESSED_PAGE_SIZE, dst);
    Ok(consumed)
  }
}
