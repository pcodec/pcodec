use std::cmp::min;

use crate::chunk_config::ChunkConfig;
use crate::data_types::Number;
use crate::errors::PcoResult;
use crate::progress::Progress;
use crate::standalone::compressor::FileCompressor;
use crate::standalone::decompressor::{FileDecompressor, MaybeChunkDecompressor};
use crate::{PagingSpec, FULL_BATCH_N};

// TODO in 1.0 make this generic to Write and make all compress methods
// accepting a Write return the number of bytes written?
// TODO in 1.0 set a uniform number type for the file
/// Takes in a slice of numbers and an exact configuration and writes compressed
/// bytes to the destination, retuning the number of bytes written.
///
/// Will return an error if the compressor config is invalid, there is an IO
/// error.
/// This will use the `PagingSpec` in `ChunkConfig` to decide where to split
/// chunks.
/// For standalone, the concepts of chunk and page are conflated since each
/// chunk has exactly one page.
pub fn simple_compress_into<T: Number>(
  nums: &[T],
  config: &ChunkConfig,
  mut dst: &mut [u8],
) -> PcoResult<usize> {
  let original_length = dst.len();
  let file_compressor = FileCompressor::default().with_n_hint(nums.len());
  dst = file_compressor.write_header(dst)?;

  // here we use the paging spec to determine chunks; each chunk has 1 page
  let n_per_page = config.paging_spec.n_per_page(nums.len())?;
  let mut start = 0;
  let mut this_chunk_config = config.clone();
  for &page_n in &n_per_page {
    let end = start + page_n;
    this_chunk_config.paging_spec = PagingSpec::Exact(vec![page_n]);
    let chunk_compressor =
      file_compressor.chunk_compressor(&nums[start..end], &this_chunk_config)?;

    dst = chunk_compressor.write_chunk(dst)?;
    start = end;
  }

  dst = file_compressor.write_footer(dst)?;
  Ok(original_length - dst.len())
}

/// Takes in a slice of numbers and an exact configuration and returns
/// compressed bytes.
///
/// Will return an error if the compressor config is invalid.
/// This will use the `PagingSpec` in `ChunkConfig` to decide where to split
/// chunks.
/// For standalone, the concepts of chunk and page are conflated since each
/// chunk has exactly one page.
pub fn simple_compress<T: Number>(nums: &[T], config: &ChunkConfig) -> PcoResult<Vec<u8>> {
  let mut dst = Vec::new();
  let file_compressor = FileCompressor::default().with_n_hint(nums.len());
  file_compressor.write_header(&mut dst)?;

  // here we use the paging spec to determine chunks; each chunk has 1 page
  let n_per_page = config.paging_spec.n_per_page(nums.len())?;
  let mut start = 0;
  let mut this_chunk_config = config.clone();
  let mut hinted_size = false;
  for &page_n in &n_per_page {
    let end = start + page_n;
    this_chunk_config.paging_spec = PagingSpec::Exact(vec![page_n]);
    let chunk_compressor =
      file_compressor.chunk_compressor(&nums[start..end], &this_chunk_config)?;

    if !hinted_size {
      let file_size_hint =
        chunk_compressor.chunk_size_hint() as f64 * nums.len() as f64 / page_n as f64;
      dst.reserve_exact(file_size_hint as usize + 10);
      hinted_size = true;
    }

    chunk_compressor.write_chunk(&mut dst)?;
    start = end;
  }

  file_compressor.write_footer(&mut dst)?;
  Ok(dst)
}

/// Takes in compressed bytes and writes numbers to the destination, returning
/// progress into the file.
///
/// Will return an error if there are any compatibility, corruption,
/// or insufficient data issues.
/// Does not error if dst is too short or too long, but that can be inferred
/// from `Progress`.
pub fn simple_decompress_into<T: Number>(src: &[u8], mut dst: &mut [T]) -> PcoResult<Progress> {
  let (file_decompressor, mut src) = FileDecompressor::new(src)?;

  let mut incomplete_batch_buffer = vec![T::default(); FULL_BATCH_N];
  let mut progress = Progress::default();
  loop {
    let maybe_cd = file_decompressor.chunk_decompressor(src)?;
    let mut chunk_decompressor;
    match maybe_cd {
      MaybeChunkDecompressor::Some(cd) => chunk_decompressor = cd,
      MaybeChunkDecompressor::EndOfData(_) => {
        progress.finished = true;
        break;
      }
    }

    let (limit, is_limited) = if dst.len() < chunk_decompressor.n() {
      (dst.len() / FULL_BATCH_N * FULL_BATCH_N, true)
    } else {
      (dst.len(), false)
    };

    let new_progress = chunk_decompressor.decompress(&mut dst[..limit])?;
    dst = &mut dst[new_progress.n_processed..];
    progress.n_processed += new_progress.n_processed;

    // If we're near the end of dst, we do one possibly incomplete batch
    // of numbers and copy them over.
    if !dst.is_empty() {
      let new_progress = chunk_decompressor.decompress(&mut incomplete_batch_buffer)?;
      let n_processed = min(dst.len(), new_progress.n_processed);
      dst[..n_processed].copy_from_slice(&incomplete_batch_buffer[..n_processed]);
      dst = &mut dst[n_processed..];
      progress.n_processed += n_processed;
    }

    if dst.is_empty() && is_limited {
      break;
    }

    src = chunk_decompressor.into_src();
  }
  Ok(progress)
}

/// Compresses the numbers using the given compression level and an otherwise
/// default configuration.
///
/// Will panic if the compression level is invalid (see
/// [`ChunkConfig`][crate::ChunkConfig] for an explanation of compression
/// levels).
/// This wraps [`simple_compress`].
pub fn simpler_compress<T: Number>(nums: &[T], compression_level: usize) -> PcoResult<Vec<u8>> {
  let config = ChunkConfig {
    compression_level,
    ..Default::default()
  };
  simple_compress(nums, &config)
}

/// Takes in compressed bytes and returns a vector of numbers.
///
/// Will return an error if there are any compatibility, corruption,
/// or insufficient data issues.
pub fn simple_decompress<T: Number>(src: &[u8]) -> PcoResult<Vec<T>> {
  let (file_decompressor, src) = FileDecompressor::new(src)?;
  file_decompressor.simple_decompress(src)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::chunk_config::DeltaSpec;

  #[test]
  fn test_simple_compress_into() -> PcoResult<()> {
    let nums = (0..100).map(|x| x as i32).collect::<Vec<_>>();
    let config = &ChunkConfig {
      delta_spec: DeltaSpec::None,
      ..Default::default()
    };
    let mut buffer = [77];
    // error if buffer is too small
    assert!(simple_compress_into(&nums, config, &mut buffer).is_err());

    let mut buffer = vec![0; 1000];
    let bytes_written = simple_compress_into(&nums, config, &mut buffer)?;
    assert!(bytes_written >= 10);
    for i in bytes_written..buffer.len() {
      assert_eq!(buffer[i], 0);
    }
    let decompressed = simple_decompress::<i32>(&buffer[..bytes_written])?;
    assert_eq!(decompressed, nums);

    Ok(())
  }

  #[test]
  fn test_simple_decompress_into() -> PcoResult<()> {
    let max_n = 600;
    let nums = (0..max_n).map(|x| x as i32).collect::<Vec<i32>>();
    let src = simple_compress(
      &nums,
      &ChunkConfig {
        compression_level: 0,
        delta_spec: DeltaSpec::None,
        paging_spec: PagingSpec::Exact(vec![300, 300]),
        ..Default::default()
      },
    )?;

    for possibly_overshooting_n in [0, 1, 256, 299, 300, 301, 556, 600, 601] {
      let mut dst = vec![0; possibly_overshooting_n];
      let progress = simple_decompress_into(&src, &mut dst)?;
      let n = min(possibly_overshooting_n, max_n);
      assert_eq!(progress.n_processed, n);
      assert_eq!(progress.finished, n >= nums.len());
      assert_eq!(
        &dst[..n],
        &(0..n).map(|x| x as i32).collect::<Vec<i32>>(),
        "n={}",
        n
      );
    }

    Ok(())
  }
}
