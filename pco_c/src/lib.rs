#![allow(clippy::missing_safety_doc)]

use std::ptr;

use libc::{c_uchar, c_uint, c_void, size_t};

use crate::PcoError::PcoInvalidType;
use pco::data_types::{Number, NumberType};
use pco::errors::{ErrorKind, PcoError as PcoLibError};
use pco::standalone::{guarantee, FileDecompressor};
use pco::{match_number_enum, ChunkConfig, PagingSpec};

#[repr(C)]
pub enum PcoError {
  PcoSuccess,
  PcoInvalidType,
  /// Generic compression failure, e.g. an IO error from the destination.
  PcoCompressionError,
  /// Generic decompression failure of a kind not covered below.
  PcoDecompressionError,
  /// The parameters provided were invalid, e.g. an unsupported compression
  /// level.
  PcoInvalidArgumentError,
  /// The provided data is inconsistent or violates the pco format.
  PcoCorruptionError,
  /// The provided data ended before decompression finished.
  PcoInsufficientDataError,
}

/// Maps a library error to the C error code, using `fallback` for kinds with
/// no dedicated code (IO errors and kinds added to the library later).
fn err_code(err: &PcoLibError, fallback: PcoError) -> PcoError {
  match err.kind {
    ErrorKind::Corruption => PcoError::PcoCorruptionError,
    ErrorKind::InsufficientData => PcoError::PcoInsufficientDataError,
    ErrorKind::InvalidArgument => PcoError::PcoInvalidArgumentError,
    _ => fallback,
  }
}

/// Configuration for compression, passed by the caller.
///
/// Only `compression_level` and `paging_spec` are supported for now; other
/// fields can be added later without breaking the ABI.
#[repr(C)]
pub struct PcoChunkConfig {
  /// Compression level 0–12 (default 8).
  pub compression_level: c_uint,
  /// Maximum number of elements per page.
  /// Set to 0 to use the library default (2^18 = 262144).
  pub max_page_n: size_t,
}

impl Default for PcoChunkConfig {
  fn default() -> Self {
    Self {
      compression_level: 8,
      max_page_n: 0,
    }
  }
}

impl PcoChunkConfig {
  fn to_chunk_config(&self) -> ChunkConfig {
    let paging_spec = if self.max_page_n == 0 {
      PagingSpec::default()
    } else {
      PagingSpec::EqualPagesUpTo(self.max_page_n)
    };
    ChunkConfig::default()
      .with_compression_level(self.compression_level as usize)
      .with_paging_spec(paging_spec)
      .with_enable_8_bit(true)
  }
}

// ---------------------------------------------------------------------------
// Caller-allocates API (thread-safe, no Rust heap ownership)
//
// Pattern:
//   1. Call pco_standalone_guarantee_file_size(n, dtype) to learn the maximum
//      output size.
//   2. Allocate that many bytes yourself.
//   3. Call pco_standalone_simple_compress_into(...) to fill your buffer.
//   4. For decompression, allocate n * sizeof(dtype) bytes for the output,
//      then call pco_standalone_simple_decompress_into(...).
//
// These functions are safe to call concurrently from multiple threads without
// any locking because they hold no shared mutable state.
// ---------------------------------------------------------------------------

fn _guarantee_file_size<T: Number>(n: size_t, paging_spec: &PagingSpec) -> size_t {
  guarantee::file_size::<T::L>(n, paging_spec).unwrap_or(0)
}

fn _compress_into<T: Number>(
  nums: *const c_void,
  n: size_t,
  config: &ChunkConfig,
  dst: *mut c_void,
  dst_cap: size_t,
  n_written: *mut size_t,
) -> PcoError {
  let src = unsafe { std::slice::from_raw_parts(nums as *const T, n) };
  // &mut [u8] implements Write; simple_compress_into returns the remaining
  // (unwritten) portion of the slice so we can compute bytes written.
  let dst_bytes: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, dst_cap) };
  let original_len = dst_bytes.len();
  match pco::standalone::simple_compress_into::<T, _>(src, config, dst_bytes) {
    Err(e) => err_code(&e, PcoError::PcoCompressionError),
    Ok(remaining) => {
      unsafe { *n_written = original_len - remaining.len() };
      PcoError::PcoSuccess
    }
  }
}

fn _decompress_into<T: Number>(
  compressed: *const c_void,
  compressed_len: size_t,
  dst: *mut c_void,
  dst_cap: size_t,
  n_written: *mut size_t,
) -> PcoError {
  let src = unsafe { std::slice::from_raw_parts(compressed as *const u8, compressed_len) };
  match pco::standalone::simple_decompress::<T>(src) {
    Err(e) => err_code(&e, PcoError::PcoDecompressionError),
    Ok(v) => {
      let n = v.len();
      if n > dst_cap {
        return PcoError::PcoDecompressionError;
      }
      unsafe {
        ptr::copy_nonoverlapping(v.as_ptr(), dst as *mut T, n);
        *n_written = n;
      }
      PcoError::PcoSuccess
    }
  }
}

/// Return the maximum possible byte size of a standalone file for `n`
/// elements of `dtype`.  Returns 0 for an invalid `dtype` or invalid
/// paging spec.
///
/// This function is thread-safe and performs no heap allocation.
#[no_mangle]
pub extern "C" fn pco_standalone_guarantee_file_size(n: size_t, dtype: c_uchar) -> size_t {
  let Some(dtype_enum) = NumberType::from_descriminant(dtype) else {
    return 0;
  };
  let paging_spec = PagingSpec::default();
  match_number_enum!(
    dtype_enum,
    NumberType<T> => { _guarantee_file_size::<T>(n, &paging_spec) }
  )
}

/// Compress `n` numbers of `dtype` from `nums` into the caller-owned buffer
/// `dst` (capacity `dst_cap` bytes).  On success `*n_written` is the number
/// of compressed bytes written.
///
/// Thread-safe: the function is stateless and operates entirely on the
/// caller-supplied buffers.
#[no_mangle]
pub unsafe extern "C" fn pco_standalone_simple_compress_into(
  nums: *const c_void,
  n: size_t,
  dtype: c_uchar,
  config: *const PcoChunkConfig,
  dst: *mut c_void,
  dst_cap: size_t,
  n_written: *mut size_t,
) -> PcoError {
  let Some(dtype_enum) = NumberType::from_descriminant(dtype) else {
    return PcoInvalidType;
  };
  let chunk_config = if config.is_null() {
    PcoChunkConfig::default().to_chunk_config()
  } else {
    unsafe { &*config }.to_chunk_config()
  };
  match_number_enum!(
    dtype_enum,
    NumberType<T> => {
      _compress_into::<T>(nums, n, &chunk_config, dst, dst_cap, n_written)
    }
  )
}

fn _decompress_partial_into<T: Number>(
  compressed: *const c_void,
  compressed_len: size_t,
  dst: *mut c_void,
  dst_cap: size_t,
  n_written: *mut size_t,
  finished: *mut c_uchar,
) -> PcoError {
  let src = unsafe { std::slice::from_raw_parts(compressed as *const u8, compressed_len) };
  let dst = unsafe { std::slice::from_raw_parts_mut(dst as *mut T, dst_cap) };
  match pco::standalone::simple_decompress_into::<T>(src, dst) {
    Err(e) => err_code(&e, PcoError::PcoDecompressionError),
    Ok(progress) => {
      unsafe {
        *n_written = progress.n_processed;
        *finished = progress.finished as c_uchar;
      }
      PcoError::PcoSuccess
    }
  }
}

/// Inspect a standalone file's header without decompressing it.
///
/// On success, `*dtype` is set to the file's uniform number type byte (or 0
/// if the file does not declare a uniform type) and `*n_hint` is set to the
/// file's count hint: the total number of elements in the file if it was
/// recorded at compression time, or 0 if unknown.  Files written by
/// `pco_standalone_simple_compress_into` always record an exact count hint.
///
/// Thread-safe: the function is stateless and operates entirely on the
/// caller-supplied buffers.
#[no_mangle]
pub unsafe extern "C" fn pco_standalone_file_info(
  compressed: *const c_void,
  compressed_len: size_t,
  dtype: *mut c_uchar,
  n_hint: *mut size_t,
) -> PcoError {
  let src = unsafe { std::slice::from_raw_parts(compressed as *const u8, compressed_len) };
  match FileDecompressor::new(src) {
    Err(e) => err_code(&e, PcoError::PcoDecompressionError),
    Ok((fd, _)) => {
      unsafe {
        *dtype = fd.uniform_type().map(|t| t as c_uchar).unwrap_or(0);
        *n_hint = fd.n_hint();
      }
      PcoError::PcoSuccess
    }
  }
}

/// Decompress `compressed_len` bytes from `compressed` into the caller-owned
/// buffer `dst` (capacity `dst_cap` *elements* of `dtype`), decompressing as
/// many elements as fit.
///
/// Unlike `pco_standalone_simple_decompress_into`, an undersized `dst` is not
/// an error: on success `*n_written` is the number of elements written and
/// `*finished` is 1 if the entire file was decompressed, 0 if elements remain.
///
/// Thread-safe: the function is stateless and operates entirely on the
/// caller-supplied buffers.
#[no_mangle]
pub unsafe extern "C" fn pco_standalone_simple_decompress_partial_into(
  compressed: *const c_void,
  compressed_len: size_t,
  dtype: c_uchar,
  dst: *mut c_void,
  dst_cap: size_t,
  n_written: *mut size_t,
  finished: *mut c_uchar,
) -> PcoError {
  let Some(dtype_enum) = NumberType::from_descriminant(dtype) else {
    return PcoInvalidType;
  };
  match_number_enum!(
    dtype_enum,
    NumberType<T> => {
      _decompress_partial_into::<T>(
        compressed,
        compressed_len,
        dst,
        dst_cap,
        n_written,
        finished,
      )
    }
  )
}

/// Decompress `compressed_len` bytes from `compressed` into the caller-owned
/// buffer `dst` (capacity `dst_cap` *elements* of `dtype`).  On success
/// `*n_written` is the number of elements written.
///
/// Thread-safe: the function is stateless and operates entirely on the
/// caller-supplied buffers.
#[no_mangle]
pub extern "C" fn pco_standalone_simple_decompress_into(
  compressed: *const c_void,
  compressed_len: size_t,
  dtype: c_uchar,
  dst: *mut c_void,
  dst_cap: size_t,
  n_written: *mut size_t,
) -> PcoError {
  let Some(dtype_enum) = NumberType::from_descriminant(dtype) else {
    return PcoInvalidType;
  };
  match_number_enum!(
    dtype_enum,
    NumberType<T> => {
      _decompress_into::<T>(compressed, compressed_len, dst, dst_cap, n_written)
    }
  )
}
