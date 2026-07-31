#![allow(clippy::missing_safety_doc)]

use std::ptr;

use libc::{c_uchar, c_uint, c_void, size_t};

use crate::PcoError::PcoInvalidType;
use pco::data_types::{Number, NumberType};
use pco::standalone::guarantee;
use pco::{match_number_enum, ChunkConfig, PagingSpec};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PcoError {
  PcoSuccess,
  PcoInvalidType,
  // TODO split this into the actual error kinds
  PcoCompressionError,
  PcoDecompressionError,
}

/// Configuration for compression, passed by the caller.
///
/// Only `compression_level` and `paging_spec` are supported for now; other
/// fields can be added later without breaking the ABI.
#[repr(C)]
// Not `Copy`: it would make `to_chunk_config(&self)` a clippy `wrong_self_convention`
// warning, and nothing here needs to pass the config by value.
#[derive(Clone, Debug)]
pub struct PcoChunkConfig {
  /// Compression level 0-12 (default 8).
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
//      output size -- or, if you pass a non-default PcoChunkConfig in step 3,
//      pco_standalone_guarantee_file_size_with_config(n, dtype, config): the
//      paging spec changes how large the file can get, and the config-free
//      variant assumes the default one.
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

/// # Safety
/// `nums` must point to `n` initialised, aligned values of `T`, and `dst` to
/// `dst_cap` writable bytes.
unsafe fn _compress_into<T: Number>(
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
    Err(_) => PcoError::PcoCompressionError,
    Ok(remaining) => {
      unsafe { *n_written = original_len - remaining.len() };
      PcoError::PcoSuccess
    }
  }
}

/// # Safety
/// `compressed` must point to `compressed_len` readable bytes, and `dst` to
/// `dst_cap` writable, aligned values of `T`.
unsafe fn _decompress_into<T: Number>(
  compressed: *const c_void,
  compressed_len: size_t,
  dst: *mut c_void,
  dst_cap: size_t,
  n_written: *mut size_t,
) -> PcoError {
  let src = unsafe { std::slice::from_raw_parts(compressed as *const u8, compressed_len) };
  match pco::standalone::simple_decompress::<T>(src) {
    Err(_) => PcoError::PcoDecompressionError,
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
/// elements of `dtype`, **assuming the default paging spec**.  Returns 0 for
/// an invalid `dtype` or invalid paging spec.
///
/// If you pass a `PcoChunkConfig` with a non-zero `max_page_n` to
/// `pco_standalone_simple_compress_into`, use
/// `pco_standalone_guarantee_file_size_with_config` instead: the file is split
/// into one chunk per page, each with its own overhead, so a smaller
/// `max_page_n` means a *larger* upper bound than this function reports.
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

/// Return the maximum possible byte size of a standalone file for `n` elements
/// of `dtype` compressed with `config`.  Returns 0 for an invalid `dtype` or
/// invalid paging spec.
///
/// This is the size to allocate before `pco_standalone_simple_compress_into`
/// whenever the config is not the default one.  A null `config` means the
/// default, making this identical to `pco_standalone_guarantee_file_size`.
///
/// This function is thread-safe and performs no heap allocation.
///
/// # Safety
/// `config` must be null or point to a valid `PcoChunkConfig`.
#[no_mangle]
pub unsafe extern "C" fn pco_standalone_guarantee_file_size_with_config(
  n: size_t,
  dtype: c_uchar,
  config: *const PcoChunkConfig,
) -> size_t {
  let Some(dtype_enum) = NumberType::from_descriminant(dtype) else {
    return 0;
  };
  let paging_spec = if config.is_null() {
    PagingSpec::default()
  } else {
    unsafe { &*config }.to_chunk_config().paging_spec
  };
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

/// Decompress `compressed_len` bytes from `compressed` into the caller-owned
/// buffer `dst` (capacity `dst_cap` *elements* of `dtype`).  On success
/// `*n_written` is the number of elements written.
///
/// Thread-safe: the function is stateless and operates entirely on the
/// caller-supplied buffers.
// `unsafe`, like its compression twin: it dereferences four caller pointers,
// so a *safe* Rust signature would have claimed that any arguments at all are
// sound. No effect on the C ABI or the generated header.
#[no_mangle]
pub unsafe extern "C" fn pco_standalone_simple_decompress_into(
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

#[cfg(test)]
mod tests {
  use super::*;

  const PCO_TYPE_I64: c_uchar = 4;

  /// Run the documented sequence for a given config and return the file.
  fn compress(nums: &[i64], config: &PcoChunkConfig) -> Vec<u8> {
    let bound =
      unsafe { pco_standalone_guarantee_file_size_with_config(nums.len(), PCO_TYPE_I64, config) };
    assert_ne!(
      bound, 0,
      "guarantee returned 0 for a valid dtype and config"
    );

    let mut dst = vec![0_u8; bound];
    let mut n_written = 0;
    let res = unsafe {
      pco_standalone_simple_compress_into(
        nums.as_ptr() as *const c_void,
        nums.len(),
        PCO_TYPE_I64,
        config,
        dst.as_mut_ptr() as *mut c_void,
        bound,
        &mut n_written,
      )
    };
    assert_eq!(
      res,
      PcoError::PcoSuccess,
      "compress_into into a {bound}-byte buffer the library sized itself"
    );
    assert!(n_written <= bound);
    dst.truncate(n_written);
    dst
  }

  /// The guarantee has to cover the config the caller will actually compress
  /// with. `pco_standalone_guarantee_file_size` assumes the default paging
  /// spec, so a small `max_page_n` -- one chunk per page, each with its own
  /// overhead -- outgrew the buffer the documented sequence allocated, and
  /// compression failed with no way for the caller to learn the right size.
  /// Found by fuzz/fuzz_targets/c_api_roundtrip.rs.
  #[test]
  fn guarantee_covers_a_non_default_paging_spec() {
    let nums: Vec<i64> = (0..21).collect();
    let config = PcoChunkConfig {
      compression_level: 11,
      max_page_n: 1,
    };
    let file = compress(&nums, &config);

    let default_bound = pco_standalone_guarantee_file_size(nums.len(), PCO_TYPE_I64);
    assert!(
      file.len() > default_bound,
      "scene proves nothing unless the paged file really does outgrow the \
       default-spec guarantee ({} vs {default_bound} bytes)",
      file.len()
    );

    // ...and it still decodes.
    let mut out = vec![0_i64; nums.len()];
    let mut n_written = 0;
    let res = unsafe {
      pco_standalone_simple_decompress_into(
        file.as_ptr() as *const c_void,
        file.len(),
        PCO_TYPE_I64,
        out.as_mut_ptr() as *mut c_void,
        out.len(),
        &mut n_written,
      )
    };
    assert_eq!(res, PcoError::PcoSuccess);
    assert_eq!(n_written, nums.len());
    assert_eq!(out, nums);
  }

  /// A destination smaller than the file decodes to must be refused without
  /// writing anything -- neither into the buffer nor into `*n_written`, which
  /// a C caller would then use as a length.
  #[test]
  fn decompress_refuses_a_too_small_destination() {
    let nums: Vec<i64> = (0..1000).collect();
    let file = compress(&nums, &PcoChunkConfig::default());

    let mut out = vec![-1_i64; 10];
    let mut n_written = usize::MAX;
    let res = unsafe {
      pco_standalone_simple_decompress_into(
        file.as_ptr() as *const c_void,
        file.len(),
        PCO_TYPE_I64,
        out.as_mut_ptr() as *mut c_void,
        out.len(),
        &mut n_written,
      )
    };
    assert_eq!(res, PcoError::PcoDecompressionError);
    assert_eq!(
      n_written,
      usize::MAX,
      "n_written was written on a failure path"
    );
    assert!(
      out.iter().all(|&x| x == -1),
      "destination was written on a failure path"
    );
  }

  /// An unknown dtype byte must be rejected by name, not dispatched to some
  /// element size.
  #[test]
  fn unknown_dtype_is_rejected() {
    let mut n_written = usize::MAX;
    let res = unsafe {
      pco_standalone_simple_decompress_into(
        [0_u8; 8].as_ptr() as *const c_void,
        8,
        255,
        [0_i64; 4].as_mut_ptr() as *mut c_void,
        4,
        &mut n_written,
      )
    };
    assert_eq!(res, PcoError::PcoInvalidType);
    assert_eq!(pco_standalone_guarantee_file_size(4, 255), 0);
  }
}
