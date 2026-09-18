#![no_main]
//! The C decompression entry point against files it did not ask for.
//!
//! `pco_standalone_simple_decompress_into` is what a C caller points at a file
//! from disk or from a socket. Unstructured bytes barely get past the magic
//! header (that is what `decompress_arbitrary` measures), so this target
//! compresses real numbers through the C API first and *then* corrupts them,
//! and -- the part no other target covers -- hands back a destination that is
//! usually too small for what the file decodes to.
//!
//! That last case is the one thing this API can get memory-wrong: the count of
//! elements is discovered while decoding, the caller's capacity is a separate
//! parameter, and the two are only reconciled at the copy. Asserted:
//!
//! * the guard bands around the destination survive whatever the file claims;
//! * on success `*n_written` is within the capacity the caller passed;
//! * on failure `*n_written` is left alone -- a caller that reads it after an
//!   error would otherwise walk off its own buffer;
//! * no panic crosses the FFI boundary (unwinding out of `extern "C"` is
//!   undefined behaviour, so this is worse for a C caller than for a Rust one).

use std::ffi::c_void;

use arbitrary::Arbitrary;
use cpcodec::{
  pco_standalone_guarantee_file_size, pco_standalone_simple_compress_into,
  pco_standalone_simple_decompress_into, PcoChunkConfig, PcoError,
};
use libfuzzer_sys::fuzz_target;

mod guarded;
use guarded::Guarded;

const PCO_TYPE_I64: u8 = 4;

/// Every valid discriminant from cpcodec.h, plus invalid ones. Decompressing
/// as the wrong type must be refused by the type byte in the file, and an
/// unknown dtype must come back as `PcoInvalidType` rather than dispatching to
/// some element size.
const DTYPES: [u8; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 12, 255];

/// Element size per dtype byte; 0 for the ones the library must reject.
fn dtype_size(dtype: u8) -> usize {
  match dtype {
    1 | 3 | 5 => 4,   // u32, i32, f32
    2 | 4 | 6 => 8,   // u64, i64, f64
    7 | 8 | 9 => 2,   // u16, i16, f16
    10 | 11 => 1,     // u8, i8
    _ => 0,
  }
}

#[derive(Arbitrary, Debug)]
struct Patch {
  offset: u16,
  value: u8,
}

#[derive(Arbitrary, Debug)]
struct Input {
  dtype_sel: u8,
  /// Destination capacity in *elements*, deliberately narrow: the point is to
  /// hand back less room than the file decodes into.
  dst_cap: u8,
  nums: Vec<i64>,
  patches: Vec<Patch>,
}

fuzz_target!(|input: Input| {
  if input.nums.is_empty() {
    return;
  }

  // Produce a real file through the same C API a caller would use.
  let bound = pco_standalone_guarantee_file_size(input.nums.len(), PCO_TYPE_I64);
  if bound == 0 {
    return;
  }
  let mut cdst = Guarded::new(bound);
  let mut compressed_len: usize = 0;
  let config = PcoChunkConfig::default();
  let res = unsafe {
    pco_standalone_simple_compress_into(
      input.nums.as_ptr() as *const c_void,
      input.nums.len(),
      PCO_TYPE_I64,
      &config,
      cdst.ptr(),
      bound,
      &mut compressed_len,
    )
  };
  cdst.check("compress dst");
  if !matches!(res, PcoError::PcoSuccess) {
    return;
  }

  let mut compressed = cdst.payload()[..compressed_len].to_vec();
  for patch in &input.patches {
    let idx = (patch.offset as usize) % compressed.len();
    compressed[idx] = patch.value;
  }

  let dtype = DTYPES[input.dtype_sel as usize % DTYPES.len()];
  let dst_cap = input.dst_cap as usize;
  // Size the payload by the dtype actually passed, so a dispatch to a wider
  // element type shows up as a guard hit and not as harness slack.
  let mut ddst = Guarded::new(dst_cap * dtype_size(dtype).max(1));
  let mut n_written: usize = usize::MAX;

  let res = unsafe {
    pco_standalone_simple_decompress_into(
      compressed.as_ptr() as *const c_void,
      compressed.len(),
      dtype,
      ddst.ptr(),
      dst_cap,
      &mut n_written,
    )
  };
  ddst.check("decompress dst");

  match res {
    PcoError::PcoSuccess => assert!(
      n_written <= dst_cap,
      "decompress_into reported {n_written} elements written into a {dst_cap}-element buffer"
    ),
    _ => assert_eq!(
      n_written,
      usize::MAX,
      "decompress_into wrote n_written on a failure path ({res:?})"
    ),
  }
});
