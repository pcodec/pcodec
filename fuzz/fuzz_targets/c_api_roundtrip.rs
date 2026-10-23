#![no_main]
//! The C API's caller-allocates contract, end to end.
//!
//! `pco_c` exposes three functions and documents them as a sequence: ask
//! `pco_standalone_guarantee_file_size` how large the output can get, allocate
//! exactly that, then `pco_standalone_simple_compress_into`. Everything about
//! that design puts a raw caller pointer in front of the library, and the only
//! thing exercising it was one happy-path C file: six f64s, default config.
//!
//! Asserted here:
//!
//! * guard bands either side of every caller buffer stay untouched -- the one
//!   failure mode of this API that is memory-unsafe rather than merely wrong;
//! * the size the library guarantees is actually sufficient, i.e. following the
//!   documented sequence never fails for lack of room;
//! * `*n_written` never exceeds the capacity the caller passed (a C caller uses
//!   it as a length, so an overstated one is a read past the end);
//! * the round trip is bit-exact, NaN payloads included.

use std::ffi::c_void;

use arbitrary::Arbitrary;
use cpcodec::{
  pco_standalone_guarantee_file_size, pco_standalone_guarantee_file_size_with_config,
  pco_standalone_simple_compress_into, pco_standalone_simple_decompress_into, PcoChunkConfig,
  PcoError,
};
use half::f16;
use libfuzzer_sys::fuzz_target;

mod guarded;
use guarded::Guarded;

// Type discriminants, copied from pco_c/include/cpcodec.h -- a C caller has
// nothing else, so the harness uses exactly what a C caller would.
const PCO_TYPE_U32: u8 = 1;
const PCO_TYPE_U64: u8 = 2;
const PCO_TYPE_I32: u8 = 3;
const PCO_TYPE_I64: u8 = 4;
const PCO_TYPE_F32: u8 = 5;
const PCO_TYPE_F64: u8 = 6;
const PCO_TYPE_U16: u8 = 7;
const PCO_TYPE_I16: u8 = 8;
const PCO_TYPE_F16: u8 = 9;
const PCO_TYPE_U8: u8 = 10;
const PCO_TYPE_I8: u8 = 11;

/// The range the C header documents for `PcoChunkConfig.compression_level`.
/// The field is a `c_uint`, so everything above this is reachable from C and
/// has to be *rejected*, not clamped.
const MAX_COMPRESSION_LEVEL: u32 = 12;

#[derive(Arbitrary, Debug)]
enum Nums {
  U32(Vec<u32>),
  U64(Vec<u64>),
  I32(Vec<i32>),
  I64(Vec<i64>),
  F32(Vec<f32>),
  F64(Vec<f64>),
  U16(Vec<u16>),
  I16(Vec<i16>),
  /// Raw bits: `half::f16` has no `Arbitrary` impl, and going through bits also
  /// reaches the payloads a float literal generator would never produce.
  F16(Vec<u16>),
  U8(Vec<u8>),
  I8(Vec<i8>),
}

#[derive(Arbitrary, Debug)]
struct Input {
  /// Not `% 13`: the C struct field is a `c_uint` and nothing on the C side
  /// stops a caller from putting anything in it, so out-of-range levels are
  /// part of the surface under test.
  compression_level: u32,
  /// 0 means "library default" per the C header.
  max_page_n: u16,
  nums: Nums,
}

/// One documented compress -> decompress cycle against guarded buffers.
///
/// `T` is only ever the Rust type matching `dtype`; the pointers handed over
/// are properly aligned for it, since a misaligned `nums` would be the
/// *caller's* contract violation, not the library's.
fn roundtrip<T: Copy>(nums: &[T], dtype: u8, config: &PcoChunkConfig) {
  let bound = unsafe { pco_standalone_guarantee_file_size_with_config(nums.len(), dtype, config) };
  // The config-free variant is documented as "assuming the default paging
  // spec", so it may only differ from the above when the caller set one.
  if config.max_page_n == 0 {
    assert_eq!(
      bound,
      pco_standalone_guarantee_file_size(nums.len(), dtype),
      "the two guarantee functions disagree on the default paging spec"
    );
  }
  if bound == 0 {
    // Documented as "invalid dtype or invalid paging spec". The dtype is valid
    // by construction, so this is the paging spec, and a caller who follows the
    // header stops here.
    return;
  }

  let mut cdst = Guarded::new(bound);
  let mut compressed_len: usize = 0;
  let res = unsafe {
    pco_standalone_simple_compress_into(
      nums.as_ptr() as *const c_void,
      nums.len(),
      dtype,
      config,
      cdst.ptr(),
      bound,
      &mut compressed_len,
    )
  };
  cdst.check("compress dst");

  // The guarantee only has to hold for a config the header actually permits.
  // An out-of-range level is a caller error and must come back as an error --
  // but it must come back, not be quietly accepted.
  let level_ok = config.compression_level <= MAX_COMPRESSION_LEVEL;
  match (res, level_ok) {
    (PcoError::PcoSuccess, true) => {}
    (PcoError::PcoSuccess, false) => panic!(
      "compress_into accepted compression_level {}, outside the documented 0..={MAX_COMPRESSION_LEVEL}",
      config.compression_level
    ),
    (_, false) => return,
    (res, true) => panic!(
      // The whole promise of step 1 is that step 3 then fits. Anything else
      // means the documented sequence cannot be followed.
      "compress_into failed ({res:?}) on a buffer of {bound} bytes that \
       guarantee_file_size itself sized for {} values of dtype {dtype}, \
       config {config:?}",
      nums.len()
    ),
  }
  assert!(
    compressed_len <= bound,
    "compress_into reported {compressed_len} bytes written into a {bound}-byte buffer"
  );

  // Decompression: capacity is in *elements* here, not bytes (the two
  // parameters of this API that are counted differently).
  let mut ddst = Guarded::new(nums.len() * size_of::<T>());
  let mut n_written: usize = 0;
  let res = unsafe {
    pco_standalone_simple_decompress_into(
      cdst.ptr() as *const c_void,
      compressed_len,
      dtype,
      ddst.ptr(),
      nums.len(),
      &mut n_written,
    )
  };
  ddst.check("decompress dst");

  assert!(
    matches!(res, PcoError::PcoSuccess),
    "decompress_into failed ({res:?}) on output this run just produced"
  );
  assert_eq!(
    n_written,
    nums.len(),
    "decompress_into wrote a different count than was compressed"
  );

  // Bitwise, not `PartialEq`: pco is lossless, so NaN payloads and both zeroes
  // must survive, and float equality would wave exactly those through.
  let src_bytes =
    unsafe { std::slice::from_raw_parts(nums.as_ptr() as *const u8, size_of_val(nums)) };
  assert_eq!(
    src_bytes,
    ddst.payload(),
    "round trip is not bit-exact for dtype {dtype}"
  );
}

fuzz_target!(|input: Input| {
  let config = PcoChunkConfig {
    compression_level: input.compression_level,
    max_page_n: input.max_page_n as usize,
  };

  match &input.nums {
    Nums::U32(v) => roundtrip(v, PCO_TYPE_U32, &config),
    Nums::U64(v) => roundtrip(v, PCO_TYPE_U64, &config),
    Nums::I32(v) => roundtrip(v, PCO_TYPE_I32, &config),
    Nums::I64(v) => roundtrip(v, PCO_TYPE_I64, &config),
    Nums::F32(v) => roundtrip(v, PCO_TYPE_F32, &config),
    Nums::F64(v) => roundtrip(v, PCO_TYPE_F64, &config),
    Nums::U16(v) => roundtrip(v, PCO_TYPE_U16, &config),
    Nums::I16(v) => roundtrip(v, PCO_TYPE_I16, &config),
    Nums::F16(bits) => {
      let v: Vec<f16> = bits.iter().map(|&b| f16::from_bits(b)).collect();
      roundtrip(&v, PCO_TYPE_F16, &config)
    }
    Nums::U8(v) => roundtrip(v, PCO_TYPE_U8, &config),
    Nums::I8(v) => roundtrip(v, PCO_TYPE_I8, &config),
  }
});
