#![no_main]
//! Decode fully arbitrary bytes.
//!
//! The decode path is full of `unsafe` raw loads guarded only by metadata
//! validation, so the contract under test is simply: pco must reject garbage
//! with an error, never panic and never read out of bounds.
//!
//! The first byte selects the destination type, since which arithmetic the
//! decoder performs depends on it.

use libfuzzer_sys::fuzz_target;

fn try_all(src: &[u8]) {
  // Each of these instantiates a different latent width, and therefore a
  // different arm of `read_uint_at`.
  let _ = pco::standalone::simple_decompress::<u8>(src);
  let _ = pco::standalone::simple_decompress::<u16>(src);
  let _ = pco::standalone::simple_decompress::<u32>(src);
  let _ = pco::standalone::simple_decompress::<u64>(src);
  let _ = pco::standalone::simple_decompress::<i16>(src);
  let _ = pco::standalone::simple_decompress::<i32>(src);
  let _ = pco::standalone::simple_decompress::<i64>(src);
  let _ = pco::standalone::simple_decompress::<f32>(src);
  let _ = pco::standalone::simple_decompress::<f64>(src);
  let _ = pco::standalone::simple_decompress::<half::f16>(src);
}

fuzz_target!(|data: &[u8]| {
  try_all(data);
});
