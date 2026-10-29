//! Tests that a corrupt or hostile file is rejected rather than crashing.
//!
//! `stability.rs` covers *truncation*; this covers *mutation* and headers whose
//! declared sizes are not backed by any data. Everything the decoder reads is
//! attacker-controlled, so the contract is: return a `PcoError`, or decode to
//! something -- but never panic, and never allocate on a number the input has
//! not earned.

use crate::chunk_config::{ChunkConfig, DeltaSpec};
use crate::data_types::Number;
use crate::errors::PcoResult;
use crate::standalone::{simple_compress, simple_decompress, FileCompressor, FileDecompressor};
use crate::ModeSpec;

/// The mutations tried at each byte offset: every single-bit flip, plus the
/// two saturating values. Cheaper than all 256 values and hits the same
/// narrow bit fields, since the interesting header fields are 1-5 bits wide.
fn corrupt_values(orig: u8) -> impl Iterator<Item = u8> {
  (0..8)
    .map(move |bit| orig ^ (1 << bit))
    .chain([0x00, 0xff])
    .filter(move |&v| v != orig)
}

#[test]
fn test_single_byte_corruption_never_panics() -> PcoResult<()> {
  let nums: Vec<i64> = (0..400).map(|i| (i * 3) % 97).collect();

  for config in [
    ChunkConfig::default().with_mode_spec(ModeSpec::Classic),
    ChunkConfig::default().with_delta_spec(DeltaSpec::TryConsecutive(1)),
    ChunkConfig::default().with_delta_spec(DeltaSpec::TryLookback),
    ChunkConfig::default().with_mode_spec(ModeSpec::TryIntMult(7)),
  ] {
    let valid = simple_compress(&nums, &config)?;
    assert_eq!(
      simple_decompress::<i64>(&valid)?.len(),
      nums.len(),
      "{:?} did not round trip before corruption",
      config
    );

    for idx in 0..valid.len() {
      for value in corrupt_values(valid[idx]) {
        let mut corrupt = valid.clone();
        corrupt[idx] = value;
        // Corrupt data may legitimately decode to garbage, so reaching this
        // line at all is the assertion.
        let _ = simple_decompress::<i64>(&corrupt);
      }
    }
  }

  Ok(())
}

#[test]
fn test_single_byte_corruption_never_panics_floats() -> PcoResult<()> {
  let nums: Vec<f64> = (0..400).map(|i| (i as f64) * 0.25).collect();

  for mode_spec in [
    ModeSpec::Classic,
    ModeSpec::TryFloatQuant(20),
    ModeSpec::TryFloatMult(0.25),
  ] {
    let config = ChunkConfig::default().with_mode_spec(mode_spec);
    let valid = simple_compress(&nums, &config)?;

    for idx in 0..valid.len() {
      for value in corrupt_values(valid[idx]) {
        let mut corrupt = valid.clone();
        corrupt[idx] = value;
        let _ = simple_decompress::<f64>(&corrupt);
      }
    }
  }

  Ok(())
}

/// `n_hint` is a hint, not a promise: a tiny file may legally declare
/// `usize::MAX` (`standalone/guarantee.rs` writes exactly that), so it must not
/// reach `Vec::with_capacity` unclamped.
#[test]
fn test_absurd_n_hint_is_clamped() -> PcoResult<()> {
  let mut file = Vec::new();
  let fc = FileCompressor::default().with_n_hint(usize::MAX);
  fc.write_header(&mut file)?;
  let mut cc = fc.chunk_compressor(&[7_i64], &ChunkConfig::default())?;
  cc.write(&mut file)?;
  fc.write_footer(&mut file)?;

  let (fd, src) = FileDecompressor::new(file.as_slice())?;
  // 8 KiB is 1024 i64s.
  let nums = fd.with_max_prealloc(8192).simple_decompress::<i64>(src)?;

  assert_eq!(nums, vec![7_i64]);
  assert!(
    nums.capacity() <= 1024,
    "preallocated {} numbers",
    nums.capacity()
  );

  Ok(())
}

/// `FloatQuant`'s `k` is used as a shift width, so it must be rejected before
/// any latent splitting happens -- not after.
#[test]
fn test_float_quant_k_beyond_type_width_is_rejected() {
  fn check<T: Number>(k: u32, bits: u32) {
    let config = ChunkConfig::default().with_mode_spec(ModeSpec::TryFloatQuant(k));
    let nums = vec![T::default(); 8];
    assert!(
      simple_compress(&nums, &config).is_err(),
      "k={} should be rejected for a {}-bit type",
      k,
      bits
    );
  }

  for k in [64, 65, 100, 255, 256, u32::MAX] {
    check::<f64>(k, 64);
  }
  for k in [32, 33, 64, 255, 256, u32::MAX] {
    check::<f32>(k, 32);
  }
}
