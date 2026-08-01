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
use crate::standalone::constants::MAX_N_HINT_PREALLOC;
use crate::standalone::decompressor::n_hint_prealloc;
use crate::standalone::{simple_compress, simple_decompress, FileCompressor};
use crate::ModeSpec;
use std::cmp;

/// The mutations tried at each byte offset: every single-bit flip, plus the
/// two saturating values. Cheaper than all 256 values and hits the same
/// narrow bit fields, since the interesting header fields are 1-5 bits wide.
fn corrupt_values(orig: u8) -> impl Iterator<Item = u8> {
  (0..8)
    .map(move |bit| orig ^ (1 << bit))
    .chain([0x00, 0xff])
    .filter(move |&v| v != orig)
}

fn configs() -> Vec<(&'static str, ChunkConfig)> {
  vec![
    (
      "classic",
      ChunkConfig::default().with_mode_spec(ModeSpec::Classic),
    ),
    (
      "consecutive",
      ChunkConfig::default().with_delta_spec(DeltaSpec::TryConsecutive(1)),
    ),
    (
      "lookback",
      ChunkConfig::default().with_delta_spec(DeltaSpec::TryLookback),
    ),
    (
      "int_mult",
      ChunkConfig::default().with_mode_spec(ModeSpec::TryIntMult(7)),
    ),
  ]
}

/// Every single-byte mutation of a valid file must be handled, not crash.
///
/// This walks every offset rather than sampling randomly: the interesting
/// fields are a few bits wide, so a seeded sample misses them.
#[test]
fn test_single_byte_corruption_never_panics() -> PcoResult<()> {
  let nums: Vec<i64> = (0..400).map(|i| (i * 3) % 97).collect();

  for (name, config) in configs() {
    let valid = simple_compress(&nums, &config)?;
    assert!(
      simple_decompress::<i64>(&valid)?.len() == nums.len(),
      "{} did not round trip before corruption",
      name
    );

    for idx in 0..valid.len() {
      for value in corrupt_values(valid[idx]) {
        let mut corrupt = valid.clone();
        corrupt[idx] = value;
        // The result is unconstrained -- corrupt data may legitimately decode
        // to garbage -- but reaching this line at all is the assertion.
        let _ = simple_decompress::<i64>(&corrupt);
      }
    }
  }

  Ok(())
}

/// Same, for float modes, which have their own latent-splitting arithmetic.
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

/// `n_hint` is a hint, not a promise: a 26-byte file may legally declare
/// `usize::MAX` (`standalone/guarantee.rs` writes exactly that), so it must be
/// clamped before it reaches `Vec::with_capacity`.
///
/// This checks the clamp arithmetic rather than decompressing such a file,
/// since the point of the clamp is that the allocation never happens.
#[test]
fn test_absurd_n_hint_is_clamped() {
  for n_hint in [0, 1 << 20, 1 << 40, usize::MAX] {
    let capped = n_hint_prealloc(n_hint);
    assert!(
      capped <= MAX_N_HINT_PREALLOC,
      "n_hint={} was not clamped ({})",
      n_hint,
      capped
    );
    assert_eq!(
      capped,
      cmp::min(n_hint, MAX_N_HINT_PREALLOC),
      "n_hint={} should be preserved when small enough",
      n_hint
    );
  }
}

/// The clamp must not disturb ordinary round trips, including for files whose
/// `n_hint` is absurd.
#[test]
fn test_absurd_n_hint_still_round_trips() -> PcoResult<()> {
  for n_hint in [1 << 20, 1 << 40, usize::MAX] {
    let mut file = Vec::new();
    let fc = FileCompressor::default().with_n_hint(n_hint);
    fc.write_header(&mut file)?;
    let mut cc = fc.chunk_compressor(&[7_i64], &ChunkConfig::default())?;
    cc.write(&mut file)?;
    fc.write_footer(&mut file)?;

    assert!(
      file.len() < 64,
      "n_hint={} produced an unexpectedly large file ({} bytes)",
      n_hint,
      file.len()
    );
    assert_eq!(
      simple_decompress::<i64>(&file)?,
      vec![7_i64],
      "n_hint={}",
      n_hint
    );
  }

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
