#![no_main]
//! Compress -> decompress must be the identity, for every input and every
//! legal config.
//!
//! The unit suite checks this (`tests/recovery.rs`) but only over a fixed,
//! hand-written corpus with a seeded RNG. Here the numbers *and* the config
//! are both driven by the fuzzer, so mode/delta combinations that no fixture
//! happens to hit get exercised too.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pco::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};

#[derive(Arbitrary, Debug)]
enum Mode {
  Auto,
  Classic,
  TryFloatMult(f64),
  TryFloatQuant(u32),
  TryIntMult(u64),
}

#[derive(Arbitrary, Debug)]
enum Delta {
  Auto,
  NoOp,
  TryConsecutive(u8),
  TryLookback,
}

#[derive(Arbitrary, Debug)]
struct Input {
  compression_level: u8,
  mode: Mode,
  delta: Delta,
  page_size: u16,
  // Which concrete number type to run. Bit width changes the latent type and
  // hence which unsafe read arm the decoder takes.
  dtype: u8,
  nums: Vec<u64>,
}

fn config(input: &Input) -> ChunkConfig {
  let mode_spec = match input.mode {
    Mode::Auto => ModeSpec::Auto,
    Mode::Classic => ModeSpec::Classic,
    Mode::TryFloatMult(x) => ModeSpec::TryFloatMult(x),
    Mode::TryFloatQuant(k) => ModeSpec::TryFloatQuant(k),
    Mode::TryIntMult(x) => ModeSpec::TryIntMult(x),
  };
  let delta_spec = match input.delta {
    Delta::Auto => DeltaSpec::Auto,
    Delta::NoOp => DeltaSpec::NoOp,
    Delta::TryConsecutive(o) => DeltaSpec::TryConsecutive(o as usize),
    Delta::TryLookback => DeltaSpec::TryLookback,
  };
  ChunkConfig::default()
    .with_compression_level((input.compression_level % 13) as usize)
    .with_mode_spec(mode_spec)
    .with_delta_spec(delta_spec)
    .with_paging_spec(PagingSpec::EqualPagesUpTo(
      1 + input.page_size as usize,
    ))
}

// `Number::to_latent_ordered` lives on a sealed supertrait, so from outside
// the crate the caller supplies the bit projection used for comparison.
fn check<T: pco::data_types::Number + std::fmt::Debug, F: Fn(&T) -> u64>(
  nums: &[T],
  config: &ChunkConfig,
  bits: F,
) {
  // Compression may legitimately refuse a config (e.g. FloatMult on ints);
  // only a *successful* compress obliges us to decompress identically.
  let Ok(compressed) = pco::standalone::simple_compress(nums, config) else {
    return;
  };
  let decompressed = pco::standalone::simple_decompress::<T>(&compressed)
    .expect("compressed data must decompress");
  assert_eq!(decompressed.len(), nums.len(), "length changed");
  for (i, (got, want)) in decompressed.iter().zip(nums).enumerate() {
    // Bitwise comparison: NaN != NaN under `==` even when identical.
    assert!(
      bits(got) == bits(want),
      "mismatch at {}: {:?} vs {:?}",
      i,
      got,
      want
    );
  }
}

fuzz_target!(|input: Input| {
  let config = config(&input);
  let raw = &input.nums;
  match input.dtype % 10 {
    0 => check::<u8, _>(&raw.iter().map(|&x| x as u8).collect::<Vec<_>>(), &config, |&x| x as u64),
    1 => check::<u16, _>(&raw.iter().map(|&x| x as u16).collect::<Vec<_>>(), &config, |&x| x as u64),
    2 => check::<u32, _>(&raw.iter().map(|&x| x as u32).collect::<Vec<_>>(), &config, |&x| x as u64),
    3 => check::<u64, _>(raw, &config, |&x| x),
    4 => check::<i16, _>(&raw.iter().map(|&x| x as i16).collect::<Vec<_>>(), &config, |&x| x as u16 as u64),
    5 => check::<i32, _>(&raw.iter().map(|&x| x as i32).collect::<Vec<_>>(), &config, |&x| x as u32 as u64),
    6 => check::<i64, _>(&raw.iter().map(|&x| x as i64).collect::<Vec<_>>(), &config, |&x| x as u64),
    7 => check::<f32, _>(
      &raw
        .iter()
        .map(|&x| f32::from_bits(x as u32))
        .collect::<Vec<_>>(),
      &config,
      |&x| x.to_bits() as u64,
    ),
    8 => check::<f64, _>(&raw.iter().map(|&x| f64::from_bits(x)).collect::<Vec<_>>(), &config, |&x| x.to_bits()),
    _ => check::<half::f16, _>(
      &raw
        .iter()
        .map(|&x| half::f16::from_bits(x as u16))
        .collect::<Vec<_>>(),
      &config,
      |&x| x.to_bits() as u64,
    ),
  }
});
