#![no_main]
//! Structure-aware corruption.
//!
//! `decompress_arbitrary` almost never gets past the header, so it barely
//! exercises metadata validation. Here we compress real data first, then let
//! the fuzzer patch bytes of the *valid* stream. Mutations therefore land on
//! parsed fields -- ANS size logs, bin counts, delta configs, offset bit
//! widths -- which is where the decoder's unsafe reads get their parameters.
//!
//! Contract: any corrupted stream must produce an error, not a panic, not an
//! out-of-bounds read, and not an unbounded allocation.

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use pco::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};

#[derive(Arbitrary, Debug)]
struct Patch {
  offset: u16,
  value: u8,
}

#[derive(Arbitrary, Debug)]
struct Input {
  delta: u8,
  mode: u8,
  compression_level: u8,
  // seed data, kept short so most of the stream is metadata
  nums: Vec<i64>,
  patches: Vec<Patch>,
}

fn base_config(input: &Input) -> ChunkConfig {
  let delta_spec = match input.delta % 5 {
    0 => DeltaSpec::Auto,
    1 => DeltaSpec::NoOp,
    2 => DeltaSpec::TryConsecutive(1),
    3 => DeltaSpec::TryConsecutive(7),
    // Lookback is the encoding whose decoder indexes a window buffer with
    // unchecked offsets read straight off the wire.
    _ => DeltaSpec::TryLookback,
  };
  let mode_spec = match input.mode % 3 {
    0 => ModeSpec::Auto,
    1 => ModeSpec::Classic,
    _ => ModeSpec::TryIntMult(7),
  };
  ChunkConfig::default()
    .with_compression_level((input.compression_level % 13) as usize)
    .with_delta_spec(delta_spec)
    .with_mode_spec(mode_spec)
    .with_paging_spec(PagingSpec::EqualPagesUpTo(64))
}

fuzz_target!(|input: Input| {
  if input.nums.is_empty() {
    return;
  }
  let Ok(mut compressed) = pco::standalone::simple_compress(&input.nums, &base_config(&input))
  else {
    return;
  };

  for patch in &input.patches {
    let idx = (patch.offset as usize) % compressed.len();
    compressed[idx] = patch.value;
  }

  // Only the absence of a crash is asserted; a corrupt stream is allowed to
  // decode to anything, or to fail.
  let _ = pco::standalone::simple_decompress::<i64>(&compressed);
});
