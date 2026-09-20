use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;

use crate::constants::{Bitlen, FULL_BATCH_N};
use crate::data_types::Latent;

/// Every decompression microbenchmark works over this many latents, in this
/// many batches, so that their throughputs are directly comparable.
pub const BENCH_BATCHES: usize = 64;
pub const BENCH_N: usize = BENCH_BATCHES * FULL_BATCH_N;

pub fn uniform_latents<L: Latent>(n_bits: Bitlen) -> Vec<L> {
  let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
  (0..BENCH_N)
    .map(|_| L::from_u64(rng.next_u64() >> (u64::BITS - n_bits)))
    .collect()
}

/// Every end-to-end benchmark compresses or decompresses this many numbers, so
/// that their throughputs are directly comparable.
pub const E2E_N: usize = 1 << 16;

fn rng() -> Xoroshiro128PlusPlus {
  Xoroshiro128PlusPlus::seed_from_u64(0)
}

/// Uniformly random i64s, which neither a mode nor a delta encoding can help
/// with, so this measures the plain Classic path.
pub fn random_i64s() -> Vec<i64> {
  let mut rng = rng();
  (0..E2E_N).map(|_| rng.next_u64() as i64).collect()
}

/// Microsecond timestamps increasing by roughly a second each, which benefit
/// from consecutive delta encoding.
pub fn timestamp_i64s() -> Vec<i64> {
  let mut rng = rng();
  let mut t = 1_600_000_000_000_000_i64;
  (0..E2E_N)
    .map(|_| {
      t += 1_000_000 + (rng.next_u64() % 1_000) as i64 * 1_000;
      t
    })
    .collect()
}

/// A few thousand distinct i32 IDs in random order, i.e. many repeated values
/// spread over a wide range.
pub fn id_i32s() -> Vec<i32> {
  let mut rng = rng();
  let dict = (0..4096).map(|_| rng.next_u64() as i32).collect::<Vec<_>>();
  (0..E2E_N)
    .map(|_| dict[rng.next_u64() as usize % dict.len()])
    .collect()
}

/// f64 prices that are all multiples of 0.01, the canonical FloatMult case.
pub fn decimal_f64s() -> Vec<f64> {
  let mut rng = rng();
  (0..E2E_N)
    .map(|_| (rng.next_u64() % 10_000_000) as f64 * 0.01)
    .collect()
}

/// Smooth f32s with a bit of noise, as in sensor readings.
pub fn smooth_f32s() -> Vec<f32> {
  let mut rng = rng();
  (0..E2E_N)
    .map(|i| {
      let noise = (rng.next_u64() % 1_000) as f32 * 1e-4;
      (i as f32 * 0.01).sin() * 100.0 + noise
    })
    .collect()
}
