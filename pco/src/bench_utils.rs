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

// CodSpeed rejects runs that report the same benchmark URI twice, so its
// multi-arch CI matrix needs arch-specific benchmark names.
#[cfg(target_arch = "x86_64")]
macro_rules! arch_bench_name {
  ($name:literal) => {
    concat!($name, "_x86_64")
  };
}
#[cfg(target_arch = "aarch64")]
macro_rules! arch_bench_name {
  ($name:literal) => {
    concat!($name, "_aarch64")
  };
}
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
macro_rules! arch_bench_name {
  ($name:literal) => {
    $name
  };
}
pub(crate) use arch_bench_name;
