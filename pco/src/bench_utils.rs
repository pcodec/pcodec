//! Fixtures for the microbenchmarks that live in `#[cfg(feature = "bench")]`
//! modules throughout the crate. See `benches/decompress.rs`.

use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;

use crate::ans::AnsState;
use crate::bit_reader::BitReader;
use crate::chunk_latent_decompressor::ChunkLatentDecompressor;
use crate::constants::{Bitlen, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::{Latent, Number};
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::page::PageMeta;
use crate::metadata::LatentVarKey;
use crate::page_latent_decompressor::PageLatentDecompressor;
use crate::wrapped::{FileCompressor, FileDecompressor};
use crate::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};

const SEED: u64 = 0;
// Reads at the end of a batch overshoot the bits they need by up to 15 bytes.
const PADDING: usize = 64;
// Enough space between distinct values that merging two of them into one bin
// would cost more offset bits than it saves in metadata.
// Distinct values are spread over the latent type's range, up to this much
// space between them, so that merging two of them into one bin would cost more
// offset bits than it saves in metadata.
const MAX_SEPARATION: u128 = 1000;

pub fn rng() -> Xoroshiro128PlusPlus {
  Xoroshiro128PlusPlus::seed_from_u64(SEED)
}

/// Values drawn from a geometrically weighted set of `n_distinct` widely
/// separated values.
///
/// Separating the values keeps bin optimization from merging them away, which
/// is what makes the bin count (and therefore the ANS table) controllable.
pub fn clustered_nums<L: Latent>(n: usize, n_distinct: usize) -> Vec<L> {
  assert!(n_distinct <= 1 << L::BITS.min(16));
  let mut rng = rng();
  let separation = ((1_u128 << L::BITS) / n_distinct as u128).min(MAX_SEPARATION) as u64;
  let mean = n_distinct as f64 / 4.0;
  (0..n)
    .map(|_| {
      let u = (rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
      let symbol = (((-(1.0 - u).ln()) * mean) as usize).min(n_distinct - 1);
      L::from_u64(symbol as u64 * separation)
    })
    .collect()
}

/// Uniformly random values in `[0, 2^precision)`.
///
/// At compression level 0 these land in a single bin whose offsets are
/// `precision` bits wide, so the page body is pure offsets and the ANS path is
/// skipped entirely.
pub fn uniform_nums<L: Latent>(n: usize, precision: Bitlen) -> Vec<L> {
  let mut rng = rng();
  (0..n)
    .map(|_| {
      let x = rng.next_u64() >> (u64::BITS - precision);
      L::from_u64(x)
    })
    .collect()
}

/// A random walk, which is the shape consecutive and conv1 delta encoding
/// exist for.
pub fn random_walk_nums<L: Latent>(n: usize) -> Vec<L> {
  let mut rng = rng();
  let mut x = 0_u64;
  (0..n)
    .map(|_| {
      x = x.wrapping_add(rng.next_u64() % 64).wrapping_sub(32);
      L::from_u64(x)
    })
    .collect()
}

/// Interleaved subsequences, which is the shape lookback delta encoding exists
/// for.
pub fn interleaved_nums<L: Latent>(n: usize) -> Vec<L> {
  const N_SUBSEQS: usize = 16;
  let mut rng = rng();
  let mut subseqs = [0_u64; N_SUBSEQS];
  (0..n)
    .map(|i| {
      let subseq = &mut subseqs[i % N_SUBSEQS];
      *subseq = subseq.wrapping_add(rng.next_u64() % 8);
      L::from_u64(*subseq)
    })
    .collect()
}

/// A batch count and latent count shared by the per-batch microbenchmarks, so
/// that their throughputs are directly comparable.
pub const BENCH_BATCHES: usize = 64;
pub const BENCH_N: usize = BENCH_BATCHES * FULL_BATCH_N;

/// A page's worth of compressed primary latents, plus the decompressor state
/// needed to read it.
///
/// Built by actually compressing `nums`, so bin weights, ANS bit consumption
/// and offset widths are all whatever pco really produces.
pub struct LatentFixture<L: Latent> {
  cld: Box<ChunkLatentDecompressor<L>>,
  ans_final_state_idxs: [AnsState; ANS_INTERLEAVING],
  delta_encoding: LatentVarDeltaEncoding,
  delta_state: Vec<L>,
  src: Vec<u8>,
  body_byte_idx: usize,
  unpadded_len: usize,
  n_batches: usize,
}

impl<L: Latent> LatentFixture<L> {
  pub fn new<T: Number<L = L>>(nums: &[T], config: &ChunkConfig) -> Self {
    assert_eq!(nums.len() % FULL_BATCH_N, 0);
    let config = ChunkConfig {
      paging_spec: PagingSpec::Exact(vec![nums.len()]),
      ..config.clone()
    };

    let fc = FileCompressor::default();
    let header = fc.write_header(Vec::new()).unwrap();
    let mut cc = fc.chunk_compressor(nums, &config).unwrap();
    let chunk_meta = cc.write_meta(Vec::new()).unwrap();
    let mut src = cc.write_page(0, Vec::new()).unwrap();
    let unpadded_len = src.len();
    src.resize(unpadded_len + PADDING, 0);

    let (fd, _) = FileDecompressor::new(header.as_slice()).unwrap();
    let (cd, _) = fd
      .chunk_decompressor::<T, _>(chunk_meta.as_slice())
      .unwrap();
    let (page_latent_var, delta_encoding, body_byte_idx) = {
      let meta = cd.meta();
      let mut reader = BitReader::new(&src, unpadded_len, 0);
      let page_meta = unsafe { PageMeta::read_from(&mut reader, meta) }.unwrap();
      (
        page_meta.per_latent_var.primary,
        meta.delta_encoding.for_latent_var(LatentVarKey::Primary),
        meta.exact_page_meta_size(),
      )
    };
    assert_eq!(delta_encoding, LatentVarDeltaEncoding::NoOp);

    let cld = cd
      .inner
      .per_latent_var
      .primary
      .downcast::<L>()
      .expect("primary latent type does not match the fixture's");

    Self {
      cld,
      ans_final_state_idxs: page_latent_var.ans_final_state_idxs,
      delta_encoding,
      delta_state: page_latent_var.delta_state.downcast::<L>().unwrap(),
      body_byte_idx,
      src,
      unpadded_len,
      n_batches: nums.len() / FULL_BATCH_N,
    }
  }

  pub fn n_bins(&self) -> usize {
    self.cld.n_bins
  }

  pub fn bytes_per_offset(&self) -> usize {
    self.cld.bytes_per_offset
  }

  pub fn n_batches(&self) -> usize {
    self.n_batches
  }

  pub fn n_latents(&self) -> usize {
    self.n_batches * FULL_BATCH_N
  }

  /// A reader positioned at the first byte of the page body, i.e. where the
  /// first batch's ANS symbols begin.
  pub fn reader(&self) -> BitReader<'_> {
    let mut reader = BitReader::new(&self.src, self.unpadded_len, 0);
    reader.stale_byte_idx = self.body_byte_idx;
    reader
  }

  pub fn pld(&self) -> PageLatentDecompressor<L> {
    PageLatentDecompressor::new(
      self.ans_final_state_idxs,
      &self.delta_encoding,
      self.delta_state.clone(),
    )
  }

  pub fn cld(&self) -> Box<ChunkLatentDecompressor<L>> {
    self.cld.clone()
  }
}

/// Configuration whose only latent var is a primary one with no delta encoding,
/// so a fixture's page body contains exactly one latent var's data.
pub fn single_latent_var_config(compression_level: usize) -> ChunkConfig {
  ChunkConfig::default()
    .with_compression_level(compression_level)
    .with_mode_spec(ModeSpec::Classic)
    .with_delta_spec(DeltaSpec::NoOp)
    .with_enable_8_bit(true)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::constants::MAX_COMPRESSION_LEVEL;

  #[test]
  fn bin_counts_track_distinct_values() {
    for (n_distinct, expected_n_bins) in [(16, 16), (64, 64)] {
      let nums = clustered_nums::<u32>(BENCH_N, n_distinct);
      let fixture = LatentFixture::new(
        &nums,
        &single_latent_var_config(MAX_COMPRESSION_LEVEL),
      );
      assert_eq!(fixture.n_bins(), expected_n_bins);
    }
  }

  #[test]
  fn offset_fixture_has_one_bin() {
    let nums = uniform_nums::<u32>(4 * FULL_BATCH_N, 32);
    let fixture = LatentFixture::new(&nums, &single_latent_var_config(0));
    assert_eq!(fixture.n_bins(), 1);
    assert_eq!(fixture.bytes_per_offset(), 5);
  }

  /// The fixtures are only meaningful if reading them back reproduces the
  /// original latents, so check that a full pass does.
  #[test]
  fn fixture_round_trips() {
    let nums = clustered_nums::<u64>(4 * FULL_BATCH_N, 64);
    let fixture = LatentFixture::new(
      &nums,
      &single_latent_var_config(MAX_COMPRESSION_LEVEL),
    );
    let mut reader = fixture.reader();
    let mut pld = fixture.pld();
    let mut cld = fixture.cld();

    let mut recovered = Vec::with_capacity(nums.len());
    for _ in 0..fixture.n_batches() {
      unsafe { pld.read_batch_pre_delta(&mut reader, FULL_BATCH_N, &mut cld) };
      recovered.extend_from_slice(&cld.scratch.latents[..FULL_BATCH_N]);
    }

    assert_eq!(recovered, nums);
  }
}
