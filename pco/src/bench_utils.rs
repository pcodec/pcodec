use rand_xoshiro::rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoroshiro128PlusPlus;

use crate::ans::AnsState;
use crate::bit_reader::BitReader;
use crate::chunk_latent_decompressor::ChunkLatentDecompressor;
use crate::constants::{Bitlen, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::{Latent, Number};
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::page::PageMeta;
use crate::metadata::{LatentVarKey, Mode};
use crate::page_latent_decompressor::PageLatentDecompressor;
use crate::wrapped::{FileCompressor, FileDecompressor};
use crate::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};

pub const BENCH_BATCHES: usize = 64;
pub const BENCH_N: usize = BENCH_BATCHES * FULL_BATCH_N;
// Reads at the end of a batch overshoot the bits they need by up to 15 bytes.
const PADDING: usize = 64;

/// Drawn from a set of `n_distinct` widely separated values.
pub fn clustered_latents<L: Latent>(n_distinct: usize) -> Vec<L> {
  assert!(n_distinct <= 1 << L::BITS.min(16));
  let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
  let separation = ((1_u128 << L::BITS) / n_distinct as u128) as u64;
  (0..BENCH_N)
    .map(|_| {
      let symbol = rng.next_u64() as usize % n_distinct;
      L::from_u64(symbol as u64 * separation)
    })
    .collect()
}

/// Uniformly random values in `[0, 2^n_bits)`.
pub fn uniform_latents<L: Latent>(n_bits: Bitlen) -> Vec<L> {
  let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
  (0..BENCH_N)
    .map(|_| {
      let x = rng.next_u64() >> (u64::BITS - n_bits);
      L::from_u64(x)
    })
    .collect()
}

/// Latents amenable to Lookback delta encoding.
pub fn interleaved_latents<L: Latent>() -> Vec<L> {
  const N_SUBSEQS: usize = 16;
  let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
  let mut subseqs = [0_u64; N_SUBSEQS];
  (0..BENCH_N)
    .map(|i| {
      let subseq = &mut subseqs[i % N_SUBSEQS];
      *subseq = subseq.wrapping_add(rng.next_u64() % 8);
      L::from_u64(*subseq)
    })
    .collect()
}

/// A page's worth of compressed primary latents, plus the decompressor state
/// needed to read it.
///
/// Built by actually compressing `nums`, so bin weights, ANS bit consumption
/// and offset widths are all whatever pco really produces.
pub struct LatentFixture<L: Latent> {
  cld: Box<ChunkLatentDecompressor<L>>,
  ans_final_state_idxs: [AnsState; ANS_INTERLEAVING],
  src: Vec<u8>,
  body_byte_idx: usize,
  unpadded_len: usize,
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
    let meta = cd.meta();
    let mut reader = BitReader::new(&src, unpadded_len, 0);
    let page_meta = unsafe { PageMeta::read_from(&mut reader, meta) }.unwrap();
    let page_latent_var = page_meta.per_latent_var.primary;
    let delta_encoding = meta.delta_encoding.for_latent_var(LatentVarKey::Primary);
    let body_byte_idx = meta.exact_page_meta_size();
    assert!(matches!(meta.mode, Mode::Classic));
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
      body_byte_idx,
      src,
      unpadded_len,
    }
  }

  pub fn reader(&self) -> BitReader<'_> {
    let mut reader = BitReader::new(&self.src, self.unpadded_len, 0);
    reader.stale_byte_idx = self.body_byte_idx;
    reader
  }

  pub fn pld(&self) -> PageLatentDecompressor<L> {
    PageLatentDecompressor::new(
      self.ans_final_state_idxs,
      &LatentVarDeltaEncoding::NoOp,
      Vec::new(),
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
    for n_distinct in [16, 64, 256] {
      let fixture = LatentFixture::new(
        &clustered_latents::<u32>(n_distinct),
        &single_latent_var_config(MAX_COMPRESSION_LEVEL),
      );
      assert_eq!(fixture.cld.n_bins, n_distinct);
    }
  }

  #[test]
  fn fixture_round_trips() {
    let latents = clustered_latents::<u64>(64);
    let fixture = LatentFixture::new(
      &latents,
      &single_latent_var_config(MAX_COMPRESSION_LEVEL),
    );
    let mut reader = fixture.reader();
    let mut pld = fixture.pld();
    let mut cld = fixture.cld();

    let mut recovered = Vec::with_capacity(latents.len());
    for _ in 0..BENCH_BATCHES {
      unsafe { pld.read_batch_pre_delta(&mut reader, FULL_BATCH_N, &mut cld) };
      recovered.extend_from_slice(&cld.scratch.latents[..FULL_BATCH_N]);
    }

    assert_eq!(recovered, latents);
  }
}
