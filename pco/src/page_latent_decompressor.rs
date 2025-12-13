use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use crate::ans::AnsState;
use crate::bit_reader::BitReader;
use crate::chunk_latent_decompressor::ChunkLatentDecompressor;
use crate::constants::{Bitlen, DeltaLookback, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::errors::{PcoError, PcoResult};
use crate::macros::define_latent_enum;
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
use crate::metadata::DynLatents;
use crate::{bit_reader, delta};

// Struct to enforce alignment of the scratch arrays to 64 bytes. This can
// improve performance for SIMD operations. The primary goal here is to avoid
// regression by ensuring that the arrays stay "well-aligned", even if the
// surrounding code is changed.
#[derive(Clone, Debug)]
#[repr(align(64))]
struct ScratchArray<L: Latent>([L; FULL_BATCH_N]);

impl<L: Latent> Deref for ScratchArray<L> {
  type Target = [L; FULL_BATCH_N];
  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
impl<L: Latent> DerefMut for ScratchArray<L> {
  fn deref_mut(&mut self) -> &mut Self::Target {
    &mut self.0
  }
}

// this is entirely state - any precomputed information is in the ChunkLatentDecompressor
#[derive(Clone, Debug)]
pub struct PageLatentDecompressor<L: Latent> {
  offset_bits_csum_scratch: ScratchArray<Bitlen>,
  offset_bits_scratch: ScratchArray<Bitlen>,
  lowers_scratch: ScratchArray<L>,

  ans_state_idxs: [AnsState; ANS_INTERLEAVING],
  delta_state: Vec<L>,
  delta_state_pos: usize,
}

impl<'a, L: Latent> PageLatentDecompressor<L> {
  pub fn new(
    cld: &'a ChunkLatentDecompressor<L>,
    ans_final_state_idxs: [AnsState; ANS_INTERLEAVING],
    stored_delta_state: Vec<L>,
  ) -> PcoResult<Self> {
    let (working_delta_state, delta_state_pos) = match cld.delta_encoding {
      LatentVarDeltaEncoding::NoOp | LatentVarDeltaEncoding::Consecutive(_) => {
        (stored_delta_state, 0)
      }
      LatentVarDeltaEncoding::Lookback(config) => {
        delta::new_lookback_window_buffer_and_pos(config, &stored_delta_state)
      }
    };

    let mut res = Self {
      offset_bits_csum_scratch: ScratchArray([0; FULL_BATCH_N]),
      offset_bits_scratch: ScratchArray([0; FULL_BATCH_N]),
      lowers_scratch: ScratchArray([L::ZERO; FULL_BATCH_N]),
      ans_state_idxs: ans_final_state_idxs,
      delta_state: working_delta_state,
      delta_state_pos,
    };

    if let Some(bin) = &cld.only_bin {
      // we optimize performance by setting state once and never again
      let mut csum = 0;
      for i in 0..FULL_BATCH_N {
        res.offset_bits_scratch[i] = bin.offset_bits;
        res.offset_bits_csum_scratch[i] = csum;
        res.lowers_scratch[i] = bin.lower;
        csum += bin.offset_bits;
      }
    }

    Ok(res)
  }

  #[inline]
  fn set_scratch(&mut self, i: usize, offset_bit_idx: Bitlen, offset_bits: Bitlen, lower: L) {
    unsafe {
      *self.offset_bits_csum_scratch.get_unchecked_mut(i) = offset_bit_idx;
      *self.offset_bits_scratch.get_unchecked_mut(i) = offset_bits;
      *self.lowers_scratch.get_unchecked_mut(i) = lower;
    };
  }

  // This implementation handles only a full batch, but is faster.
  #[inline(never)]
  unsafe fn decompress_full_ans_symbols(
    &mut self,
    reader: &mut BitReader,
    cld: &ChunkLatentDecompressor<L>,
  ) {
    // At each iteration, this loads a single u64 and has all ANS decoders
    // read a single symbol from it.
    // Therefore it requires that ANS_INTERLEAVING * MAX_BITS_PER_ANS <= 57.
    // Additionally, we're unpacking all ANS states using the fact that
    // ANS_INTERLEAVING == 4.
    let src = reader.src;
    let mut stale_byte_idx = reader.stale_byte_idx;
    let mut bits_past_byte = reader.bits_past_byte;
    let mut offset_bit_idx = 0;
    let [mut state_idx_0, mut state_idx_1, mut state_idx_2, mut state_idx_3] = self.ans_state_idxs;
    let ans_nodes = cld.decoder.nodes.as_slice();
    let lowers = cld.state_lowers.as_slice();
    for base_i in (0..FULL_BATCH_N).step_by(ANS_INTERLEAVING) {
      stale_byte_idx += bits_past_byte as usize / 8;
      bits_past_byte %= 8;
      let packed = bit_reader::u64_at(src, stale_byte_idx);
      // I hate that I have to do this with a macro, but it gives a serious
      // performance gain. If I use a [AnsState; 4] for the state_idxs instead
      // of separate identifiers, it tries to repeatedly load and write to
      // the array instead of keeping the states in registers.
      macro_rules! handle_single_symbol {
        ($j: expr, $state_idx: ident) => {
          let i = base_i + $j;
          let node = unsafe { ans_nodes.get_unchecked($state_idx as usize) };
          let bits_to_read = node.bits_to_read as Bitlen;
          let ans_val = (packed >> bits_past_byte) as AnsState & ((1 << bits_to_read) - 1);
          let lower = unsafe { *lowers.get_unchecked($state_idx as usize) };
          let offset_bits = node.offset_bits as Bitlen;
          self.set_scratch(i, offset_bit_idx, offset_bits, lower);
          bits_past_byte += bits_to_read;
          offset_bit_idx += offset_bits;
          $state_idx = node.next_state_idx_base as AnsState + ans_val;
        };
      }
      handle_single_symbol!(0, state_idx_0);
      handle_single_symbol!(1, state_idx_1);
      handle_single_symbol!(2, state_idx_2);
      handle_single_symbol!(3, state_idx_3);
    }

    reader.stale_byte_idx = stale_byte_idx;
    reader.bits_past_byte = bits_past_byte;
    self.ans_state_idxs = [state_idx_0, state_idx_1, state_idx_2, state_idx_3];
  }

  // This implementation handles arbitrary batch size and looks simpler, but is
  // slower, so we only use it at the end of the page.
  #[inline(never)]
  unsafe fn decompress_ans_symbols(
    &mut self,
    reader: &mut BitReader,
    cld: &ChunkLatentDecompressor<L>,
    batch_n: usize,
  ) {
    let src = reader.src;
    let mut stale_byte_idx = reader.stale_byte_idx;
    let mut bits_past_byte = reader.bits_past_byte;
    let mut offset_bit_idx = 0;
    let mut state_idxs = self.ans_state_idxs;
    for i in 0..batch_n {
      let j = i % ANS_INTERLEAVING;
      let state_idx = state_idxs[j] as usize;
      stale_byte_idx += bits_past_byte as usize / 8;
      bits_past_byte %= 8;
      let packed = bit_reader::u64_at(src, stale_byte_idx);
      let node = unsafe { cld.decoder.nodes.get_unchecked(state_idx) };
      let bits_to_read = node.bits_to_read as Bitlen;
      let ans_val = (packed >> bits_past_byte) as AnsState & ((1 << bits_to_read) - 1);
      let lower = unsafe { *cld.state_lowers.get_unchecked(state_idx) };
      let offset_bits = node.offset_bits as Bitlen;
      self.set_scratch(i, offset_bit_idx, offset_bits, lower);
      bits_past_byte += bits_to_read;
      offset_bit_idx += offset_bits;
      state_idxs[j] = node.next_state_idx_base as AnsState + ans_val;
    }

    reader.stale_byte_idx = stale_byte_idx;
    reader.bits_past_byte = bits_past_byte;
    self.ans_state_idxs = state_idxs;
  }

  #[inline(never)]
  unsafe fn decompress_offsets<const READ_BYTES: usize>(
    &mut self,
    reader: &mut BitReader,
    dst: &mut [L],
  ) {
    let base_bit_idx = reader.bit_idx();
    let src = reader.src;
    for (dst, (&offset_bits, (&offset_bits_csum, &lower))) in dst.iter_mut().zip(
      self.offset_bits_scratch.iter().zip(
        self
          .offset_bits_csum_scratch
          .iter()
          .zip(self.lowers_scratch.iter()),
      ),
    ) {
      let bit_idx = base_bit_idx as Bitlen + offset_bits_csum;
      let byte_idx = bit_idx / 8;
      let bits_past_byte = bit_idx % 8;
      let latent_minus_lower = bit_reader::read_uint_at::<L, READ_BYTES>(
        src,
        byte_idx as usize,
        bits_past_byte,
        offset_bits,
      );

      // On aarch64, lowers are added outside this loop for better SIMD; otherwise, add here.
      *dst = if cfg!(target_arch = "aarch64") {
        latent_minus_lower
      } else {
        latent_minus_lower.wrapping_add(lower)
      };
    }
    let final_bit_idx = base_bit_idx
      + self.offset_bits_csum_scratch[dst.len() - 1] as usize
      + self.offset_bits_scratch[dst.len() - 1] as usize;
    reader.stale_byte_idx = final_bit_idx / 8;
    reader.bits_past_byte = final_bit_idx as Bitlen % 8;

    // On aarch64, lower is added outside decompress_offsets loop for better SIMD.
    if cfg!(target_arch = "aarch64") {
      self.add_lowers(dst);
    }
  }

  #[inline(never)]
  fn add_lowers(&self, dst: &mut [L]) {
    for (dst, &lower) in dst.iter_mut().zip(self.lowers_scratch.iter()) {
      *dst = dst.wrapping_add(lower);
    }
  }

  // If hits a corruption, it returns an error and leaves reader and self unchanged.
  // May contaminate dst.
  pub unsafe fn decompress_batch_pre_delta(
    &mut self,
    reader: &mut BitReader,
    cld: &ChunkLatentDecompressor<L>,
    dst: &mut [L],
  ) {
    if dst.is_empty() {
      return;
    }

    if cld.n_bins > 1 {
      let batch_n = dst.len();
      assert!(batch_n <= FULL_BATCH_N);

      if batch_n == FULL_BATCH_N {
        self.decompress_full_ans_symbols(reader, cld);
      } else {
        self.decompress_ans_symbols(reader, cld, batch_n);
      }
    }

    // We want to read the offsets for each latent type as fast as possible.
    // Depending on the number of bits per offset, we can read them in
    // different chunk sizes. We use the smallest chunk size that can hold
    // the maximum possible offset.
    // The matching is intentionally verbose to make it clear how different
    // latent types are handled.
    // Note: Providing a 2 byte read appears to degrade performance for 16-bit
    // latents.
    match cld.bytes_per_offset {
      // all
      0 => dst.copy_from_slice(&self.lowers_scratch[..dst.len()]),

      // u16
      1..=4 if L::BITS == 16 => self.decompress_offsets::<4>(reader, dst),
      // u32
      1..=4 if L::BITS == 32 => self.decompress_offsets::<4>(reader, dst),
      5..=8 if L::BITS == 32 => self.decompress_offsets::<8>(reader, dst),
      // u64
      1..=8 if L::BITS == 64 => self.decompress_offsets::<8>(reader, dst),
      9..=15 if L::BITS == 64 => self.decompress_offsets::<15>(reader, dst),
      _ => panic!(
        "[PageLatentDecompressor] {} byte read not supported for {}-bit Latents",
        cld.bytes_per_offset,
        L::BITS
      ),
    }
  }

  pub unsafe fn decompress_batch(
    &mut self,
    reader: &mut BitReader,
    cld: &ChunkLatentDecompressor<L>,
    delta_latents: Option<&DynLatents>,
    n_remaining_in_page: usize,
    dst: &mut [L],
  ) -> PcoResult<()> {
    let n_remaining_pre_delta =
      n_remaining_in_page.saturating_sub(cld.delta_encoding.n_latents_per_state());
    let pre_delta_len = if dst.len() <= n_remaining_pre_delta {
      dst.len()
    } else {
      // If we're at the end, this won't initialize the last
      // few elements before delta decoding them, so we do that manually here to
      // satisfy MIRI. This step isn't really necessary.
      dst[n_remaining_pre_delta..].fill(L::default());
      n_remaining_pre_delta
    };
    self.decompress_batch_pre_delta(reader, cld, &mut dst[..pre_delta_len]);

    match cld.delta_encoding {
      LatentVarDeltaEncoding::NoOp => Ok(()),
      LatentVarDeltaEncoding::Consecutive(_) => {
        delta::decode_consecutive_in_place(&mut self.delta_state, dst);
        Ok(())
      }
      LatentVarDeltaEncoding::Lookback(config) => {
        let has_oob_lookbacks = delta::decode_with_lookbacks_in_place(
          config,
          delta_latents
            .unwrap()
            .downcast_ref::<DeltaLookback>()
            .unwrap(),
          &mut self.delta_state_pos,
          &mut self.delta_state,
          dst,
        );
        if has_oob_lookbacks {
          Err(PcoError::corruption(
            "delta lookback exceeded window n",
          ))
        } else {
          Ok(())
        }
      }
    }
  }
}

// Because the size of PageLatentDecompressor is enormous (largely due to
// scratch buffers), it makes more sense to allocate them on the heap. We only
// need to derefernce them once per batch, which is plenty infrequent.
// TODO: consider an arena for the scratch arrays?
type BoxedPageLatentDecompressor<L> = Box<PageLatentDecompressor<L>>;

define_latent_enum!(
  #[derive()]
  pub DynPageLatentDecompressor(BoxedPageLatentDecompressor)
);
