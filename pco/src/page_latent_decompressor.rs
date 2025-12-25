use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use crate::ans::AnsState;
use crate::bit_reader::BitReader;
use crate::chunk_latent_decompressor::ChunkLatentDecompressor;
use crate::constants::{Bitlen, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::dyn_latent_slice::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::macros::define_latent_enum;
use crate::metadata::delta_encoding::LatentVarDeltaEncoding;
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

#[inline(never)]
unsafe fn decompress_offsets<L: Latent, const READ_BYTES: usize>(
  reader: &mut BitReader,
  offset_bits_csum: &[u32],
  offset_bits: &[u32],
  latents: &mut [L],
  n: usize,
) {
  let base_bit_idx = reader.bit_idx();
  let src = reader.src;
  for i in 0..n {
    let offset_bits = offset_bits[i];
    let offset_bits_csum = offset_bits_csum[i];
    let bit_idx = base_bit_idx as Bitlen + offset_bits_csum;
    let byte_idx = bit_idx / 8;
    let bits_past_byte = bit_idx % 8;
    let offset = bit_reader::read_uint_at::<L, READ_BYTES>(
      src,
      byte_idx as usize,
      bits_past_byte,
      offset_bits,
    );

    latents[i] = latents[i].wrapping_add(offset);
  }

  let final_bit_idx = base_bit_idx + offset_bits_csum[n - 1] as usize + offset_bits[n - 1] as usize;
  reader.stale_byte_idx = final_bit_idx / 8;
  reader.bits_past_byte = final_bit_idx as Bitlen % 8;
}

// Here we do something very strange to ensure vectorization of
// decompress_offsets on aarch64: we force specializations to be exported by the
// pco compiled library, as opposed to downstream compilation units. I'm not
// sure why this changes vectorization rules, but it's a significant speedup.
macro_rules! force_export {
  ($name: ident, $l: ty, $rb: literal) => {
    #[used]
    static $name: unsafe fn(&mut BitReader, &[u32], &[u32], &mut [$l], usize) =
      decompress_offsets::<$l, $rb>;
  };
}
force_export!(_FORCE_EXPORT_U16_4, u16, 4);
force_export!(_FORCE_EXPORT_U32_4, u32, 4);
force_export!(_FORCE_EXPORT_U32_8, u32, 8);
force_export!(_FORCE_EXPORT_U64_8, u64, 8);

// this is entirely state - any precomputed information is in the ChunkLatentDecompressor
#[derive(Clone, Debug)]
pub struct PageLatentDecompressor<L: Latent> {
  offset_bits_csum_scratch: ScratchArray<Bitlen>,
  offset_bits_scratch: ScratchArray<Bitlen>,
  latents: ScratchArray<L>,

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
      latents: ScratchArray([L::ZERO; FULL_BATCH_N]),
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
        res.latents[i] = bin.lower;
        csum += bin.offset_bits;
      }
    }

    Ok(res)
  }

  #[inline]
  unsafe fn set_scratch(
    &mut self,
    i: usize,
    offset_bit_idx: Bitlen,
    offset_bits: Bitlen,
    lower: L,
  ) {
    *self.offset_bits_csum_scratch.get_unchecked_mut(i) = offset_bit_idx;
    *self.offset_bits_scratch.get_unchecked_mut(i) = offset_bits;
    *self.latents.get_unchecked_mut(i) = lower;
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

  // If hits a corruption, it returns an error and leaves reader and self unchanged.
  // May contaminate dst.
  pub unsafe fn decompress_batch_pre_delta(
    &mut self,
    reader: &mut BitReader,
    cld: &ChunkLatentDecompressor<L>,
    batch_n: usize,
  ) {
    if batch_n == 0 {
      return;
    }

    assert!(batch_n <= FULL_BATCH_N);
    if cld.n_bins > 1 {
      assert!(batch_n <= FULL_BATCH_N);

      if batch_n == FULL_BATCH_N {
        self.decompress_full_ans_symbols(reader, cld);
      } else {
        self.decompress_ans_symbols(reader, cld, batch_n);
      }
    } else {
      self.latents[..batch_n].fill(cld.state_lowers[0]);
    }

    // We want to read the offsets for each latent type as fast as possible.
    // Depending on the number of bits per offset, we can read them in
    // different chunk sizes. We use the smallest chunk size that can hold
    // the maximum possible offset.
    // The matching is intentionally verbose to make it clear how different
    // latent types are handled.
    // Note: Providing a 2 byte read appears to degrade performance for 16-bit
    // latents.
    macro_rules! specialized_decompress_offsets {
      ($rb: literal) => {
        decompress_offsets::<L, $rb>(
          reader,
          &self.offset_bits_csum_scratch.0,
          &self.offset_bits_scratch.0,
          &mut self.latents.0,
          batch_n,
        )
      };
    }
    match (cld.bytes_per_offset, L::BITS) {
      (0, _) => (),
      (1..=4, 16) => specialized_decompress_offsets!(4),
      (1..=4, 32) => specialized_decompress_offsets!(4),
      (5..=8, 32) => specialized_decompress_offsets!(8),
      (1..=8, 64) => specialized_decompress_offsets!(8),
      (9..=15, 64) => specialized_decompress_offsets!(15),
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
    delta_latents: Option<DynLatentSlice>,
    n_remaining_in_page: usize,
  ) -> PcoResult<()> {
    let n_remaining_pre_delta =
      n_remaining_in_page.saturating_sub(cld.delta_encoding.n_latents_per_state());
    let pre_delta_len = FULL_BATCH_N.min(n_remaining_pre_delta);
    self.decompress_batch_pre_delta(reader, cld, pre_delta_len);
    let dst = &mut self.latents[..n_remaining_in_page.min(FULL_BATCH_N)];

    match cld.delta_encoding {
      LatentVarDeltaEncoding::NoOp => Ok(()),
      LatentVarDeltaEncoding::Consecutive(_) => {
        delta::decode_consecutive_in_place(&mut self.delta_state, dst);
        Ok(())
      }
      LatentVarDeltaEncoding::Lookback(config) => {
        let Some(DynLatentSlice::U32(lookbacks)) = delta_latents else {
          unreachable!()
        };
        let has_oob_lookbacks = delta::decode_with_lookbacks_in_place(
          config,
          lookbacks,
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
// TODO: consider an arena for these?
type Boxed<L> = Box<PageLatentDecompressor<L>>;

define_latent_enum!(
  #[derive()]
  pub DynPageLatentDecompressor(Boxed)
);

impl DynPageLatentDecompressor {
  pub fn latents<'a>(&'a mut self) -> DynLatentSlice<'a> {
    match self {
      Self::U16(inner) => DynLatentSlice::U16(&mut *inner.latents),
      Self::U32(inner) => DynLatentSlice::U32(&mut *inner.latents),
      Self::U64(inner) => DynLatentSlice::U64(&mut *inner.latents),
    }
  }
}
