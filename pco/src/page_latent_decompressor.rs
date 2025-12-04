use std::fmt::Debug;
use std::mem;

use crate::ans::{AnsState, Spec};
use crate::bit_reader::BitReader;
use crate::constants::{Bitlen, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::dyn_latent_slice::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::macros::{define_latent_enum, match_latent_enum};
use crate::metadata::{bins, Bin, DeltaEncoding};
use crate::scratch_array::{DynScratchArray, ScratchArray};
use crate::{ans, bit_reader, delta, read_write_uint};

#[derive(Clone, Debug)]
struct State<L: Latent> {
  offset_bits_csum_scratch: ScratchArray<Bitlen>,
  offset_bits_scratch: ScratchArray<Bitlen>,

  ans_state_idxs: [AnsState; ANS_INTERLEAVING],
  delta_state: Vec<L>,
  delta_state_pos: usize,
}

impl<L: Latent> State<L> {
  #[inline]
  unsafe fn set_scratch(&mut self, i: usize, offset_bit_idx: Bitlen, offset_bits: Bitlen) {
    *self.offset_bits_csum_scratch.get_unchecked_mut(i) = offset_bit_idx;
    *self.offset_bits_scratch.get_unchecked_mut(i) = offset_bits;
  }
}

#[derive(Clone, Copy, Debug)]
pub enum WriteTo {
  Scratch,
  Dst,
}

#[derive(Clone, Debug)]
pub struct PageLatentDecompressor<L: Latent> {
  // known information about this latent variable
  bytes_per_offset: usize,
  state_lowers: Vec<L>,
  needs_ans: bool,
  maybe_constant_latent: Option<L>,
  decoder: ans::Decoder,
  delta_encoding: DeltaEncoding,

  // mutable state
  state: State<L>,
}

// It would be somewhat easier to implement this with generics, but we find that
// doing so fails to vectorize for some combinations of platform and data type
// (e.g. single u64 reads on aarch64). The following code has been written very
// carefully to compile well on all architectures, so be careful changing it.
macro_rules! impl_specialized_decompress_offsets {
  ($f:ident, $t:ty, $read:ident) => {
    #[inline(never)]
    unsafe fn $f(
      src: &[u8],
      base_bit_idx: u32,
      offset_bits_csum: &[u32],
      offset_bits: &[u32],
      dst: &mut [$t],
    ) {
      let batch_n = dst.len();
      for i in 0..batch_n {
        let bit_idx = base_bit_idx + offset_bits_csum[i];
        let byte_idx = bit_idx / 8;
        let bits_past_byte = bit_idx % 8;
        dst[i] = dst[i].wrapping_add(bit_reader::$read(
          src,
          byte_idx as usize,
          bits_past_byte,
          offset_bits[i],
        ) as $t);
      }
    }
  };
}
impl_specialized_decompress_offsets!(decompress_offsets_u16_4, u16, read_u32_at);
impl_specialized_decompress_offsets!(decompress_offsets_u32_4, u32, read_u32_at);
impl_specialized_decompress_offsets!(decompress_offsets_u32_8, u32, read_u64_at);
impl_specialized_decompress_offsets!(decompress_offsets_u64_8, u64, read_u64_at);
impl_specialized_decompress_offsets!(
  decompress_offsets_u64_15,
  u64,
  read_wide_u64_at
);

impl<L: Latent> PageLatentDecompressor<L> {
  pub fn make_external_scratch(&self) -> ScratchArray<L> {
    ScratchArray([self.maybe_constant_latent.unwrap_or_default(); FULL_BATCH_N])
  }

  // This implementation handles only a full batch, but is faster.
  #[inline(never)]
  unsafe fn decompress_full_ans_symbols(&mut self, reader: &mut BitReader, dst: &mut [L]) {
    // At each iteration, this loads a single u64 and has all ANS decoders
    // read a single symbol from it.
    // Therefore it requires that ANS_INTERLEAVING * MAX_BITS_PER_ANS <= 57.
    // Additionally, we're unpacking all ANS states using the fact that
    // ANS_INTERLEAVING == 4.
    let src = reader.src;
    let mut stale_byte_idx = reader.stale_byte_idx;
    let mut bits_past_byte = reader.bits_past_byte;
    let mut offset_bit_idx = 0;
    let [mut state_idx_0, mut state_idx_1, mut state_idx_2, mut state_idx_3] =
      self.state.ans_state_idxs;
    let ans_nodes = self.decoder.nodes.as_slice();
    let lowers = self.state_lowers.as_slice();
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
          self.state.set_scratch(i, offset_bit_idx, offset_bits);
          *dst.get_unchecked_mut(i) = lower;
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
    self.state.ans_state_idxs = [state_idx_0, state_idx_1, state_idx_2, state_idx_3];
  }

  // This implementation handles arbitrary batch size and looks simpler, but is
  // slower, so we only use it at the end of the page.
  #[inline(never)]
  unsafe fn decompress_ans_symbols(
    &mut self,
    reader: &mut BitReader,
    batch_n: usize,
    dst: &mut [L],
  ) {
    let src = reader.src;
    let mut stale_byte_idx = reader.stale_byte_idx;
    let mut bits_past_byte = reader.bits_past_byte;
    let mut offset_bit_idx = 0;
    let mut state_idxs = self.state.ans_state_idxs;
    for i in 0..batch_n {
      let j = i % ANS_INTERLEAVING;
      let state_idx = state_idxs[j] as usize;
      stale_byte_idx += bits_past_byte as usize / 8;
      bits_past_byte %= 8;
      let packed = bit_reader::u64_at(src, stale_byte_idx);
      let node = unsafe { self.decoder.nodes.get_unchecked(state_idx) };
      let bits_to_read = node.bits_to_read as Bitlen;
      let ans_val = (packed >> bits_past_byte) as AnsState & ((1 << bits_to_read) - 1);
      let lower = unsafe { *self.state_lowers.get_unchecked(state_idx) };
      let offset_bits = node.offset_bits as Bitlen;
      self.state.set_scratch(i, offset_bit_idx, offset_bits);
      *dst.get_unchecked_mut(i) = lower;
      bits_past_byte += bits_to_read;
      offset_bit_idx += offset_bits;
      state_idxs[j] = node.next_state_idx_base as AnsState + ans_val;
    }

    reader.stale_byte_idx = stale_byte_idx;
    reader.bits_past_byte = bits_past_byte;
    self.state.ans_state_idxs = state_idxs;
  }

  // If hits a corruption, it returns an error and leaves reader and self unchanged.
  // May contaminate dst.
  pub unsafe fn decompress_batch_pre_delta(&mut self, reader: &mut BitReader, dst: &mut [L]) {
    let batch_n = dst.len();
    if batch_n == 0 {
      return;
    }

    assert!(batch_n <= FULL_BATCH_N);
    if self.needs_ans {
      if batch_n == FULL_BATCH_N {
        self.decompress_full_ans_symbols(reader, dst);
      } else {
        self.decompress_ans_symbols(reader, batch_n, dst);
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
    let base_bit_idx = reader.bit_idx();
    macro_rules! run {
      ($specialization: ident, $t: ty) => {
        $specialization(
          &reader.src,
          base_bit_idx as u32,
          &self.state.offset_bits_csum_scratch.0,
          &self.state.offset_bits_scratch.0,
          mem::transmute::<&mut [L], &mut [$t]>(dst),
        )
      };
    }
    match (self.bytes_per_offset, L::BITS) {
      (0, _) => (),
      (1..=4, 16) => run!(decompress_offsets_u16_4, u16),
      (1..=4, 32) => run!(decompress_offsets_u32_4, u32),
      (5..=8, 32) => run!(decompress_offsets_u32_8, u32),
      (1..=8, 64) => run!(decompress_offsets_u64_8, u64),
      (9..=15, 64) => run!(decompress_offsets_u64_15, u64),
      _ => panic!(
        "[PageLatentDecompressor] {} byte read not supported for {}-bit Latents",
        self.bytes_per_offset,
        L::BITS
      ),
    }
    let state = &mut self.state;
    let final_bit_idx = base_bit_idx
      + state.offset_bits_csum_scratch[batch_n - 1] as usize
      + state.offset_bits_scratch[batch_n - 1] as usize;
    reader.stale_byte_idx = final_bit_idx / 8;
    reader.bits_past_byte = final_bit_idx as Bitlen % 8;
  }

  pub unsafe fn decompress_batch(
    &mut self,
    delta_latents: &Option<DynLatentSlice>,
    reader: &mut BitReader,
    n_remaining_in_page: usize,
    dst: &mut [L],
  ) -> PcoResult<()> {
    // The data flow here is complicated and worth explaining. A PLD may be
    // configured to write to either its own internal buffer (latents) or a
    // provided dst. AND decoding always write to the internal buffers; offsets
    // get written to the ultimate destination, and then delta encoding is done
    // in place on the ultimate destination. If ANS or offset decoding or both
    // are trivial, we can skip steps or fill with a constant value.
    let pre_delta_limit =
      n_remaining_in_page.saturating_sub(self.delta_encoding.n_latents_per_state());
    let pre_delta_len = dst.len().min(pre_delta_limit);
    self.decompress_batch_pre_delta(reader, &mut dst[..pre_delta_len]);

    match self.delta_encoding {
      DeltaEncoding::None => Ok(()),
      DeltaEncoding::Consecutive(_) => {
        delta::decode_consecutive_in_place(&mut self.state.delta_state, dst);
        Ok(())
      }
      DeltaEncoding::Lookback(config) => {
        let Some(DynLatentSlice::U32(lookbacks)) = delta_latents else {
          unreachable!()
        };
        let has_oob_lookbacks = delta::decode_with_lookbacks_in_place(
          config,
          lookbacks,
          &mut self.state.delta_state_pos,
          &mut self.state.delta_state,
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
type BoxedPageLatentDecompressor<L> = Box<PageLatentDecompressor<L>>;

define_latent_enum!(
  #[derive()]
  pub DynPageLatentDecompressor(BoxedPageLatentDecompressor)
);

impl DynPageLatentDecompressor {
  pub fn create<L: Latent>(
    ans_size_log: Bitlen,
    bins: &[Bin<L>],
    delta_encoding: DeltaEncoding,
    ans_final_state_idxs: [AnsState; ANS_INTERLEAVING],
    stored_delta_state: Vec<L>,
  ) -> PcoResult<Self> {
    let bytes_per_offset = read_write_uint::calc_max_bytes(bins::max_offset_bits(bins));
    let bin_offset_bits = bins.iter().map(|bin| bin.offset_bits).collect::<Vec<_>>();
    let weights = bins::weights(bins);
    let ans_spec = Spec::from_weights(ans_size_log, weights)?;
    let state_lowers = ans_spec
      .state_symbols
      .iter()
      .map(|&s| bins.get(s as usize).map_or(L::ZERO, |b| b.lower))
      .collect();
    let decoder = ans::Decoder::new(&ans_spec, &bin_offset_bits);

    let (working_delta_state, delta_state_pos) = match delta_encoding {
      DeltaEncoding::None | DeltaEncoding::Consecutive(_) => (stored_delta_state, 0),
      DeltaEncoding::Lookback(config) => {
        delta::new_lookback_window_buffer_and_pos(config, &stored_delta_state)
      }
    };

    let mut state = State {
      offset_bits_csum_scratch: ScratchArray([0; FULL_BATCH_N]),
      offset_bits_scratch: ScratchArray([0; FULL_BATCH_N]),
      ans_state_idxs: ans_final_state_idxs,
      delta_state: working_delta_state,
      delta_state_pos,
    };

    let needs_ans = bins.len() != 1;
    if !needs_ans {
      // we optimize performance by setting state once and never again
      let bin = &bins[0];
      let mut csum = 0;
      for i in 0..FULL_BATCH_N {
        state.offset_bits_scratch[i] = bin.offset_bits;
        state.offset_bits_csum_scratch[i] = csum;
        csum += bin.offset_bits;
      }
    }

    let maybe_constant_latent =
      if bins::are_trivial(bins) && matches!(delta_encoding, DeltaEncoding::None) {
        Some(bins[0].lower)
      } else {
        None
      };

    let pld = PageLatentDecompressor {
      bytes_per_offset,
      state_lowers,
      needs_ans,
      maybe_constant_latent,
      decoder,
      delta_encoding,
      state,
    };
    Ok(Self::new(Box::new(pld)).unwrap())
  }

  pub fn make_external_scratch(&self) -> DynScratchArray {
    match_latent_enum!(
      self,
      DynPageLatentDecompressor<L>(inner) => {
        DynScratchArray::new(inner.make_external_scratch()).unwrap()
      }
    )
  }
}
