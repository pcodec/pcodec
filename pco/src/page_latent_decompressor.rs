use std::fmt::Debug;
use std::mem;
use std::ops::{Deref, DerefMut};

use crate::ans::{AnsState, Spec};
use crate::bit_reader::BitReader;
use crate::constants::{Bitlen, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::dyn_latent_slice::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::macros::define_latent_enum;
use crate::metadata::{bins, Bin, DeltaEncoding};
use crate::{ans, bit_reader, delta, read_write_uint};

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

#[derive(Clone, Debug)]
struct State<L: Latent> {
  // scratch needs no backup
  offset_bits_csum_scratch: ScratchArray<Bitlen>,
  offset_bits_scratch: ScratchArray<Bitlen>,
  latents: ScratchArray<L>,

  ans_state_idxs: [AnsState; ANS_INTERLEAVING],
  delta_state: Vec<L>,
  delta_state_pos: usize,
}

impl<L: Latent> State<L> {
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
}

#[derive(Clone, Debug)]
pub struct PageLatentDecompressor<L: Latent> {
  // known information about this latent variable
  bytes_per_offset: usize,
  state_lowers: Vec<L>,
  needs_ans: bool,
  is_constant: bool,
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
      latents: &mut [$t],
    ) {
      for i in 0..256 {
        let bit_idx = base_bit_idx + offset_bits_csum[i];
        let byte_idx = bit_idx / 8;
        let bits_past_byte = bit_idx % 8;
        latents[i] = latents[i].wrapping_add(bit_reader::$read(
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
  // This implementation handles only a full batch, but is faster.
  #[inline(never)]
  unsafe fn decompress_full_ans_symbols(&mut self, reader: &mut BitReader) {
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
          self
            .state
            .set_scratch(i, offset_bit_idx, offset_bits, lower);
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
  unsafe fn decompress_ans_symbols(&mut self, reader: &mut BitReader, batch_n: usize) {
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
      self
        .state
        .set_scratch(i, offset_bit_idx, offset_bits, lower);
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
  pub unsafe fn decompress_batch_pre_delta(&mut self, reader: &mut BitReader, batch_n: usize) {
    if batch_n == 0 || self.is_constant {
      return;
    }

    assert!(batch_n <= FULL_BATCH_N);
    if self.needs_ans {
      if batch_n == FULL_BATCH_N {
        self.decompress_full_ans_symbols(reader);
      } else {
        self.decompress_ans_symbols(reader, batch_n);
      }
    } else {
      self.state.latents[..batch_n].fill(self.state_lowers[0]);
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
    let bbi32 = base_bit_idx as u32;
    macro_rules! run {
      ($specialization: ident) => {
        $specialization(
          &reader.src,
          bbi32,
          &self.state.offset_bits_csum_scratch.0,
          &self.state.offset_bits_scratch.0,
          mem::transmute(self.state.latents.0.as_mut_slice()),
        )
      };
    }
    match self.bytes_per_offset {
      0 => (),
      1..=4 if L::BITS == 16 => run!(decompress_offsets_u16_4),
      1..=4 if L::BITS == 32 => run!(decompress_offsets_u32_4),
      5..=8 if L::BITS == 32 => run!(decompress_offsets_u32_8),
      1..=8 if L::BITS == 64 => run!(decompress_offsets_u64_8),
      9..=15 if L::BITS == 64 => run!(decompress_offsets_u64_15),
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
    limit: usize,
  ) -> PcoResult<()> {
    let n_remaining_pre_delta =
      n_remaining_in_page.saturating_sub(self.delta_encoding.n_latents_per_state());
    let pre_delta_len = limit.min(n_remaining_pre_delta);
    self.decompress_batch_pre_delta(reader, pre_delta_len);
    let dst = &mut self.state.latents[..limit];

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
      latents: ScratchArray([L::ZERO; FULL_BATCH_N]),
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
        state.latents[i] = bin.lower;
        csum += bin.offset_bits;
      }
    }

    let is_constant = bins::are_trivial(bins) && matches!(delta_encoding, DeltaEncoding::None);

    let pld = PageLatentDecompressor {
      bytes_per_offset,
      state_lowers,
      needs_ans,
      is_constant,
      decoder,
      delta_encoding,
      state,
    };
    Ok(Self::new(Box::new(pld)).unwrap())
  }

  pub fn latents<'a>(&'a mut self) -> DynLatentSlice<'a> {
    match self {
      Self::U16(inner) => DynLatentSlice::U16(&mut *inner.state.latents),
      Self::U32(inner) => DynLatentSlice::U32(&mut *inner.state.latents),
      Self::U64(inner) => DynLatentSlice::U64(&mut *inner.state.latents),
    }
  }
}
