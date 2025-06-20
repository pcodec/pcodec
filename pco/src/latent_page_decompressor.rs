use std::fmt::Debug;

use crate::ans::{AnsState, Spec};
use crate::bit_reader::BitReader;
use crate::constants::{Bitlen, DeltaLookback, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::errors::{PcoError, PcoResult};
use crate::macros::define_latent_enum;
use crate::metadata::{bins, Bin, DeltaEncoding, DynLatents};
use crate::{ans, bit_reader, delta, read_write_uint};

#[derive(Clone, Debug)]
struct State<L: Latent> {
  // scratch needs no backup
  offset_bits_csum_scratch: [Bitlen; FULL_BATCH_N],
  offset_bits_scratch: [Bitlen; FULL_BATCH_N],
  lowers_scratch: [L; FULL_BATCH_N],

  ans_state_idxs: [AnsState; ANS_INTERLEAVING],
  delta_state: Vec<L>,
  delta_state_pos: usize,
}

impl<L: Latent> State<L> {
  #[inline]
  fn set_scratch(&mut self, i: usize, offset_bit_idx: Bitlen, offset_bits: Bitlen, lower: L) {
    unsafe {
      *self.offset_bits_csum_scratch.get_unchecked_mut(i) = offset_bit_idx;
      *self.offset_bits_scratch.get_unchecked_mut(i) = offset_bits;
      *self.lowers_scratch.get_unchecked_mut(i) = lower;
    };
  }
}

// LatentBatchDecompressor does the main work of decoding bytes into Latents
#[derive(Clone, Debug)]
pub struct LatentPageDecompressor<L: Latent> {
  // known information about this latent variable
  u64s_per_offset: usize,
  bin_lowers: Vec<L>,
  needs_ans: bool,
  decoder: ans::Decoder,
  delta_encoding: DeltaEncoding,
  pub maybe_constant_value: Option<L>,

  // mutable state
  state: State<L>,
}

impl<L: Latent> LatentPageDecompressor<L> {
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
    let bin_lowers = self.bin_lowers.as_slice();
    let ans_nodes = self.decoder.nodes.as_slice();
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
          let lower = *unsafe { bin_lowers.get_unchecked(node.symbol as usize) };
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
      stale_byte_idx += bits_past_byte as usize / 8;
      bits_past_byte %= 8;
      let packed = bit_reader::u64_at(src, stale_byte_idx);
      let node = unsafe { self.decoder.nodes.get_unchecked(state_idxs[j] as usize) };
      let bits_to_read = node.bits_to_read as Bitlen;
      let ans_val = (packed >> bits_past_byte) as AnsState & ((1 << bits_to_read) - 1);
      let lower = self.bin_lowers[node.symbol as usize];
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

  #[inline(never)]
  unsafe fn decompress_offsets<const MAX_U64S: usize>(
    &mut self,
    reader: &mut BitReader,
    dst: &mut [L],
  ) {
    let base_bit_idx = reader.bit_idx();
    let src = reader.src;
    let state = &mut self.state;
    for (dst, (&offset_bits, &offset_bits_csum)) in dst.iter_mut().zip(
      state
        .offset_bits_scratch
        .iter()
        .zip(state.offset_bits_csum_scratch.iter()),
    ) {
      let bit_idx = base_bit_idx + offset_bits_csum as usize;
      let byte_idx = bit_idx / 8;
      let bits_past_byte = bit_idx as Bitlen % 8;
      *dst = bit_reader::read_uint_at::<L, MAX_U64S>(src, byte_idx, bits_past_byte, offset_bits);
    }
    let final_bit_idx = base_bit_idx
      + state.offset_bits_csum_scratch[dst.len() - 1] as usize
      + state.offset_bits_scratch[dst.len() - 1] as usize;
    reader.stale_byte_idx = final_bit_idx / 8;
    reader.bits_past_byte = final_bit_idx as Bitlen % 8;
  }

  #[inline(never)]
  fn add_lowers(&self, dst: &mut [L]) {
    for (&lower, dst) in self.state.lowers_scratch[0..dst.len()]
      .iter()
      .zip(dst.iter_mut())
    {
      *dst = dst.wrapping_add(lower);
    }
  }

  // If hits a corruption, it returns an error and leaves reader and self unchanged.
  // May contaminate dst.
  pub unsafe fn decompress_batch_pre_delta(&mut self, reader: &mut BitReader, dst: &mut [L]) {
    if dst.is_empty() {
      return;
    }

    if self.needs_ans {
      let batch_n = dst.len();
      assert!(batch_n <= FULL_BATCH_N);

      if batch_n == FULL_BATCH_N {
        self.decompress_full_ans_symbols(reader);
      } else {
        self.decompress_ans_symbols(reader, batch_n);
      }
    }

    // this assertion saves some unnecessary specializations in the compiled assembly
    assert!(self.u64s_per_offset <= read_write_uint::calc_max_u64s(L::BITS));
    match self.u64s_per_offset {
      0 => {
        dst.copy_from_slice(&self.state.lowers_scratch[..dst.len()]);
        return;
      }
      1 => self.decompress_offsets::<1>(reader, dst),
      2 => self.decompress_offsets::<2>(reader, dst),
      3 => self.decompress_offsets::<3>(reader, dst),
      _ => panic!(
        "[LatentBatchDecompressor] data type too large (extra u64's {} > 2)",
        self.u64s_per_offset
      ),
    }

    self.add_lowers(dst);
  }

  pub unsafe fn decompress_batch(
    &mut self,
    delta_latents: Option<&DynLatents>,
    n_remaining_in_page: usize,
    reader: &mut BitReader,
    dst: &mut [L],
  ) -> PcoResult<()> {
    let n_remaining_pre_delta =
      n_remaining_in_page.saturating_sub(self.delta_encoding.n_latents_per_state());
    let pre_delta_len = if dst.len() <= n_remaining_pre_delta {
      dst.len()
    } else {
      // If we're at the end, this won't initialize the last
      // few elements before delta decoding them, so we do that manually here to
      // satisfy MIRI. This step isn't really necessary.
      dst[n_remaining_pre_delta..].fill(L::default());
      n_remaining_pre_delta
    };
    self.decompress_batch_pre_delta(reader, &mut dst[..pre_delta_len]);

    match self.delta_encoding {
      DeltaEncoding::None => Ok(()),
      DeltaEncoding::Consecutive(_) => {
        delta::decode_consecutive_in_place(&mut self.state.delta_state, dst);
        Ok(())
      }
      DeltaEncoding::Lookback(config) => {
        let has_oob_lookbacks = delta::decode_with_lookbacks_in_place(
          config,
          delta_latents
            .unwrap()
            .downcast_ref::<DeltaLookback>()
            .unwrap(),
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

// Because the size of LatentPageDecompressor is enormous (largely due to
// scratch buffers), it makes more sense to allocate them on the heap. We only
// need to derefernce them once per batch, which is plenty infrequent.
// TODO: consider an arena for these?
type BoxedLatentPageDecompressor<L> = Box<LatentPageDecompressor<L>>;

define_latent_enum!(
  #[derive()]
  pub DynLatentPageDecompressor(BoxedLatentPageDecompressor)
);

impl DynLatentPageDecompressor {
  pub fn create<L: Latent>(
    ans_size_log: Bitlen,
    bins: &[Bin<L>],
    delta_encoding: DeltaEncoding,
    ans_final_state_idxs: [AnsState; ANS_INTERLEAVING],
    stored_delta_state: Vec<L>,
  ) -> PcoResult<Self> {
    let u64s_per_offset = read_write_uint::calc_max_u64s(bins::max_offset_bits(bins));
    let bin_lowers = bins.iter().map(|bin| bin.lower).collect();
    let bin_offset_bits = bins.iter().map(|bin| bin.offset_bits).collect::<Vec<_>>();
    let weights = bins::weights(bins);
    let ans_spec = Spec::from_weights(ans_size_log, weights)?;
    let decoder = ans::Decoder::new(&ans_spec, &bin_offset_bits);

    let (working_delta_state, delta_state_pos) = match delta_encoding {
      DeltaEncoding::None | DeltaEncoding::Consecutive(_) => (stored_delta_state, 0),
      DeltaEncoding::Lookback(config) => {
        delta::new_lookback_window_buffer_and_pos(config, &stored_delta_state)
      }
    };

    let mut state = State {
      offset_bits_csum_scratch: [0; FULL_BATCH_N],
      offset_bits_scratch: [0; FULL_BATCH_N],
      lowers_scratch: [L::ZERO; FULL_BATCH_N],
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
        state.lowers_scratch[i] = bin.lower;
        csum += bin.offset_bits;
      }
    }

    let maybe_constant_value =
      if bins::are_trivial(bins) && matches!(delta_encoding, DeltaEncoding::None) {
        bins.first().map(|bin| bin.lower)
      } else {
        None
      };

    let lpd = LatentPageDecompressor {
      u64s_per_offset,
      bin_lowers,
      needs_ans,
      decoder,
      delta_encoding,
      maybe_constant_value,
      state,
    };
    Ok(Self::new(Box::new(lpd)).unwrap())
  }
}
