use std::cmp::min;

use crate::ans::{AnsState, Symbol};
use crate::compression_intermediates::DissectedPageVar;
use crate::compression_table::CompressionTable;
use crate::constants::{Bitlen, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::{ans, bits};

pub struct LatentBatchDissector<'a, L: Latent> {
  // immutable
  table: &'a CompressionTable<L>,
  encoder: &'a ans::Encoder,

  // mutable
  // TODO: use an arena and heap-allocate these?
  lower_scratch: [L; FULL_BATCH_N],
  symbol_scratch: [Symbol; FULL_BATCH_N],
}

impl<'a, L: Latent> LatentBatchDissector<'a, L> {
  pub fn new(table: &'a CompressionTable<L>, encoder: &'a ans::Encoder) -> Self {
    Self {
      table,
      encoder,
      lower_scratch: [L::ZERO; FULL_BATCH_N],
      symbol_scratch: [0; FULL_BATCH_N],
    }
  }

  #[inline(never)]
  fn binary_search(&self, latents: &[L]) -> [usize; FULL_BATCH_N] {
    let mut search_idxs = [0; FULL_BATCH_N];

    // we do this as `size_log` SIMD loops over the batch
    for depth in 0..self.table.search_size_log {
      let bisection_idx = 1 << (self.table.search_size_log - 1 - depth);
      for (&latent, search_idx) in latents.iter().zip(search_idxs.iter_mut()) {
        let candidate_idx = *search_idx + bisection_idx;
        let value = unsafe { *self.table.search_lowers.get_unchecked(candidate_idx) };
        *search_idx += ((latent >= value) as usize) * bisection_idx;
      }
    }

    let max_idx = self.table.infos.len().saturating_sub(1);
    search_idxs
      .iter_mut()
      .for_each(|search_idx| *search_idx = min(*search_idx, max_idx));

    search_idxs
  }

  #[inline(never)]
  fn dissect_bins(&mut self, search_idxs: &[usize], dst_offset_bits: &mut [Bitlen]) {
    for (i, &search_idx) in search_idxs.iter().enumerate() {
      let info = &self.table.infos[search_idx];
      self.lower_scratch[i] = info.lower;
      self.symbol_scratch[i] = info.symbol;
      dst_offset_bits[i] = info.offset_bits;
    }
  }

  #[inline(never)]
  fn set_offsets(&self, latents: &[L], offsets: &mut [L]) {
    for (offset, (&latent, &lower)) in offsets
      .iter_mut()
      .zip(latents.iter().zip(self.lower_scratch.iter()))
    {
      *offset = latent - lower;
    }
  }

  #[inline(never)]
  fn encode_ans_in_reverse(
    &self,
    ans_vals: &mut [AnsState],
    ans_bits: &mut [Bitlen],
    ans_final_states: &mut [AnsState; ANS_INTERLEAVING],
  ) {
    let final_base_i = (ans_vals.len() / ANS_INTERLEAVING) * ANS_INTERLEAVING;
    let final_j = ans_vals.len() % ANS_INTERLEAVING;

    // first get the jagged part out of the way
    for j in (0..final_j).rev() {
      let i = final_base_i + j;
      let (new_state, bitlen) = self
        .encoder
        .encode(ans_final_states[j], self.symbol_scratch[i]);
      ans_vals[i] = bits::lowest_bits_fast(ans_final_states[j], bitlen);
      ans_bits[i] = bitlen;
      ans_final_states[j] = new_state;
    }

    // then do the main loop
    for base_i in (0..final_base_i).step_by(ANS_INTERLEAVING).rev() {
      for j in (0..ANS_INTERLEAVING).rev() {
        let i = base_i + j;
        let (new_state, bitlen) = self
          .encoder
          .encode(ans_final_states[j], self.symbol_scratch[i]);
        ans_vals[i] = bits::lowest_bits_fast(ans_final_states[j], bitlen);
        ans_bits[i] = bitlen;
        ans_final_states[j] = new_state;
      }
    }
  }

  pub fn dissect_latent_batch(&mut self, latents: &[L], base_i: usize, dst: &mut DissectedPageVar) {
    let DissectedPageVar {
      ans_vals,
      ans_bits,
      offsets,
      offset_bits,
      ans_final_states,
    } = dst;

    let search_idxs = self.binary_search(latents);

    let end_i = min(base_i + FULL_BATCH_N, ans_vals.len());

    self.dissect_bins(
      &search_idxs[..latents.len()],
      &mut offset_bits[base_i..end_i],
    );

    let offsets = offsets.downcast_mut::<L>().unwrap();
    self.set_offsets(latents, &mut offsets[base_i..end_i]);

    self.encode_ans_in_reverse(
      &mut ans_vals[base_i..end_i],
      &mut ans_bits[base_i..end_i],
      ans_final_states,
    );
  }
}
