use std::cmp::min;

use crate::ans::{AnsState, Symbol};
use crate::compression_intermediates::PageDissectedVar;
use crate::compression_table::CompressionTable;
use crate::constants::{Bitlen, ANS_INTERLEAVING, FULL_BATCH_N};
use crate::data_types::Latent;
use crate::metadata::DynLatents;
use crate::{ans, bits};

// TODO: We could reduce max memory usage by bit packing each batch immediately
// rather than wait to write each page out. Requires some care to avoid
// repeatedly allocating on the heap.
pub struct ChunkLatentDissector<'a, L: Latent> {
  table: &'a CompressionTable<L>,
  encoder: &'a ans::Encoder,
  default_lower: L,
}

struct Scratch<L: Latent> {
  lowers: [L; FULL_BATCH_N],
  symbols: [Symbol; FULL_BATCH_N],
}

unsafe fn uninit_vec<T>(n: usize) -> Vec<T> {
  let mut res = Vec::with_capacity(n);
  res.set_len(n);
  res
}

impl<'a, L: Latent> ChunkLatentDissector<'a, L> {
  pub fn new(table: &'a CompressionTable<L>, encoder: &'a ans::Encoder) -> Self {
    // We initialize the scratch buffer for bin lowers carefully to enable
    // a shortcut when there's only one bin.
    // For symbol scratch we initialize to zeros, which also happens to be
    // correct when there's only one bin.
    let default_lower = table.only_bin().map(|info| info.lower).unwrap_or(L::ZERO);
    Self {
      table,
      encoder,
      default_lower,
    }
  }

  fn build_scratch(&self) -> Scratch<L> {
    Scratch {
      lowers: [self.default_lower; FULL_BATCH_N],
      symbols: [0; FULL_BATCH_N],
    }
  }

  #[inline(never)]
  fn dissect_bins(
    &self,
    search_idxs: &[usize],
    scratch: &mut Scratch<L>,
    dst_offset_bits: &mut [Bitlen],
  ) {
    if self.table.is_trivial() {
      // trivial case: there's at most one bin. We've prepopulated the scratch
      // buffers with the correct values in this case.
      let default_offset_bits = self
        .table
        .only_bin()
        .map(|info| info.offset_bits)
        .unwrap_or(0);
      dst_offset_bits.fill(default_offset_bits);
      return;
    }

    for (i, &search_idx) in search_idxs.iter().enumerate() {
      let info = &self.table.infos[search_idx];
      scratch.lowers[i] = info.lower;
      scratch.symbols[i] = info.symbol;
      dst_offset_bits[i] = info.offset_bits;
    }
  }

  #[inline(never)]
  fn set_offsets(&self, latents: &[L], scratch: &mut Scratch<L>, offsets: &mut [L]) {
    for (offset, (&latent, &lower)) in offsets
      .iter_mut()
      .zip(latents.iter().zip(scratch.lowers.iter()))
    {
      *offset = latent - lower;
    }
  }

  #[inline(never)]
  fn encode_ans_in_reverse(
    &self,
    scratch: &mut Scratch<L>,
    ans_vals: &mut [AnsState],
    ans_bits: &mut [Bitlen],
    ans_final_states: &mut [AnsState; ANS_INTERLEAVING],
  ) {
    if self.encoder.size_log() == 0 {
      // trivial case: there's only one symbol. ANS values and states don't
      // matter.
      ans_bits.fill(0);
      return;
    }

    let final_base_i = (ans_vals.len() / ANS_INTERLEAVING) * ANS_INTERLEAVING;
    let final_j = ans_vals.len() % ANS_INTERLEAVING;

    // first get the jagged part out of the way
    for j in (0..final_j).rev() {
      let i = final_base_i + j;
      let (new_state, bitlen) = self.encoder.encode(ans_final_states[j], scratch.symbols[i]);
      ans_vals[i] = bits::lowest_bits_fast(ans_final_states[j], bitlen);
      ans_bits[i] = bitlen;
      ans_final_states[j] = new_state;
    }

    // then do the main loop
    for base_i in (0..final_base_i).step_by(ANS_INTERLEAVING).rev() {
      for j in (0..ANS_INTERLEAVING).rev() {
        let i = base_i + j;
        let (new_state, bitlen) = self.encoder.encode(ans_final_states[j], scratch.symbols[i]);
        ans_vals[i] = bits::lowest_bits_fast(ans_final_states[j], bitlen);
        ans_bits[i] = bitlen;
        ans_final_states[j] = new_state;
      }
    }
  }

  pub fn dissect_batch_latents(&self, latents: &[L], base_i: usize, dst: &mut PageDissectedVar) {
    let mut scratch = self.build_scratch();
    let PageDissectedVar {
      ans_vals,
      ans_bits,
      offsets,
      offset_bits,
      ans_final_states,
    } = dst;

    let search_idxs = self.table.binary_search(latents);

    let end_i = min(base_i + FULL_BATCH_N, ans_vals.len());

    self.dissect_bins(
      &search_idxs[..latents.len()],
      &mut scratch,
      &mut offset_bits[base_i..end_i],
    );

    let offsets = offsets.downcast_mut::<L>().unwrap();
    self.set_offsets(
      latents,
      &mut scratch,
      &mut offsets[base_i..end_i],
    );

    self.encode_ans_in_reverse(
      &mut scratch,
      &mut ans_vals[base_i..end_i],
      &mut ans_bits[base_i..end_i],
      ans_final_states,
    );
  }

  pub unsafe fn uninit_page_dissected_var(&self, n: usize) -> PageDissectedVar {
    let ans_final_states = [self.encoder.default_state(); ANS_INTERLEAVING];
    PageDissectedVar {
      ans_vals: uninit_vec(n),
      ans_bits: uninit_vec(n),
      offsets: DynLatents::new(uninit_vec::<L>(n)).unwrap(),
      offset_bits: uninit_vec(n),
      ans_final_states,
    }
  }

  pub fn dissect_page(&self, latents: &[L]) -> PageDissectedVar {
    let mut page_dissected_var = unsafe { self.uninit_page_dissected_var(latents.len()) };

    // we go through in reverse for ANS!
    for (batch_idx, batch) in latents.chunks(FULL_BATCH_N).enumerate().rev() {
      let base_i = batch_idx * FULL_BATCH_N;
      self.dissect_batch_latents(batch, base_i, &mut page_dissected_var)
    }
    page_dissected_var
  }
}
