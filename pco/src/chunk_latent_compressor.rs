use crate::ans::AnsState;
use crate::ans::Symbol;
use crate::bit_writer::BitWriter;
use crate::bits;
use crate::compression_intermediates::PageDissectedVar;
use crate::compression_intermediates::TrainedBins;
use crate::compression_table::CompressionTable;
use crate::constants::Bitlen;
use crate::constants::ANS_INTERLEAVING;
use crate::constants::MAX_BATCH_LATENT_VAR_SIZE;
use crate::data_types::Latent;
use crate::errors::PcoResult;
use crate::macros::{define_latent_enum, match_latent_enum};
use crate::metadata::dyn_latents::DynLatents;
use crate::metadata::{bins, Bin};
use crate::read_write_uint::ReadWriteUint;
use crate::scratch_array::ScratchArray;
use crate::{ans, bit_reader, bit_writer, read_write_uint, FULL_BATCH_N};
use std::io::Write;
use std::ops::Range;

#[derive(Clone, Debug)]
pub struct ChunkLatentCompressorScratch<L: Latent> {
  lowers: ScratchArray<L>,
  symbols: ScratchArray<Symbol>,
}

unsafe fn uninit_vec<T>(n: usize) -> Vec<T> {
  let mut res = Vec::with_capacity(n);
  res.set_len(n);
  res
}

// This would be very hard to combine with write_uints because it makes use of
// an optimization that only works easily for single-u64 writes of 56 bits or
// less: we keep the `target_u64` value we're updating in a register instead
// of referring back to `dst` (recent values of which will be in L1 cache). If
// a write exceeds 56 bits, we may need to shift target_u64 by 64 bits, which
// would be an overflow panic.
#[inline(never)]
unsafe fn write_short_uints<U: ReadWriteUint>(
  vals: &[U],
  bitlens: &[Bitlen],
  mut stale_byte_idx: usize,
  mut bits_past_byte: Bitlen,
  dst: &mut [u8],
) -> (usize, Bitlen) {
  stale_byte_idx += bits_past_byte as usize / 8;
  bits_past_byte %= 8;
  let mut target_u64 = bit_reader::u64_at(dst, stale_byte_idx);

  for (&val, &bitlen) in vals.iter().zip(bitlens).take(FULL_BATCH_N) {
    let bytes_added = bits_past_byte as usize / 8;
    stale_byte_idx += bytes_added;
    target_u64 >>= bytes_added * 8;
    bits_past_byte %= 8;

    target_u64 |= val.to_u64() << bits_past_byte;
    bit_writer::write_u64_to(target_u64, stale_byte_idx, dst);

    bits_past_byte += bitlen;
  }
  (stale_byte_idx, bits_past_byte)
}

#[inline(never)]
unsafe fn write_uints<U: ReadWriteUint, const MAX_U64S: usize>(
  vals: &[U],
  bitlens: &[Bitlen],
  mut stale_byte_idx: usize,
  mut bits_past_byte: Bitlen,
  dst: &mut [u8],
) -> (usize, Bitlen) {
  for (&val, &bitlen) in vals.iter().zip(bitlens).take(FULL_BATCH_N) {
    stale_byte_idx += bits_past_byte as usize / 8;
    bits_past_byte %= 8;
    bit_writer::write_uint_to::<_, MAX_U64S>(val, stale_byte_idx, bits_past_byte, dst);
    bits_past_byte += bitlen;
  }
  (stale_byte_idx, bits_past_byte)
}

#[derive(Clone, Debug)]
pub struct ChunkLatentCompressor<L: Latent> {
  table: CompressionTable<L>,
  pub encoder: ans::Encoder,
  pub avg_bits_per_latent: f64,
  is_trivial: bool, // if the page body will always be empty
  needs_ans: bool,
  max_u64s_per_offset: usize,
  latents: Vec<L>,
  scratch: ChunkLatentCompressorScratch<L>,
}

impl<L: Latent> ChunkLatentCompressor<L> {
  pub fn new(trained: TrainedBins<L>, bins: &[Bin<L>], latents: Vec<L>) -> PcoResult<Box<Self>> {
    let needs_ans = bins.len() != 1;

    let table = CompressionTable::from(trained.infos);
    let weights = bins::weights(bins);
    let ans_spec = ans::Spec::from_weights(trained.ans_size_log, weights)?;
    let encoder = ans::Encoder::new(&ans_spec);

    let max_bits_per_offset = bins::max_offset_bits(bins);
    let max_u64s_per_offset = read_write_uint::calc_max_u64s_for_writing(max_bits_per_offset);
    let default_lower = table.only_bin().map(|info| info.lower).unwrap_or(L::ZERO);

    Ok(Box::new(ChunkLatentCompressor {
      table,
      encoder,
      avg_bits_per_latent: bins::avg_bits_per_latent(bins, trained.ans_size_log),
      is_trivial: bins::are_trivial(bins),
      needs_ans,
      max_u64s_per_offset,
      latents,
      scratch: ChunkLatentCompressorScratch {
        lowers: ScratchArray([default_lower; FULL_BATCH_N]),
        symbols: ScratchArray([0; FULL_BATCH_N]),
      },
    }))
  }

  #[inline(never)]
  fn dissect_bins(&mut self, search_idxs: &[usize], dst_offset_bits: &mut [Bitlen]) {
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
      self.scratch.lowers[i] = info.lower;
      self.scratch.symbols[i] = info.symbol;
      dst_offset_bits[i] = info.offset_bits;
    }
  }

  #[inline(never)]
  fn set_offsets(&self, latents: &[L], offsets: &mut [L]) {
    for (offset, (&latent, &lower)) in offsets
      .iter_mut()
      .zip(latents.iter().zip(self.scratch.lowers.iter()))
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
      let (new_state, bitlen) = self
        .encoder
        .encode(ans_final_states[j], self.scratch.symbols[i]);
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
          .encode(ans_final_states[j], self.scratch.symbols[i]);
        ans_vals[i] = bits::lowest_bits_fast(ans_final_states[j], bitlen);
        ans_bits[i] = bitlen;
        ans_final_states[j] = new_state;
      }
    }
  }

  fn dissect_batch_latents(
    &mut self,
    page_start: usize,
    relative_batch_range: Range<usize>,
    dst: &mut PageDissectedVar,
  ) {
    let PageDissectedVar {
      ans_vals,
      ans_bits,
      offsets,
      offset_bits,
      ans_final_states,
    } = dst;

    let absolute_batch_range =
      page_start + relative_batch_range.start..page_start + relative_batch_range.end;
    let batch_n = relative_batch_range.len();
    let search_idxs = self
      .table
      .binary_search(&self.latents[absolute_batch_range.clone()]);

    self.dissect_bins(
      &search_idxs[..batch_n],
      &mut offset_bits[relative_batch_range.clone()],
    );

    let offsets = offsets.downcast_mut::<L>().unwrap();
    self.set_offsets(
      &self.latents[absolute_batch_range],
      &mut offsets[relative_batch_range.clone()],
    );

    self.encode_ans_in_reverse(
      &mut ans_vals[relative_batch_range.clone()],
      &mut ans_bits[relative_batch_range],
      ans_final_states,
    );
  }

  unsafe fn uninit_page_dissected_var(&self, n: usize) -> PageDissectedVar {
    let ans_final_states = [self.encoder.default_state(); ANS_INTERLEAVING];
    PageDissectedVar {
      ans_vals: uninit_vec(n),
      ans_bits: uninit_vec(n),
      offsets: DynLatents::new(uninit_vec::<L>(n)),
      offset_bits: uninit_vec(n),
      ans_final_states,
    }
  }

  pub fn dissect_page(&mut self, page_range: Range<usize>) -> PageDissectedVar {
    if self.is_trivial {
      // safe because length of uninit elements is 0
      return unsafe { self.uninit_page_dissected_var(0) };
    }

    let page_n = page_range.len();
    let mut page_dissected_var = unsafe { self.uninit_page_dissected_var(page_n) };

    // we go through in reverse for ANS!
    let n_batches = page_n.div_ceil(FULL_BATCH_N);
    for batch_idx in (0..n_batches).rev() {
      let relative_batch_range =
        batch_idx * FULL_BATCH_N..((batch_idx + 1) * FULL_BATCH_N).min(page_n);
      self.dissect_batch_latents(
        page_range.start,
        relative_batch_range,
        &mut page_dissected_var,
      )
    }
    page_dissected_var
  }

  pub fn write_dissected_batch<W: Write>(
    &self,
    page_dissected_var: &PageDissectedVar,
    batch_start: usize,
    writer: &mut BitWriter<W>,
  ) -> PcoResult<()> {
    assert!(writer.buf.len() >= MAX_BATCH_LATENT_VAR_SIZE);
    writer.flush()?;

    if batch_start >= page_dissected_var.offsets.len() {
      return Ok(());
    }

    // write ANS
    if self.needs_ans {
      (writer.stale_byte_idx, writer.bits_past_byte) = unsafe {
        write_short_uints(
          &page_dissected_var.ans_vals[batch_start..],
          &page_dissected_var.ans_bits[batch_start..],
          writer.stale_byte_idx,
          writer.bits_past_byte,
          &mut writer.buf,
        )
      };
    }

    // write offsets
    (writer.stale_byte_idx, writer.bits_past_byte) = unsafe {
      match_latent_enum!(
        &page_dissected_var.offsets,
        DynLatents<L>(offsets) => {
          match self.max_u64s_per_offset {
            0 => (writer.stale_byte_idx, writer.bits_past_byte),
            1 => write_short_uints::<L>(
              &offsets[batch_start..],
              &page_dissected_var.offset_bits[batch_start..],
              writer.stale_byte_idx,
              writer.bits_past_byte,
              &mut writer.buf,
            ),
            2 => write_uints::<L, 2>(
              &offsets[batch_start..],
              &page_dissected_var.offset_bits[batch_start..],
              writer.stale_byte_idx,
              writer.bits_past_byte,
              &mut writer.buf,
            ),
            3 => write_uints::<L, 3>(
              &offsets[batch_start..],
              &page_dissected_var.offset_bits[batch_start..],
              writer.stale_byte_idx,
              writer.bits_past_byte,
              &mut writer.buf,
            ),
            _ => panic!("[ChunkCompressor] data type is too large"),
          }
        }
      )
    };

    Ok(())
  }
}

// we allocate these on the heap because they're enormous
type Boxed<L> = Box<ChunkLatentCompressor<L>>;

define_latent_enum!(
  #[derive(Clone, Debug)]
  pub DynChunkLatentCompressor(Boxed)
);
