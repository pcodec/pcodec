use crate::bit_writer::BitWriter;
use crate::chunk_latent_dissector::ChunkLatentDissector;
use crate::compression_intermediates::BinCompressionInfo;
use crate::compression_intermediates::PageDissectedVar;
use crate::compression_table::CompressionTable;
use crate::constants::{Bitlen, Weight, ANS_INTERLEAVING, PAGE_PADDING};
use crate::data_types::Latent;
use crate::errors::PcoResult;
use crate::macros::{define_latent_enum, match_latent_enum};
use crate::metadata::dyn_latents::DynLatents;
use crate::metadata::{bins, Bin};
use crate::read_write_uint::ReadWriteUint;
use crate::{ans, bit_reader, bit_writer, read_write_uint, FULL_BATCH_N};
use std::io::Write;
use std::ops::Range;

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

fn uninit_vec<T>(n: usize) -> Vec<T> {
  unsafe {
    let mut res = Vec::with_capacity(n);
    res.set_len(n);
    res
  }
}

#[derive(Default)]
pub(crate) struct TrainedBins<L: Latent> {
  pub infos: Vec<BinCompressionInfo<L>>,
  pub ans_size_log: Bitlen,
  pub counts: Vec<Weight>,
}

#[derive(Clone, Debug)]
pub struct ChunkLatentCompressor<L: Latent> {
  table: CompressionTable<L>,
  pub encoder: ans::Encoder,
  pub avg_bits_per_latent: f64,
  is_trivial: bool,
  needs_ans: bool,
  max_u64s_per_offset: usize,
  latents: Vec<L>,
}

impl<L: Latent> ChunkLatentCompressor<L> {
  pub(crate) fn new(trained: TrainedBins<L>, bins: &[Bin<L>], latents: Vec<L>) -> PcoResult<Self> {
    let needs_ans = bins.len() != 1;

    let table = CompressionTable::from(trained.infos);
    let weights = bins::weights(bins);
    let ans_spec = ans::Spec::from_weights(trained.ans_size_log, weights)?;
    let encoder = ans::Encoder::new(&ans_spec);

    let max_bits_per_offset = bins::max_offset_bits(bins);
    let max_u64s_per_offset = read_write_uint::calc_max_u64s_for_writing(max_bits_per_offset);

    Ok(ChunkLatentCompressor {
      table,
      encoder,
      avg_bits_per_latent: bins::avg_bits_per_latent(bins, trained.ans_size_log),
      is_trivial: bins::are_trivial(bins),
      needs_ans,
      max_u64s_per_offset,
      latents,
    })
  }

  pub fn dissect_page(&self, page_range: Range<usize>) -> PageDissectedVar {
    let uninit_page_dissected_var = |n, ans_default_state| {
      let ans_final_states = [ans_default_state; ANS_INTERLEAVING];
      PageDissectedVar {
        ans_vals: uninit_vec(n),
        ans_bits: uninit_vec(n),
        offsets: DynLatents::new(uninit_vec::<L>(n)).unwrap(),
        offset_bits: uninit_vec(n),
        ans_final_states,
      }
    };

    if self.is_trivial {
      return uninit_page_dissected_var(0, self.encoder.default_state());
    }

    let mut page_dissected_var = uninit_page_dissected_var(
      page_range.len(),
      self.encoder.default_state(),
    );

    // we go through in reverse for ANS!
    let mut cld = ChunkLatentDissector::new(&self.table, &self.encoder);
    for (batch_idx, batch) in self.latents[page_range]
      .chunks(FULL_BATCH_N)
      .enumerate()
      .rev()
    {
      let base_i = batch_idx * FULL_BATCH_N;
      cld.dissect_batch_latents(batch, base_i, &mut page_dissected_var)
    }
    page_dissected_var
  }

  pub fn write_dissected_batch<W: Write>(
    &self,
    page_dissected_var: &PageDissectedVar,
    batch_start: usize,
    writer: &mut BitWriter<W>,
  ) -> PcoResult<()> {
    assert!(writer.buf.len() >= PAGE_PADDING);
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

define_latent_enum!(
  #[derive(Clone, Debug)]
  pub DynChunkLatentCompressor(ChunkLatentCompressor)
);
