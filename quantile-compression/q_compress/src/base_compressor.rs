use std::cmp::{max, min};
use std::fmt::Debug;

use crate::bit_writer::BitWriter;
use crate::chunk_metadata::{ChunkMetadata, PrefixMetadata};
use crate::chunk_spec::ChunkSpec;
use crate::compression_table::CompressionTable;
use crate::constants::*;
use crate::data_types::{NumberLike, UnsignedLike};
use crate::delta_encoding::DeltaMoments;
use crate::errors::{QCompressError, QCompressResult};
use crate::gcd_utils::{GcdOperator, GeneralGcdOp, TrivialGcdOp};
use crate::prefix::{Prefix, PrefixCompressionInfo};
use crate::{delta_encoding, gcd_utils, huffman_encoding, prefix_optimization, Flags};

/// All configurations available for a compressor.
///
/// Some, like `delta_encoding_order`, are explicitly stored as `Flags` in the
/// compressed bytes.
/// Others, like `compression_level`, affect compression but are not explicitly
/// stored in the output.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct CompressorConfig {
  /// `compression_level` ranges from 0 to 12 inclusive (default 8).
  ///
  /// The compressor uses up to 2^`compression_level` prefixes.
  ///
  /// For example,
  /// * Level 0 achieves a modest amount of compression with 1 prefix and can
  /// be twice as fast as level 8.
  /// * Level 8 achieves nearly the best compression with 256 prefixes and still
  /// runs in reasonable time. In some cases, its compression ratio is 3-4x as
  /// high as level level 0's.
  /// * Level 12 can achieve a few % better compression than 8 with 4096
  /// prefixes but runs ~5x slower in many cases.
  pub compression_level: usize,
  /// `delta_encoding_order` ranges from 0 to 7 inclusive (default 0).
  ///
  /// It is the number of times to apply delta encoding
  /// before compressing. For instance, say we have the numbers
  /// `[0, 2, 2, 4, 4, 6, 6]` and consider different delta encoding orders.
  /// * 0 indicates no delta encoding, it compresses numbers
  /// as-is. This is perfect for columnar data were the order is essentially
  /// random.
  /// * 1st order delta encoding takes consecutive differences, leaving
  /// `[0, 2, 0, 2, 0, 2, 0]`. This is perfect for continuous but noisy time
  /// series data, like stock prices.
  /// * 2nd order delta encoding takes consecutive differences again,
  /// leaving `[2, -2, 2, -2, 2, -2]`. This is perfect for locally linear data,
  /// like a sequence of timestamps sampled approximately periodically.
  /// * Higher-order delta encoding is good for time series that are very
  /// smooth, like temperature or light sensor readings.
  ///
  /// Setting delta encoding order too high or low will hurt compression ratio.
  /// If you're unsure, use
  /// [`auto_compressor_config()`][crate::auto_compressor_config] to choose it.
  pub delta_encoding_order: usize,
  /// `use_gcds` improves compression ratio in cases where all
  /// numbers in a range share a nontrivial Greatest Common Divisor
  /// (default true).
  ///
  /// Examples where this helps:
  /// * integers `[7, 107, 207, 307, ... 100007]` shuffled
  /// * floats `[1.0, 2.0, ... 1000.0]` shuffled
  /// * nanosecond-precision timestamps that are all whole numbers of
  /// microseconds
  ///
  /// When this is helpful and in rare cases when it isn't, compression speed
  /// is slightly reduced.
  pub use_gcds: bool,
}

impl Default for CompressorConfig {
  fn default() -> Self {
    Self {
      compression_level: DEFAULT_COMPRESSION_LEVEL,
      delta_encoding_order: 0,
      use_gcds: true,
    }
  }
}

impl CompressorConfig {
  /// Sets [`compression_level`][CompressorConfig::compression_level].
  pub fn with_compression_level(mut self, level: usize) -> Self {
    self.compression_level = level;
    self
  }

  /// Sets [`delta_encoding_order`][CompressorConfig::delta_encoding_order].
  pub fn with_delta_encoding_order(mut self, order: usize) -> Self {
    self.delta_encoding_order = order;
    self
  }

  /// Sets [`use_gcds`][CompressorConfig::use_gcds].
  pub fn with_use_gcds(mut self, use_gcds: bool) -> Self {
    self.use_gcds = use_gcds;
    self
  }
}

// InternalCompressorConfig captures all settings that don't belong in flags
// i.e. these don't get written to the resulting bytes and aren't needed for
// decoding
#[derive(Clone, Debug)]
struct InternalCompressorConfig {
  pub compression_level: usize,
}

impl From<&CompressorConfig> for InternalCompressorConfig {
  fn from(config: &CompressorConfig) -> Self {
    InternalCompressorConfig {
      compression_level: config.compression_level,
    }
  }
}

impl Default for InternalCompressorConfig {
  fn default() -> Self {
    Self::from(&CompressorConfig::default())
  }
}

fn cumulative_sum(sizes: &[usize]) -> Vec<usize> {
  // there has got to be a better way to write this
  let mut res = Vec::with_capacity(sizes.len());
  let mut sum = 0;
  for s in sizes {
    res.push(sum);
    sum += s;
  }
  res
}

struct PrefixBuffer<'a, T: NumberLike> {
  pub seq: Vec<Prefix<T>>,
  prefix_idx: usize,
  max_n_pref: usize,
  n_unsigneds: usize,
  sorted: &'a [T::Unsigned],
  use_gcd: bool,
  pub target_j: usize,
}

impl<'a, T: NumberLike> PrefixBuffer<'a, T> {
  fn calc_target_j(&mut self) {
    self.target_j = ((self.prefix_idx + 1) * self.n_unsigneds) / self.max_n_pref
  }

  fn new(max_n_pref: usize, n_unsigneds: usize, sorted: &'a [T::Unsigned], use_gcd: bool) -> Self {
    let mut res = Self {
      seq: Vec::with_capacity(max_n_pref),
      prefix_idx: 0,
      max_n_pref,
      n_unsigneds,
      sorted,
      use_gcd,
      target_j: 0,
    };
    res.calc_target_j();
    res
  }

  fn push_pref(&mut self, i: usize, j: usize) {
    let sorted = self.sorted;
    let n_unsigneds = self.n_unsigneds;

    let count = j - i;
    let new_prefix_idx = max(
      self.prefix_idx + 1,
      (j * self.max_n_pref) / n_unsigneds,
    );
    let lower = T::from_unsigned(sorted[i]);
    let upper = T::from_unsigned(sorted[j - 1]);
    let gcd = if self.use_gcd {
      gcd_utils::gcd(&sorted[i..j])
    } else {
      T::Unsigned::ONE
    };
    // code and run_len_jumpstart get filled in later
    let p = Prefix {
      count,
      lower,
      upper,
      gcd,
      code: Vec::new(),
      run_len_jumpstart: None,
    };
    self.seq.push(p);
    self.prefix_idx = new_prefix_idx;
    self.calc_target_j();
  }
}

// 2 ^ comp level, with 2 caveats:
// * Enforce n_prefixes <= n_unsigneds
// * Due to prefix optimization compute cost ~ O(4 ^ comp level), limit max comp level when
// n_unsigneds is small
fn choose_max_n_prefixes(comp_level: usize, n_unsigneds: usize) -> usize {
  let log_n = (n_unsigneds as f64).log2().floor() as usize;
  let fast_comp_level = log_n.saturating_sub(4);
  let real_comp_level = if comp_level <= fast_comp_level {
    comp_level
  } else {
    fast_comp_level + comp_level.saturating_sub(fast_comp_level) / 2
  };
  min(1_usize << real_comp_level, n_unsigneds)
}

fn choose_unoptimized_prefixes<T: NumberLike>(
  sorted: &[T::Unsigned],
  internal_config: &InternalCompressorConfig,
  flags: &Flags,
) -> Vec<Prefix<T>> {
  let n_unsigneds = sorted.len();
  let max_n_pref = choose_max_n_prefixes(
    internal_config.compression_level,
    n_unsigneds,
  );

  let use_gcd = flags.use_gcds;
  let mut i = 0;
  let mut backup_j = 0_usize;
  let mut prefix_buffer = PrefixBuffer::<T>::new(max_n_pref, n_unsigneds, sorted, use_gcd);

  for j in 1..n_unsigneds {
    let target_j = prefix_buffer.target_j;
    if sorted[j] == sorted[j - 1] {
      if j >= target_j && j - target_j >= target_j - backup_j && backup_j > i {
        prefix_buffer.push_pref(i, backup_j);
        i = backup_j;
      }
    } else {
      backup_j = j;
      if j >= target_j {
        prefix_buffer.push_pref(i, j);
        i = j;
      }
    }
  }
  prefix_buffer.push_pref(i, n_unsigneds);

  prefix_buffer.seq
}

fn train_prefixes<T: NumberLike>(
  unsigneds: Vec<T::Unsigned>,
  internal_config: &InternalCompressorConfig,
  flags: &Flags,
  n: usize, // can be greater than unsigneds.len() if delta encoding is on
) -> QCompressResult<Vec<Prefix<T>>> {
  if unsigneds.is_empty() {
    return Ok(Vec::new());
  }

  let comp_level = internal_config.compression_level;
  if comp_level > MAX_COMPRESSION_LEVEL {
    return Err(QCompressError::invalid_argument(format!(
      "compression level may not exceed {} (was {})",
      MAX_COMPRESSION_LEVEL, comp_level,
    )));
  }
  if n > MAX_ENTRIES {
    return Err(QCompressError::invalid_argument(format!(
      "count may not exceed {} per chunk (was {})",
      MAX_ENTRIES, n,
    )));
  }

  let unoptimized_prefs = {
    let mut sorted = unsigneds;
    sorted.sort_unstable();
    choose_unoptimized_prefixes(&sorted, internal_config, flags)
  };

  // combine adjacent prefixes when advantageous and fill in run_len_jumpstart
  let mut optimized_prefs = prefix_optimization::optimize_prefixes(unoptimized_prefs, flags, n);

  // fill in Huffman codes for the now optimized prefixes
  huffman_encoding::make_huffman_code(&mut optimized_prefs, n);

  Ok(optimized_prefs)
}

fn trained_compress_body<U: UnsignedLike>(
  table: &CompressionTable<U>,
  use_gcd: bool,
  unsigneds: &[U],
  writer: &mut BitWriter,
) -> QCompressResult<()> {
  if use_gcd {
    compress_data_page::<U, GeneralGcdOp>(table, unsigneds, writer)
  } else {
    compress_data_page::<U, TrivialGcdOp>(table, unsigneds, writer)
  }
}

fn compress_data_page<U: UnsignedLike, GcdOp: GcdOperator<U>>(
  table: &CompressionTable<U>,
  unsigneds: &[U],
  writer: &mut BitWriter,
) -> QCompressResult<()> {
  let mut i = 0;
  while i < unsigneds.len() {
    let unsigned = unsigneds[i];
    let p = table.search(unsigned)?;
    writer.write_usize(p.code, p.code_len);
    match p.run_len_jumpstart {
      None => {
        compress_offset::<U, GcdOp>(unsigned, p, writer);
        i += 1;
      }
      Some(jumpstart) => {
        let mut reps = 1;
        for &other in unsigneds.iter().skip(i + 1) {
          if p.contains(other) {
            reps += 1;
          } else {
            break;
          }
        }

        // we store 1 less than the number of occurrences
        // because the prefix already implies there is at least 1 occurrence
        writer.write_varint(reps - 1, jumpstart);

        for &unsigned in unsigneds.iter().skip(i).take(reps) {
          compress_offset::<U, GcdOp>(unsigned, p, writer);
        }
        i += reps;
      }
    }
  }
  writer.finish_byte();
  Ok(())
}

fn compress_offset<U: UnsignedLike, GcdOp: GcdOperator<U>>(
  unsigned: U,
  p: &PrefixCompressionInfo<U>,
  writer: &mut BitWriter,
) {
  let off = GcdOp::get_offset(unsigned - p.lower, p.gcd);
  writer.write_diff(off, p.k);
  if off < p.only_k_bits_lower || off > p.only_k_bits_upper {
    // most significant bit, if necessary, comes last
    writer.write_one((off & (U::ONE << p.k)) > U::ZERO);
  }
}

#[derive(Clone, Debug)]
pub struct MidChunkInfo<T: NumberLike> {
  // immutable:
  unsigneds: Vec<T::Unsigned>,
  use_gcd: bool,
  table: CompressionTable<T::Unsigned>,
  delta_momentss: Vec<DeltaMoments<T::Signed>>,
  page_sizes: Vec<usize>,
  // mutable:
  idx: usize,
  page_idx: usize,
}

impl<T: NumberLike> MidChunkInfo<T> {
  fn data_page_n(&self) -> usize {
    self.page_sizes[self.page_idx]
  }

  fn data_page_moments(&self) -> &DeltaMoments<T::Signed> {
    &self.delta_momentss[self.page_idx]
  }

  fn n_pages(&self) -> usize {
    self.page_sizes.len()
  }
}

#[derive(Clone, Debug, Default)]
pub enum State<T: NumberLike> {
  #[default]
  PreHeader,
  StartOfChunk,
  MidChunk(MidChunkInfo<T>),
  Terminated,
}

impl<T: NumberLike> State<T> {
  pub fn wrong_step_err(&self, description: &str) -> QCompressError {
    let step_str = match self {
      State::PreHeader => "has not yet written header",
      State::StartOfChunk => "is at the start of a chunk",
      State::MidChunk(_) => "is mid-chunk",
      State::Terminated => "has already written the footer",
    };
    QCompressError::invalid_argument(format!(
      "attempted to write {} when compressor {}",
      description, step_str,
    ))
  }
}

#[derive(Clone, Debug)]
pub struct BaseCompressor<T: NumberLike> {
  internal_config: InternalCompressorConfig,
  pub flags: Flags,
  pub writer: BitWriter,
  pub state: State<T>,
}

impl<T: NumberLike> BaseCompressor<T> {
  pub fn from_config(config: CompressorConfig, use_wrapped_mode: bool) -> Self {
    Self {
      internal_config: InternalCompressorConfig::from(&config),
      flags: Flags::from_config(&config, use_wrapped_mode),
      writer: BitWriter::default(),
      state: State::default(),
    }
  }

  pub fn header(&mut self) -> QCompressResult<()> {
    if !matches!(self.state, State::PreHeader) {
      return Err(self.state.wrong_step_err("header"));
    }

    self.writer.write_aligned_bytes(&MAGIC_HEADER)?;
    self.writer.write_aligned_byte(T::HEADER_BYTE)?;
    self.flags.write(&mut self.writer)?;
    self.state = State::StartOfChunk;
    Ok(())
  }

  pub fn chunk_metadata_internal(
    &mut self,
    nums: &[T],
    spec: &ChunkSpec,
  ) -> QCompressResult<ChunkMetadata<T>> {
    if !matches!(self.state, State::StartOfChunk) {
      return Err(self.state.wrong_step_err("chunk metadata"));
    }

    if nums.is_empty() {
      return Err(QCompressError::invalid_argument(
        "cannot compress empty chunk",
      ));
    }

    let n = nums.len();
    let page_sizes = spec.page_sizes(nums.len())?;
    let n_pages = page_sizes.len();

    if !self.flags.use_wrapped_mode {
      self.writer.write_aligned_byte(MAGIC_CHUNK_BYTE)?;
    }

    let order = self.flags.delta_encoding_order;
    let (unsigneds, prefix_meta, table, delta_momentss) = if order == 0 {
      let unsigneds = nums.iter().map(|x| x.to_unsigned()).collect::<Vec<_>>();
      let prefixes = train_prefixes(
        unsigneds.clone(),
        &self.internal_config,
        &self.flags,
        n,
      )?;
      let table = CompressionTable::from(prefixes.as_slice());
      let prefix_metadata = PrefixMetadata::Simple { prefixes };
      (
        unsigneds,
        prefix_metadata,
        table,
        vec![DeltaMoments::default(); n_pages],
      )
    } else {
      let page_idxs = cumulative_sum(&page_sizes);
      let (deltas, momentss) = delta_encoding::nth_order_deltas(nums, order, &page_idxs);
      let unsigneds = deltas.iter().map(|x| x.to_unsigned()).collect::<Vec<_>>();
      let prefixes = train_prefixes(
        unsigneds.clone(),
        &self.internal_config,
        &self.flags,
        n,
      )?;
      let table = CompressionTable::from(prefixes.as_slice());
      let prefix_metadata = PrefixMetadata::Delta { prefixes };
      (unsigneds, prefix_metadata, table, momentss)
    };

    let chunk_meta_moments = delta_momentss[0].clone();
    let use_gcd = prefix_meta.use_gcd();
    let meta = ChunkMetadata::new(n, prefix_meta, chunk_meta_moments);
    meta.write_to(&mut self.writer, &self.flags);

    self.state = State::MidChunk(MidChunkInfo {
      unsigneds,
      use_gcd,
      table,
      delta_momentss,
      page_sizes,
      idx: 0,
      page_idx: 0,
    });

    Ok(meta)
  }

  pub fn data_page_internal(&mut self) -> QCompressResult<()> {
    let has_pages_remaining = {
      let info = match &mut self.state {
        State::MidChunk(info) => Ok(info),
        other => Err(other.wrong_step_err("data page")),
      }?;

      let start = info.idx;
      let data_page_n = info.data_page_n();
      let end = start + data_page_n.saturating_sub(self.flags.delta_encoding_order);
      if self.flags.use_wrapped_mode {
        info.data_page_moments().write_to(&mut self.writer);
      }
      let slice = if end > start {
        &info.unsigneds[start..end]
      } else {
        &[]
      };
      trained_compress_body(
        &info.table,
        info.use_gcd,
        slice,
        &mut self.writer,
      )?;

      info.idx += data_page_n;
      info.page_idx += 1;

      info.page_idx < info.n_pages()
    };

    if !has_pages_remaining {
      self.state = State::StartOfChunk;
    }

    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::choose_max_n_prefixes;

  #[test]
  fn test_choose_max_n_prefixes() {
    assert_eq!(choose_max_n_prefixes(0, 100), 1);
    assert_eq!(choose_max_n_prefixes(12, 200), 1 << 7);
    assert_eq!(choose_max_n_prefixes(12, 1 << 10), 1 << 9);
    assert_eq!(choose_max_n_prefixes(8, 1 << 10), 1 << 7);
    assert_eq!(choose_max_n_prefixes(1, 1 << 10), 2);
    assert_eq!(
      choose_max_n_prefixes(12, (1 << 12) - 1),
      1 << 9
    );
    assert_eq!(choose_max_n_prefixes(12, 1 << 12), 1 << 10);
    assert_eq!(
      choose_max_n_prefixes(12, (1 << 16) - 1),
      1 << 11
    );
    assert_eq!(choose_max_n_prefixes(12, 1 << 16), 1 << 12);
    assert_eq!(choose_max_n_prefixes(12, 1 << 20), 1 << 12);
  }
}
