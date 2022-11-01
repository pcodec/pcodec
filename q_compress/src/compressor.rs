use std::cmp::{max, min};
use std::fmt::Debug;
use std::marker::PhantomData;

use crate::{Flags, gcd_utils, huffman_encoding};
use crate::bit_writer::BitWriter;
use crate::chunk_metadata::{ChunkMetadata, ChunkSpec, PrefixMetadata};
use crate::compression_table::CompressionTable;
use crate::constants::*;
use crate::data_page::DataPageMetadata;
use crate::data_types::{NumberLike, UnsignedLike};
use crate::delta_encoding;
use crate::delta_encoding::DeltaMoments;
use crate::errors::{QCompressError, QCompressResult};
use crate::gcd_utils::{GcdOperator, GeneralGcdOp, TrivialGcdOp};
use crate::prefix::{Prefix, PrefixCompressionInfo, WeightedPrefix};
use crate::prefix_optimization;

const MIN_N_TO_USE_RUN_LEN: usize = 1001;
const MIN_FREQUENCY_TO_USE_RUN_LEN: f64 = 0.8;
const DEFAULT_CHUNK_SIZE: usize = 1000000;

struct JumpstartConfiguration {
  weight: usize,
  jumpstart: usize,
}

/// All configurations available for a [`Compressor`].
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
  // TODO
  pub use_wrapped_mode: bool,
}

impl Default for CompressorConfig {
  fn default() -> Self {
    Self {
      compression_level: DEFAULT_COMPRESSION_LEVEL,
      delta_encoding_order: 0,
      use_gcds: true,
      use_wrapped_mode: false,
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

  /// Sets [`use_wrapped_mode`][CompressorConfig::use_wrapped_mode]
  pub fn with_use_wrapped_mode(mut self, use_wrapped_mode: bool) -> Self {
    self.use_wrapped_mode = use_wrapped_mode;
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

fn choose_run_len_jumpstart(
  count: usize,
  n: usize,
) -> JumpstartConfiguration {
  let freq = (count as f64) / (n as f64);
  let non_freq = 1.0 - freq;
  let jumpstart = min((-non_freq.log2()).ceil() as usize, MAX_JUMPSTART);
  let expected_n_runs = (freq * non_freq * n as f64).ceil() as usize;
  JumpstartConfiguration {
    weight: expected_n_runs,
    jumpstart,
  }
}

struct PrefixBuffer<'a, T: NumberLike> {
  pub seq: &'a mut Vec<WeightedPrefix<T>>,
  pub prefix_idx: &'a mut usize,
  pub max_n_pref: usize,
  pub n_unsigneds: usize,
  pub sorted: &'a [T::Unsigned],
  pub use_gcd: bool,
}

fn push_pref<T: NumberLike>(
  buffer: &mut PrefixBuffer<'_, T>,
  i: usize,
  j: usize,
) {
  let sorted = buffer.sorted;
  let n_unsigneds = buffer.n_unsigneds;

  let count = j - i;
  let frequency = count as f64 / buffer.n_unsigneds as f64;
  let new_prefix_idx = max(*buffer.prefix_idx + 1, (j * buffer.max_n_pref) / n_unsigneds);
  let lower = T::from_unsigned(sorted[i]);
  let upper = T::from_unsigned(sorted[j - 1]);
  let gcd = if buffer.use_gcd {
    gcd_utils::gcd(&sorted[i..j])
  } else {
    T::Unsigned::ONE
  };
  if n_unsigneds < MIN_N_TO_USE_RUN_LEN || frequency < MIN_FREQUENCY_TO_USE_RUN_LEN || count == n_unsigneds {
    // The usual case - a prefix for a range that represents either 100% or
    // <=80% of the data.
    buffer.seq.push(WeightedPrefix::new(
      count,
      count,
      lower,
      upper,
      None,
      gcd,
    ));
  } else {
    // The weird case - a range that represents almost all (but not all) the data.
    // We create extra prefixes that can describe `reps` copies of the range at once.
    let config = choose_run_len_jumpstart(count, n_unsigneds);
    buffer.seq.push(WeightedPrefix::new(
      count,
      config.weight,
      lower,
      upper,
      Some(config.jumpstart),
      gcd,
    ));
  }
  *buffer.prefix_idx = new_prefix_idx;
}

// 2 ^ comp level, with 2 caveats:
// * Enforce n_prefixes <= n_unsigneds
// * Due to prefix optimization compute cost ~ O(4 ^ comp level), limit max comp level when
// n_unsigneds is small
fn choose_max_n_prefixes(comp_level: usize, n_unsigneds: usize) -> usize {
  let log_n = (n_unsigneds as f64).log2().floor() as usize;
  let max_comp_level_for_n = min(MAX_COMPRESSION_LEVEL, log_n / 2 + 5);
  let real_comp_level = comp_level.saturating_sub(MAX_COMPRESSION_LEVEL - max_comp_level_for_n);
  min(1_usize << real_comp_level, n_unsigneds)
}

fn choose_unoptimized_prefixes<T: NumberLike>(
  sorted: &[T::Unsigned],
  internal_config: &InternalCompressorConfig,
  flags: &Flags,
) -> Vec<WeightedPrefix<T>> {
  let n_unsigneds = sorted.len();
  let max_n_pref = choose_max_n_prefixes(internal_config.compression_level, n_unsigneds);
  let mut raw_prefs: Vec<WeightedPrefix<T>> = Vec::new();
  let mut pref_idx = 0_usize;

  let use_gcd = flags.use_gcds;
  let mut i = 0;
  let mut backup_j = 0_usize;
  let mut prefix_buffer = PrefixBuffer::<T> {
    seq: &mut raw_prefs,
    prefix_idx: &mut pref_idx,
    max_n_pref,
    n_unsigneds,
    sorted,
    use_gcd,
  };

  for j in 0..n_unsigneds {
    let target_j = ((*prefix_buffer.prefix_idx + 1) * n_unsigneds) / max_n_pref;
    if j > 0 && sorted[j] == sorted[j - 1] {
      if j >= target_j && j - target_j >= target_j - backup_j && backup_j > i {
        push_pref(&mut prefix_buffer, i, backup_j);
        i = backup_j;
      }
    } else {
      backup_j = j;
      if j >= target_j {
        push_pref(&mut prefix_buffer, i, j);
        i = j;
      }
    }
  }
  push_pref(&mut prefix_buffer, i, n_unsigneds);

  raw_prefs
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
      MAX_COMPRESSION_LEVEL,
      comp_level,
    )));
  }
  if n > MAX_ENTRIES {
    return Err(QCompressError::invalid_argument(format!(
      "count may not exceed {} per chunk (was {})",
      MAX_ENTRIES,
      n,
    )));
  }

  let unoptimized_prefs = {
    let mut sorted = unsigneds;
    sorted.sort_unstable();
    choose_unoptimized_prefixes(
      &sorted,
      internal_config,
      flags
    )
  };

  let mut optimized_prefs = prefix_optimization::optimize_prefixes(
    unoptimized_prefs,
    flags,
    n,
  );

  huffman_encoding::make_huffman_code(&mut optimized_prefs);

  let prefixes = optimized_prefs.iter()
    .map(|wp| wp.prefix.clone())
    .collect();
  Ok(prefixes)
}

#[derive(Clone)]
struct TrainedChunkCompressor<'a, U: UnsignedLike, GcdOp: GcdOperator<U>> {
  pub table: &'a CompressionTable<U>,
  op: PhantomData<GcdOp>,
}

fn trained_compress_body<U: UnsignedLike>(
  table: &CompressionTable<U>,
  use_gcd: bool,
  unsigneds: &[U],
  writer: &mut BitWriter,
) -> QCompressResult<()> {
  if use_gcd {
    TrainedChunkCompressor::<U, GeneralGcdOp> { table, op: PhantomData }
      .compress_data_page(unsigneds, writer)
  } else {
    TrainedChunkCompressor::<U, TrivialGcdOp> { table, op: PhantomData }
      .compress_data_page(unsigneds, writer)
  }
}

impl<'a, U, GcdOp> TrainedChunkCompressor<'a, U, GcdOp> where U: UnsignedLike, GcdOp: GcdOperator<U> {
  fn compress_data_page(
    &self,
    unsigneds: &[U],
    writer: &mut BitWriter,
  ) -> QCompressResult<()> {
    let mut i = 0;
    while i < unsigneds.len() {
      let unsigned = unsigneds[i];
      let p = self.table.search(unsigned)?;
      writer.write_usize(p.code, p.code_len);
      match p.run_len_jumpstart {
        None => {
          Self::compress_offset_bits_w_prefix(unsigned, p, writer);
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
            Self::compress_offset_bits_w_prefix(unsigned, p, writer);
          }
          i += reps;
        }
      }
    }
    writer.finish_byte();
    Ok(())
  }

  fn compress_offset_bits_w_prefix(
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
}

#[derive(Clone, Debug)]
struct MidChunkInfo<T: NumberLike> {
  // immutable:
  unsigneds: Vec<T::Unsigned>,
  use_gcd: bool,
  table: CompressionTable<T::Unsigned>,
  data_pages: Vec<DataPageMetadata<T>>,
  // mutable:
  idx: usize,
  page_idx: usize,
}

impl<T: NumberLike> MidChunkInfo<T> {
  fn data_page(&self) -> &DataPageMetadata<T> {
    &self.data_pages[self.page_idx]
  }
}

#[derive(Clone, Debug)]
enum State<T: NumberLike> {
  PreHeader,
  StartOfChunk,
  MidChunk(MidChunkInfo<T>),
  Terminated,
}

impl<T: NumberLike> Default for State<T> {
  fn default() -> Self {
    State::PreHeader
  }
}

impl<T: NumberLike> State<T> {
  fn wrong_step_err(&self, description: &str) -> QCompressError {
    let step_str = match self {
      State::PreHeader => "has not yet written header",
      State::StartOfChunk => "is at the start of a chunk",
      State::MidChunk(_) => "is mid-chunk",
      State::Terminated => "has already written the footer",
    };
    QCompressError::invalid_argument(format!(
      "attempted to write {} when compressor {}",
      description,
      step_str,
    ))
  }
}

/// Converts vectors of numbers into compressed bytes.
///
/// All `Compressor` methods leave its state unchanged if they return an error.
/// You can configure behavior like compression level by instantiating with
/// [`.from_config()`][Compressor::from_config]
///
/// You can use the compressor at a file or chunk level.
/// ```
/// use q_compress::Compressor;
///
/// let my_nums = vec![1, 2, 3];
///
/// // FILE LEVEL
/// let mut compressor = Compressor::<i32>::default();
/// let bytes = compressor.simple_compress(&my_nums);
///
/// // CHUNK LEVEL
/// let mut compressor = Compressor::<i32>::default();
/// compressor.header().expect("header");
/// compressor.chunk(&my_nums).expect("chunk");
/// compressor.footer().expect("footer");
/// let bytes = compressor.drain_bytes();
/// ```
/// Note that in practice we would need larger chunks than this to
/// achieve good compression, preferably containing 3k-10M numbers.
#[derive(Clone, Debug)]
pub struct Compressor<T> where T: NumberLike {
  internal_config: InternalCompressorConfig,
  flags: Flags,
  writer: BitWriter,
  state: State<T>,
}

impl<T: NumberLike> Default for Compressor<T> {
  fn default() -> Self {
    Self::from_config(CompressorConfig::default())
  }
}

impl<T> Compressor<T> where T: NumberLike {
  /// Creates a new compressor, given a [`CompressorConfig`].
  /// Internally, the compressor builds [`Flags`] as well as an internal
  /// configuration that doesn't show up in the output file.
  /// You can inspect the flags it chooses with [`.flags()`][Self::flags].
  pub fn from_config(config: CompressorConfig) -> Self {
    Self {
      internal_config: InternalCompressorConfig::from(&config),
      flags: Flags::from(&config),
      writer: BitWriter::default(),
      state: State::default(),
    }
  }

  /// Returns a reference to the compressor's flags.
  pub fn flags(&self) -> &Flags {
    &self.flags
  }

  /// Writes out a header using the compressor's data type and flags.
  /// Will return an error if the compressor has already written the header or
  /// footer.
  ///
  /// Each .qco file must start with such a header, which contains:
  /// * a 4-byte magic header for "qco!" in ascii,
  /// * a byte for the data type (e.g. `i64` has byte 1 and `f64` has byte
  /// 5), and
  /// * bytes for the flags used to compress.
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

  /// TODO: documentation
  pub fn chunk_metadata(
    &mut self,
    nums: &[T],
    spec: &ChunkSpec,
  ) -> QCompressResult<ChunkMetadata<T>> {
    self.flags.check_wrapped_mode()?;
    self.chunk_metadata_internal(nums, spec)
  }

  fn chunk_metadata_internal(
    &mut self,
    nums: &[T],
    spec: &ChunkSpec,
  ) -> QCompressResult<ChunkMetadata<T>> {
    if !matches!(self.state, State::StartOfChunk) {
      return Err(self.state.wrong_step_err("chunk metadata"));
    }

    if nums.is_empty() {
      return Err(QCompressError::invalid_argument(
        "cannot compress empty chunk"
      ));
    }

    let n = nums.len();
    let page_sizes = spec.page_sizes(nums.len())?;
    let n_pages = page_sizes.len();

    self.writer.write_aligned_byte(MAGIC_CHUNK_BYTE)?;

    let order = self.flags.delta_encoding_order;
    let (
      unsigneds,
      prefix_meta,
      use_gcd,
      table,
      delta_momentss,
    ) = if order == 0 {
      let unsigneds = nums.iter()
        .map(|x| x.to_unsigned())
        .collect::<Vec<_>>();
      let prefixes = train_prefixes(
        unsigneds.clone(),
        &self.internal_config,
        &self.flags,
        n,
      )?;
      let use_gcd = gcd_utils::use_gcd_arithmetic(&prefixes);
      let table = CompressionTable::from(prefixes.as_slice());
      let prefix_metadata = PrefixMetadata::Simple {
        prefixes,
      };
      (unsigneds, prefix_metadata, use_gcd, table, vec![DeltaMoments::default(); n_pages])
    } else {
      let page_idxs = cumulative_sum(&page_sizes);
      let (deltas, momentss) = delta_encoding::nth_order_deltas(
        nums,
        order,
        &page_idxs,
      );
      println!("MOMENTSS {:?}", momentss);
      let unsigneds = deltas.iter()
        .map(|x| x.to_unsigned())
        .collect::<Vec<_>>();
      let prefixes = train_prefixes(
        unsigneds.clone(),
        &self.internal_config,
        &self.flags,
        n,
      )?;
      let use_gcd = gcd_utils::use_gcd_arithmetic(&prefixes);
      let table = CompressionTable::from(prefixes.as_slice());
      let prefix_metadata = PrefixMetadata::Delta {
        delta_moments: momentss[0].clone(),
        prefixes,
      };
      (unsigneds, prefix_metadata, use_gcd, table, momentss)
    };

    let data_pages = delta_momentss.into_iter()
      .zip(page_sizes.into_iter())
      .map(|(moments, n)| DataPageMetadata::new(moments, n))
      .collect::<Vec<_>>();
    let meta = ChunkMetadata::new(n, prefix_meta);
    meta.write_to(&mut self.writer, &self.flags);

    self.state = State::MidChunk(MidChunkInfo {
      unsigneds,
      use_gcd,
      table,
      data_pages,
      idx: 0,
      page_idx: 0,
    });

    Ok(meta)
  }

  /// TODO documentation
  pub fn data_page(&mut self) -> QCompressResult<bool> {
    if !self.flags.use_wrapped_mode {
      return Err(QCompressError::invalid_argument(
        "data pages are not supported in standalone mode"
      ))
    }
    self.data_page_internal()
  }

  fn data_page_internal(&mut self) -> QCompressResult<bool> {
    let has_pages_remaining = {
      let info = match &mut self.state {
        State::MidChunk(info) => Ok(info),
        other => Err(other.wrong_step_err("data page")),
      }?;

      let start = info.idx;
      let data_page_meta = info.data_page();
      let end = start + data_page_meta.n.saturating_sub(self.flags.delta_encoding_order);
      if self.flags.use_wrapped_mode {
        data_page_meta.write_to(&mut self.writer);
      }
      println!("WRITING FROM {} TO {} OUT OF {}", start, end, data_page_meta.n);
      trained_compress_body(
        &info.table,
        info.use_gcd,
        &info.unsigneds[start..end],
        &mut self.writer,
      )?;

      info.idx += data_page_meta.n;
      info.page_idx += 1;

      info.page_idx < info.data_pages.len()
    };

    if !has_pages_remaining {
      self.state = State::StartOfChunk;
    }

    Ok(has_pages_remaining)
  }

  /// Writes out a chunk of data representing the provided numbers.
  /// Will return an error if the compressor has not yet written the header
  /// or already written the footer.
  ///
  /// Each chunk contains a [`ChunkMetadata`] section followed by the chunk body.
  /// The chunk body encodes the numbers passed in here.
  pub fn chunk(&mut self, nums: &[T]) -> QCompressResult<ChunkMetadata<T>> {
    let pre_meta_bit_idx = self.writer.bit_size();
    let mut meta = self.chunk_metadata_internal(nums, &ChunkSpec::default())?;
    let post_meta_byte_idx = self.writer.byte_size();

    self.data_page_internal()?;

    meta.compressed_body_size = self.writer.byte_size() - post_meta_byte_idx;
    meta.update_write_compressed_body_size(&mut self.writer, pre_meta_bit_idx);
    Ok(meta)
  }

  /// Writes out a single footer byte indicating that the .qco file has ended.
  /// Will return an error if the compressor has not yet written the header
  /// or already written the footer.
  pub fn footer(&mut self) -> QCompressResult<()> {
    if !matches!(self.state, State::StartOfChunk) {
      return Err(self.state.wrong_step_err("footer"));
    }

    self.writer.write_aligned_byte(MAGIC_TERMINATION_BYTE)?;
    self.state = State::Terminated;
    Ok(())
  }

  /// Takes in a slice of numbers and returns compressed bytes.
  pub fn simple_compress(&mut self, nums: &[T]) -> Vec<u8> {
    // The following unwraps are safe because the writer will be byte-aligned
    // after each step and ensure each chunk has appropriate size.
    self.header().unwrap();
    nums.chunks(DEFAULT_CHUNK_SIZE)
      .for_each(|chunk| {
        self.chunk(chunk).unwrap();
      });

    self.footer().unwrap();
    self.drain_bytes()
  }

  /// Returns all bytes produced by the compressor so far that have not yet
  /// been read.
  ///
  /// In the future we may implement a method to write to a `std::io::Write` or
  /// implement `Compressor` as `std::io::Read`, TBD.
  pub fn drain_bytes(&mut self) -> Vec<u8> {
    self.writer.drain_bytes()
  }

  /// Returns the number of bytes produced by the compressor so far that have
  /// not yet been read.
  pub fn byte_size(&mut self) -> usize {
    self.writer.byte_size()
  }
}

#[cfg(test)]
mod tests {
  use super::choose_max_n_prefixes;

  #[test]
  fn test_choose_max_n_prefixes() {
    assert_eq!(choose_max_n_prefixes(0, 100), 1);
    assert_eq!(choose_max_n_prefixes(12, 100), 100);
    assert_eq!(choose_max_n_prefixes(12, 1 << 10), 1 << 10);
    assert_eq!(choose_max_n_prefixes(8, 1 << 10), 1 << 6);
    assert_eq!(choose_max_n_prefixes(1, 1 << 10), 1);
    assert_eq!(choose_max_n_prefixes(12, (1 << 12) - 1), 1 << 10);
    assert_eq!(choose_max_n_prefixes(12, 1 << 12), 1 << 11);
    assert_eq!(choose_max_n_prefixes(12, (1 << 14) - 1), 1 << 11);
    assert_eq!(choose_max_n_prefixes(12, 1 << 14), 1 << 12);
    assert_eq!(choose_max_n_prefixes(12, 1 << 20), 1 << 12);
  }
}
