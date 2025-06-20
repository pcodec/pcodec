use crate::constants::{Bitlen, DEFAULT_MAX_PAGE_N};
use crate::errors::{PcoError, PcoResult};
use crate::DEFAULT_COMPRESSION_LEVEL;

/// Specifies how Pco should choose a [`mode`][crate::metadata::Mode] to compress this
/// chunk of data.
///
/// The `Try*` variants almost always use the provided mode, but fall back to
/// `Classic` if the provided mode is especially bad.
/// It is recommended that you only use the `Try*` variants if you know for
/// certain that your numbers benefit from that mode.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum ModeSpec {
  /// Automatically detects a good mode.
  ///
  /// This works well most of the time, but costs some compression time and can
  /// select a bad mode in adversarial cases.
  #[default]
  Auto,
  /// Only uses `Classic` mode.
  Classic,
  /// Tries using `FloatMult` mode with a given `base`.
  ///
  /// Only applies to floating-point types.
  TryFloatMult(f64),
  /// Tries using `FloatQuant` mode with `k` bits of quantization.
  ///
  /// Only applies to floating-point types.
  TryFloatQuant(Bitlen),
  /// Tries using `IntMult` mode with a given `base`.
  ///
  /// Only applies to integer types.
  TryIntMult(u64),
}

/// Specifies how Pco should choose a
/// [`delta encoding`][crate::metadata::DeltaEncoding] to compress this
/// chunk of data.
///
/// The `Try*` variants almost always use the provided encoding, but fall back
/// to `None` if the provided encoding is especially bad.
/// It is recommended that you only use the `Try*` variants if you know for
/// certain that your numbers benefit from delta encoding.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[non_exhaustive]
pub enum DeltaSpec {
  /// Automatically detects a good delta encoding.
  ///
  /// This works well most of the time, but costs some compression time and can
  /// select a bad delta encoding in adversarial cases.
  #[default]
  Auto,
  /// Never uses delta encoding.
  ///
  /// This is best if your data is in a random order or adjacent numbers have
  /// no relation to each other.
  None,
  /// Tries taking nth order consecutive deltas.
  ///
  /// Supports a delta encoding order up to 7.
  /// For instance, 1st order is just regular delta encoding, 2nd is
  /// deltas-of-deltas, etc.
  /// It is legal to use 0th order, but it is identical to `None`.
  TryConsecutive(usize),
  /// Tries delta encoding according to an extra latent variable of "lookback".
  ///
  /// This can improve compression ratio when there are nontrivial patterns in
  /// your numbers, but reduces compression speed substantially.
  TryLookback,
}

// TODO consider adding a "lossiness" spec that allows dropping secondary latent
// vars.
/// All configurations available for a compressor.
///
/// Some, like `delta_encoding_order`, are explicitly stored in the
/// compressed bytes.
/// Others, like `compression_level`, affect compression but are not explicitly
/// stored.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ChunkConfig {
  /// Ranges from 0 to 12 inclusive (default: 8).
  ///
  /// At present,
  /// * Level 0 achieves only a small amount of compression.
  /// * Level 8 achieves very good compression.
  /// * Level 12 achieves marginally better compression than 8.
  ///
  /// The meaning of the compression levels is subject to change with
  /// new releases.
  pub compression_level: usize,
  /// Specifies how the mode should be determined.
  ///
  /// See [`Mode`](crate::metadata::Mode) to understand what modes are.
  pub mode_spec: ModeSpec,
  /// Specifies how delta encoding should be chosen.
  ///
  /// See [`DeltaEncoding`](crate::metadata::DeltaEncoding) to understand what
  /// delta encoding is.
  /// If you would like to automatically choose this once and reuse it for all
  /// chunks, you can create a
  /// [`ChunkDecompressor`][crate::wrapped::ChunkDecompressor] and read the
  /// delta encoding it chose.
  pub delta_spec: DeltaSpec,
  /// Specifies how the chunk should be split into pages (default: equal pages
  /// up to 2^18 numbers each).
  pub paging_spec: PagingSpec,
}

impl Default for ChunkConfig {
  fn default() -> Self {
    Self {
      compression_level: DEFAULT_COMPRESSION_LEVEL,
      mode_spec: ModeSpec::default(),
      delta_spec: DeltaSpec::default(),
      paging_spec: PagingSpec::EqualPagesUpTo(DEFAULT_MAX_PAGE_N),
    }
  }
}

impl ChunkConfig {
  /// Sets [`compression_level`][ChunkConfig::compression_level].
  pub fn with_compression_level(mut self, level: usize) -> Self {
    self.compression_level = level;
    self
  }

  /// Sets [`mode_spec`][ChunkConfig::mode_spec].
  pub fn with_mode_spec(mut self, mode_spec: ModeSpec) -> Self {
    self.mode_spec = mode_spec;
    self
  }

  /// Sets [`delta_spec`][ChunkConfig::delta_spec].
  pub fn with_delta_spec(mut self, delta_spec: DeltaSpec) -> Self {
    self.delta_spec = delta_spec;
    self
  }

  /// Sets [`paging_spec`][ChunkConfig::paging_spec].
  pub fn with_paging_spec(mut self, paging_spec: PagingSpec) -> Self {
    self.paging_spec = paging_spec;
    self
  }
}

/// `PagingSpec` specifies how a chunk is split into pages.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum PagingSpec {
  /// Divide the chunk into equal pages of up to this many numbers.
  ///
  /// For example, with equal pages up to 100,000, a chunk of 150,000
  /// numbers would be divided into 2 pages, each of 75,000 numbers.
  EqualPagesUpTo(usize),
  /// Divide the chunk into the exactly provided counts.
  ///
  /// Will return an InvalidArgument error during compression if
  /// any of the counts are 0 or the sum does not equal the chunk count.
  Exact(Vec<usize>),
}

impl Default for PagingSpec {
  fn default() -> Self {
    Self::EqualPagesUpTo(DEFAULT_MAX_PAGE_N)
  }
}

impl PagingSpec {
  pub(crate) fn n_per_page(&self, n: usize) -> PcoResult<Vec<usize>> {
    let n_per_page = match self {
      // You might think it would be beneficial to do either of these:
      // * greedily fill pages since compressed chunk size seems like a concave
      //   function of chunk_n
      // * limit most pages to full batches for efficiency
      //
      // But in practice compressed chunk size has an inflection point upward
      // at some point, so the first idea doesn't work.
      // And the 2nd idea has only shown mixed/negative results, so I'm leaving
      // this as-is.
      PagingSpec::EqualPagesUpTo(max_page_n) => {
        // Create a sequence of page lengths satisfying these constraints:
        // * All pages have length at most `max_page_n`
        // * The page lengths are approximately equal
        // * As few pages as possible, within the above two constraints.
        if n == 0 {
          return Ok(Vec::new());
        }
        let n_pages = n.div_ceil(*max_page_n);
        let page_n_low = n / n_pages;
        let page_n_high = page_n_low + 1;
        let r = n % n_pages;
        debug_assert!(r == 0 || page_n_high <= *max_page_n);
        let mut res = vec![page_n_low; n_pages];
        res[..r].fill(page_n_high);
        res
      }
      PagingSpec::Exact(n_per_page) => n_per_page.to_vec(),
    };

    let summed_n: usize = n_per_page.iter().sum();
    if summed_n != n {
      return Err(PcoError::invalid_argument(format!(
        "paging spec suggests {} numbers but {} were given",
        summed_n, n,
      )));
    }

    for &page_n in &n_per_page {
      if page_n == 0 {
        return Err(PcoError::invalid_argument(
          "cannot write data page of 0 numbers",
        ));
      }
    }

    Ok(n_per_page)
  }
}
