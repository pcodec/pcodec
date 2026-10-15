use better_io::BetterBufRead;
use std::marker::PhantomData;

use crate::chunk_latent_decompressor::{DynChunkLatentDecompressor, DynLatentScratch};
use crate::data_types::Number;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::{ChunkMeta, LatentVarKey, PerLatentVar};
use crate::wrapped::PageDecompressor;

#[derive(Clone, Debug)]
pub struct ChunkDecompressorInner {
  pub(crate) meta: ChunkMeta,
  pub(crate) per_latent_var: PerLatentVar<DynChunkLatentDecompressor>,
}

impl ChunkDecompressorInner {
  fn new(meta: ChunkMeta) -> PcoResult<Self> {
    let per_latent_var = meta.per_latent_var.as_ref().map_result(|key, latent_var| {
      let delta_encoding = meta.delta_encoding.for_latent_var(key);
      DynChunkLatentDecompressor::create(latent_var, delta_encoding)
    })?;

    Ok(Self {
      meta,
      per_latent_var,
    })
  }

  pub fn n_latents_per_delta_state(&self) -> usize {
    self
      .meta
      .delta_encoding
      .for_latent_var(LatentVarKey::Primary)
      .n_latents_per_state()
  }
}

/// Working memory for decompressing pages, separate from any chunk.
///
/// A [`ChunkDecompressor`] owns one and reuses it for every page it
/// decompresses via [`ChunkDecompressor::page_decompressor`]. Supplying your
/// own via [`ChunkDecompressor::page_decompressor_with_scratch`] lets several
/// threads decompress pages of the same chunk at once from a shared
/// `&ChunkDecompressor`, each with its own scratch, without cloning the chunk
/// decompressor.
///
/// A scratch is not tied to a chunk or number type: it adapts to whichever
/// chunk it is used with, and reusing it across pages and chunks avoids
/// reallocating it. It is a few KiB per latent variable once used.
#[derive(Clone, Debug, Default)]
pub struct DecompressorScratch {
  pub(crate) delta: Option<DynLatentScratch>,
  pub(crate) primary: Option<DynLatentScratch>,
  pub(crate) secondary: Option<DynLatentScratch>,
}

impl DecompressorScratch {
  /// Creates an empty scratch; its buffers are allocated on first use.
  pub fn new() -> Self {
    Self::default()
  }

  pub(crate) fn prepare(&mut self, cd: &ChunkDecompressorInner) {
    fn prepare_slot(slot: &mut Option<DynLatentScratch>, cld: Option<&DynChunkLatentDecompressor>) {
      if let Some(cld) = cld {
        if !slot
          .as_ref()
          .is_some_and(|scratch| cld.scratch_matches(scratch))
        {
          *slot = Some(cld.new_scratch());
        }
      }
    }

    let per_latent_var = &cd.per_latent_var;
    prepare_slot(
      &mut self.delta,
      per_latent_var.delta.as_ref(),
    );
    prepare_slot(
      &mut self.primary,
      Some(&per_latent_var.primary),
    );
    prepare_slot(
      &mut self.secondary,
      per_latent_var.secondary.as_ref(),
    );
  }
}

/// Holds metadata about a chunk and can produce page decompressors.
#[derive(Clone, Debug)]
pub struct ChunkDecompressor<T: Number> {
  pub(crate) inner: ChunkDecompressorInner,
  pub(crate) scratch: DecompressorScratch,
  phantom: PhantomData<T>,
}

impl<T: Number> ChunkDecompressor<T> {
  pub(crate) fn new(meta: ChunkMeta) -> PcoResult<Self> {
    if !T::mode_is_valid(&meta.mode) {
      return Err(PcoError::corruption(format!(
        "invalid mode for {} number type: {:?}",
        std::any::type_name::<T>(),
        meta.mode
      )));
    }

    ChunkDecompressorInner::new(meta).map(|cd| Self {
      inner: cd,
      scratch: DecompressorScratch::new(),
      phantom: PhantomData,
    })
  }

  /// Returns pre-computed information about the chunk.
  pub fn meta(&self) -> &ChunkMeta {
    &self.inner.meta
  }

  /// Reads metadata for a page and returns a `PageDecompressor` and the
  /// remaining input.
  ///
  /// Will return an error if corruptions or insufficient data are found.
  ///
  /// Even though this takes `&mut self`, the page decompressor only mutates the
  /// chunk decompressor's scratch buffers and has no effect on the
  /// decompression of later pages.
  /// To decompress pages of one chunk concurrently, use
  /// [`page_decompressor_with_scratch`][Self::page_decompressor_with_scratch].
  pub fn page_decompressor<R: BetterBufRead>(
    &mut self,
    src: R,
    n: usize,
  ) -> PcoResult<PageDecompressor<'_, T, R>> {
    PageDecompressor::<T, R>::new(src, &self.inner, &mut self.scratch, n)
  }

  /// Like [`page_decompressor`][Self::page_decompressor], but uses the
  /// given scratch instead of this chunk decompressor's own, so it only needs
  /// `&self`.
  ///
  /// This lets several threads decompress pages of the same chunk at once,
  /// each with its own [`DecompressorScratch`], without cloning the chunk
  /// decompressor.
  pub fn page_decompressor_with_scratch<'a, R: BetterBufRead>(
    &'a self,
    scratch: &'a mut DecompressorScratch,
    src: R,
    n: usize,
  ) -> PcoResult<PageDecompressor<'a, T, R>> {
    PageDecompressor::<T, R>::new(src, &self.inner, scratch, n)
  }
}

#[cfg(feature = "bench")]
mod micro {
  use std::cell::{Cell, RefCell};
  use std::sync::atomic::{AtomicU64, Ordering};
  use std::sync::LazyLock;

  use divan::counter::ItemsCount;
  use divan::Bencher;
  use rand_xoshiro::rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoroshiro128PlusPlus;

  use super::*;
  use crate::wrapped::{FileCompressor, FileDecompressor};
  use crate::{ChunkConfig, PagingSpec};

  const N: usize = 1 << 16;
  const MAX_PAGE_N: usize = 1024;
  // (data, rows per page). Uniform data compresses to a single bin, so pages
  // are nothing but offsets; heavy-tailed data gets 20 bins and a 2^10-state
  // ANS table at the default level.
  const CASES: [&str; 4] = ["1bin/256", "1bin/1024", "20bins/256", "20bins/1024"];

  struct Chunk {
    cd: ChunkDecompressor<u64>,
    pages: Vec<Vec<u8>>,
    page_n: usize,
  }

  // One chunk of N numbers in equal pages, with each page's bytes kept
  // separately so any page can be decompressed on its own.
  fn chunk(case: &str, seed: u64) -> Chunk {
    let (data, page_n) = case.split_once('/').unwrap();
    let page_n = page_n.parse().unwrap();
    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(seed);
    let nums = (0..N)
      .map(|_| {
        let r = rng.next_u64();
        match data {
          "1bin" => r >> 24,
          _ => {
            let uniform = (r >> 11) as f64 / (1_u64 << 53) as f64;
            (1000.0 / (uniform + 1e-9).powf(0.7)) as u64
          }
        }
      })
      .collect::<Vec<_>>();
    let config = ChunkConfig::default().with_paging_spec(PagingSpec::EqualPagesUpTo(page_n));
    let fc = FileCompressor::default();
    let header = fc.write_header(Vec::new()).unwrap();
    let mut cc = fc.chunk_compressor(&nums, &config).unwrap();
    let meta = cc.write_meta(Vec::new()).unwrap();
    let pages = (0..cc.n_per_page().len())
      .map(|page_idx| cc.write_page(page_idx, Vec::new()).unwrap())
      .collect();
    let (fd, _) = FileDecompressor::new(header.as_slice()).unwrap();
    let (cd, _) = fd.chunk_decompressor(meta.as_slice()).unwrap();
    Chunk { cd, pages, page_n }
  }

  thread_local! {
    static PAGE_IDX: Cell<usize> = const { Cell::new(0) };
    static SCRATCH: RefCell<DecompressorScratch> = RefCell::new(DecompressorScratch::new());
  }

  fn next_page_idx(chunk: &Chunk) -> usize {
    let idx = PAGE_IDX.get();
    PAGE_IDX.set(idx + 1);
    idx % chunk.pages.len()
  }

  fn read_page<R: BetterBufRead>(mut pd: PageDecompressor<u64, R>, page_n: usize) -> u64 {
    let mut dst = [0_u64; MAX_PAGE_N];
    pd.read(&mut dst[..page_n]).unwrap();
    dst[page_n - 1]
  }

  /// One decoder walking pages, reusing its own scratch: the existing path.
  #[divan::bench(args = CASES)]
  fn page_own_scratch(bencher: Bencher, case: &str) {
    let mut chunk = chunk(case, 0);
    let page_n = chunk.page_n;
    bencher.counter(ItemsCount::new(page_n)).bench_local(|| {
      let page = &chunk.pages[next_page_idx(&chunk)];
      read_page(
        chunk.cd.page_decompressor(page.as_slice(), page_n).unwrap(),
        page_n,
      )
    });
  }

  /// Decompressing pages of one shared chunk decompressor from several
  /// threads, each with its own scratch.
  #[divan::bench(threads = [1, 8, 32], args = CASES)]
  fn page_shared_with_scratch(bencher: Bencher, case: &str) {
    let chunk = chunk(case, 0);
    let page_n = chunk.page_n;
    bencher.counter(ItemsCount::new(page_n)).bench(|| {
      let page = &chunk.pages[next_page_idx(&chunk)];
      SCRATCH.with_borrow_mut(|scratch| {
        read_page(
          chunk
            .cd
            .page_decompressor_with_scratch(scratch, page.as_slice(), page_n)
            .unwrap(),
          page_n,
        )
      })
    });
  }

  // An index of many chunks serving point lookups: each lookup decompresses
  // a random page of a random chunk. The table benches run for a minimum
  // time, since a per-thread clone cache only warms up after a few hundred
  // lookups and by default divan takes 100 samples of one lookup each.
  const TABLE_CHUNKS: usize = 256;
  const TABLE_PAGE_N: usize = 1024;
  static TABLE: LazyLock<Vec<Chunk>> = LazyLock::new(|| {
    (0..TABLE_CHUNKS as u64)
      .map(|seed| chunk(&format!("20bins/{TABLE_PAGE_N}"), seed))
      .collect()
  });

  // a thread's clones of table chunk decompressors, direct-mapped by chunk
  // index
  type CloneCache = Vec<Option<(usize, ChunkDecompressor<u64>)>>;

  thread_local! {
    static LOOKUP_RNG: RefCell<Xoroshiro128PlusPlus> = RefCell::new({
      static NEXT_SEED: AtomicU64 = AtomicU64::new(0);
      Xoroshiro128PlusPlus::seed_from_u64(NEXT_SEED.fetch_add(1, Ordering::Relaxed))
    });
    static CLONE_CACHE: RefCell<CloneCache> = const { RefCell::new(Vec::new()) };
  }

  fn random_lookup() -> (usize, &'static Chunk, &'static [u8]) {
    let r = LOOKUP_RNG.with_borrow_mut(|rng| rng.next_u64());
    let chunk_idx = r as usize % TABLE_CHUNKS;
    let chunk = &TABLE[chunk_idx];
    let page = &chunk.pages[(r >> 32) as usize % chunk.pages.len()];
    (chunk_idx, chunk, page)
  }

  /// Point lookups across a table, cloning the chunk decompressor per page.
  #[divan::bench(threads = [1, 8, 32], min_time = 2)]
  fn table_clone_per_decode(bencher: Bencher) {
    LazyLock::force(&TABLE);
    bencher.counter(ItemsCount::new(TABLE_PAGE_N)).bench(|| {
      let (_, chunk, page) = random_lookup();
      let mut cd = chunk.cd.clone();
      read_page(
        cd.page_decompressor(page, chunk.page_n).unwrap(),
        chunk.page_n,
      )
    });
  }

  /// Point lookups across a table, with each thread keeping clones of up to
  /// SLOTS chunk decompressors and cloning on a miss.
  #[divan::bench(threads = [1, 8, 32], consts = [16, 64, 256], min_time = 2)]
  fn table_clone_cache<const SLOTS: usize>(bencher: Bencher) {
    LazyLock::force(&TABLE);
    bencher.counter(ItemsCount::new(TABLE_PAGE_N)).bench(|| {
      let (chunk_idx, chunk, page) = random_lookup();
      CLONE_CACHE.with_borrow_mut(|cache| {
        if cache.len() != SLOTS {
          *cache = (0..SLOTS).map(|_| None).collect();
        }
        let slot = &mut cache[chunk_idx % SLOTS];
        if slot.as_ref().is_none_or(|(idx, _)| *idx != chunk_idx) {
          *slot = Some((chunk_idx, chunk.cd.clone()));
        }
        let (_, cd) = slot.as_mut().unwrap();
        read_page(
          cd.page_decompressor(page, chunk.page_n).unwrap(),
          chunk.page_n,
        )
      })
    });
  }

  /// Point lookups across a table, sharing every chunk decompressor, with one
  /// scratch per thread.
  #[divan::bench(threads = [1, 8, 32], min_time = 2)]
  fn table_shared_with_scratch(bencher: Bencher) {
    LazyLock::force(&TABLE);
    bencher.counter(ItemsCount::new(TABLE_PAGE_N)).bench(|| {
      let (_, chunk, page) = random_lookup();
      SCRATCH.with_borrow_mut(|scratch| {
        read_page(
          chunk
            .cd
            .page_decompressor_with_scratch(scratch, page, chunk.page_n)
            .unwrap(),
          chunk.page_n,
        )
      })
    });
  }
}
