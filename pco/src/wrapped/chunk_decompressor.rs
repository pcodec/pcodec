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

  use divan::counter::ItemsCount;
  use divan::Bencher;
  use rand_xoshiro::rand_core::{RngCore, SeedableRng};
  use rand_xoshiro::Xoroshiro128PlusPlus;

  use super::*;
  use crate::bench_utils::BENCH_N;
  use crate::wrapped::{FileCompressor, FileDecompressor};
  use crate::{ChunkConfig, PagingSpec};

  const PAGE_N: usize = 1024;

  struct Chunk {
    cd: ChunkDecompressor<u64>,
    pages: Vec<Vec<u8>>,
  }

  // One chunk of BENCH_N heavy-tailed numbers in pages of PAGE_N, with each
  // page's bytes kept separately so any page can be decompressed on its own.
  fn chunk() -> Chunk {
    let mut rng = Xoroshiro128PlusPlus::seed_from_u64(0);
    let nums = (0..BENCH_N)
      .map(|_| {
        let uniform = (rng.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        (1000.0 / (uniform + 1e-9).powf(0.7)) as u64
      })
      .collect::<Vec<_>>();
    let config = ChunkConfig::default().with_paging_spec(PagingSpec::EqualPagesUpTo(PAGE_N));
    let fc = FileCompressor::default();
    let header = fc.write_header(Vec::new()).unwrap();
    let mut cc = fc.chunk_compressor(&nums, &config).unwrap();
    let meta = cc.write_meta(Vec::new()).unwrap();
    let pages = (0..cc.n_per_page().len())
      .map(|page_idx| cc.write_page(page_idx, Vec::new()).unwrap())
      .collect();
    let (fd, _) = FileDecompressor::new(header.as_slice()).unwrap();
    let (cd, _) = fd.chunk_decompressor(meta.as_slice()).unwrap();
    Chunk { cd, pages }
  }

  thread_local! {
    static PAGE_IDX: Cell<usize> = const { Cell::new(0) };
    static SCRATCH: RefCell<DecompressorScratch> = RefCell::new(DecompressorScratch::new());
  }

  fn next_page(chunk: &Chunk) -> &[u8] {
    let idx = PAGE_IDX.get();
    PAGE_IDX.set(idx + 1);
    &chunk.pages[idx % chunk.pages.len()]
  }

  fn read_page<R: BetterBufRead>(mut pd: PageDecompressor<u64, R>) -> u64 {
    let mut dst = [0_u64; PAGE_N];
    pd.read(&mut dst).unwrap();
    dst[PAGE_N - 1]
  }

  /// One decoder walking pages, reusing its own scratch: the baseline.
  #[divan::bench]
  fn page_own_scratch(bencher: Bencher) {
    let mut chunk = chunk();
    bencher.counter(ItemsCount::new(PAGE_N)).bench_local(|| {
      let idx = PAGE_IDX.get();
      PAGE_IDX.set(idx + 1);
      let page = &chunk.pages[idx % chunk.pages.len()];
      read_page(chunk.cd.page_decompressor(page.as_slice(), PAGE_N).unwrap())
    });
  }

  /// Decompressing pages of one chunk from several threads by cloning the
  /// chunk decompressor for each page.
  #[divan::bench(threads = [1, 8, 32])]
  fn page_clone_per_decode(bencher: Bencher) {
    let chunk = chunk();
    bencher.counter(ItemsCount::new(PAGE_N)).bench(|| {
      let mut cd = chunk.cd.clone();
      read_page(cd.page_decompressor(next_page(&chunk), PAGE_N).unwrap())
    });
  }

  /// Decompressing pages of one shared chunk decompressor from several
  /// threads, each with its own scratch.
  #[divan::bench(threads = [1, 8, 32])]
  fn page_shared_with_scratch(bencher: Bencher) {
    let chunk = chunk();
    bencher.counter(ItemsCount::new(PAGE_N)).bench(|| {
      SCRATCH.with_borrow_mut(|scratch| {
        read_page(
          chunk
            .cd
            .page_decompressor_with_scratch(scratch, next_page(&chunk), PAGE_N)
            .unwrap(),
        )
      })
    });
  }
}
