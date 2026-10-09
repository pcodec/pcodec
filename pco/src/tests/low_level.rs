use std::cmp::min;
use std::fs::File;
use std::io::Write;

use better_io::{BetterBufRead, BetterBufReader};

use crate::chunk_config::{ChunkConfig, DeltaSpec};
use crate::errors::PcoResult;
use crate::macros::match_latent_enum;
use crate::metadata::DynBins;
use crate::wrapped::{
  ChunkDecompressor, DecompressorScratch, FileCompressor, FileDecompressor, PageDecompressor,
};
use crate::{PagingSpec, FULL_BATCH_N};

struct Chunk {
  nums: Vec<u32>,
  config: ChunkConfig,
}

fn decompress_by_batch<R: BetterBufRead>(
  pd: &mut PageDecompressor<u32, R>,
  page_n: usize,
) -> PcoResult<Vec<u32>> {
  let mut nums = vec![0; page_n];
  let mut start = 0;
  loop {
    let end = min(start + FULL_BATCH_N, page_n);
    let batch_size = end - start;
    let progress = pd.read(&mut nums[start..end])?;
    assert_eq!(progress.n_processed, batch_size);
    start = end;
    if end == page_n {
      assert!(progress.finished);
    }
    if progress.finished {
      break;
    }
  }
  Ok(nums)
}

fn test_wrapped_compress<W: Write>(chunks: &[Chunk], dst: W) -> PcoResult<W> {
  let fc = FileCompressor::default();
  let mut dst = fc.write_header(dst)?;

  for chunk in chunks {
    let mut cc = fc.chunk_compressor(&chunk.nums, &chunk.config)?;
    dst = cc.write_meta(dst)?;
    for page_idx in 0..cc.n_per_page().len() {
      dst = cc.write_page(page_idx, dst)?;
    }
  }

  Ok(dst)
}

fn test_wrapped_decompress<R: BetterBufRead>(chunks: &[Chunk], src: R) -> PcoResult<()> {
  let (fd, mut src) = FileDecompressor::new(src)?;

  // antagonistically keep setting the buf read capacity to 0
  for chunk in chunks {
    src.resize_capacity(0);
    let (mut cd, new_src) = fd.chunk_decompressor(src)?;
    src = new_src;

    let mut page_start = 0;
    let n_per_page = chunk.config.paging_spec.n_per_page(chunk.nums.len())?;
    for &page_n in &n_per_page {
      let page_end = page_start + page_n;

      src.resize_capacity(0);
      let mut pd = cd.page_decompressor(src, page_n)?;
      let page_nums = decompress_by_batch(&mut pd, page_n)?;
      src = pd.into_src();

      assert_eq!(&page_nums, &chunk.nums[page_start..page_end]);
      page_start = page_end;
    }
  }

  Ok(())
}

fn test_wrapped(chunks: &[Chunk]) -> PcoResult<()> {
  // IN MEMORY
  let mut compressed = Vec::new();
  test_wrapped_compress(chunks, &mut compressed)?;
  test_wrapped_decompress(chunks, compressed.as_slice())?;

  // ON DISK
  let file_path = std::env::temp_dir().join("pco_test_file");
  let f = File::create(&file_path)?;
  test_wrapped_compress(chunks, f)?;
  let f = File::open(file_path)?;
  let buf_read = BetterBufReader::new(&[], f, 0);
  test_wrapped_decompress(chunks, buf_read)?;

  Ok(())
}

#[test]
fn test_low_level_wrapped() -> PcoResult<()> {
  test_wrapped(&[
    Chunk {
      nums: (0..1700).collect::<Vec<_>>(),
      config: ChunkConfig {
        delta_spec: DeltaSpec::NoOp,
        paging_spec: PagingSpec::EqualPagesUpTo(600),
        ..Default::default()
      },
    },
    Chunk {
      nums: (0..500).collect::<Vec<_>>(),
      config: ChunkConfig {
        delta_spec: DeltaSpec::TryConsecutive(2),
        paging_spec: PagingSpec::Exact(vec![1, 499]),
        ..Default::default()
      },
    },
    Chunk {
      nums: vec![1, 2, 3],
      config: ChunkConfig::default(),
    },
    Chunk {
      nums: vec![1, 2, 3],
      config: ChunkConfig {
        paging_spec: PagingSpec::EqualPagesUpTo(1),
        ..Default::default()
      },
    },
  ])
}

fn pseudo_random(n: usize, seed: u64, modulus: u64) -> Vec<u32> {
  let mut state = seed;
  (0..n)
    .map(|_| {
      state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
      ((state >> 33) % modulus) as u32
    })
    .collect()
}

// mostly small numbers with an occasional large one, so several bins
fn mixed_magnitudes(n: usize) -> Vec<u32> {
  let small = pseudo_random(n, 6, 1 << 3);
  let large = pseudo_random(n, 7, 1 << 28);
  (0..n)
    .map(|i| if i % 7 == 0 { large[i] } else { small[i] })
    .collect()
}

// (number of bins, offset bits of the first bin) of the primary latent var
fn primary_bins(cd: &ChunkDecompressor<u32>) -> (usize, u32) {
  match_latent_enum!(
    &cd.meta().per_latent_var.primary.bins,
    DynBins<L>(bins) => { (bins.len(), bins.first().map_or(0, |bin| bin.offset_bits)) }
  )
}

#[test]
fn test_scratch_reused_across_chunks() -> PcoResult<()> {
  let paging_spec = PagingSpec::EqualPagesUpTo(300);
  let chunks = [
    Chunk {
      nums: mixed_magnitudes(1000),
      config: ChunkConfig {
        delta_spec: DeltaSpec::NoOp,
        paging_spec: paging_spec.clone(),
        ..Default::default()
      },
    },
    Chunk {
      nums: pseudo_random(1000, 1, 1 << 4),
      config: ChunkConfig {
        delta_spec: DeltaSpec::NoOp,
        paging_spec: paging_spec.clone(),
        ..Default::default()
      },
    },
    Chunk {
      nums: pseudo_random(1000, 2, 1 << 20),
      config: ChunkConfig {
        delta_spec: DeltaSpec::NoOp,
        paging_spec: paging_spec.clone(),
        ..Default::default()
      },
    },
    Chunk {
      nums: mixed_magnitudes(1000),
      config: ChunkConfig {
        delta_spec: DeltaSpec::NoOp,
        paging_spec: paging_spec.clone(),
        ..Default::default()
      },
    },
    Chunk {
      nums: pseudo_random(1000, 3, 1 << 20),
      config: ChunkConfig {
        delta_spec: DeltaSpec::NoOp,
        paging_spec: paging_spec.clone(),
        ..Default::default()
      },
    },
    Chunk {
      nums: pseudo_random(40, 4, 1 << 30).repeat(25),
      config: ChunkConfig {
        delta_spec: DeltaSpec::TryLookback,
        paging_spec: paging_spec.clone(),
        ..Default::default()
      },
    },
  ];
  let mut compressed = Vec::new();
  test_wrapped_compress(&chunks, &mut compressed)?;

  // One scratch across every chunk: single-bin chunks with different offset
  // widths, multi-bin chunks between them, a single-bin chunk returning to an
  // earlier width, and a chunk with a delta latent var.
  let mut scratch = DecompressorScratch::new();
  let (fd, mut src) = FileDecompressor::new(compressed.as_slice())?;
  let mut shapes = Vec::new();
  for chunk in &chunks {
    let (cd, new_src) = fd.chunk_decompressor::<u32, _>(src)?;
    src = new_src;
    shapes.push((
      primary_bins(&cd),
      cd.meta().per_latent_var.delta.is_some(),
    ));

    let mut page_start = 0;
    for page_n in chunk.config.paging_spec.n_per_page(chunk.nums.len())? {
      let mut pd = cd.page_decompressor_with_scratch(&mut scratch, src, page_n)?;
      let page_nums = decompress_by_batch(&mut pd, page_n)?;
      src = pd.into_src();
      assert_eq!(
        &page_nums,
        &chunk.nums[page_start..page_start + page_n]
      );
      page_start += page_n;
    }
  }

  // make sure the chunks have the shapes this test is about
  assert!(shapes[0].0 .0 > 1);
  assert_eq!(shapes[1].0 .0, 1);
  assert_eq!(shapes[2].0 .0, 1);
  assert_ne!(shapes[1].0 .1, shapes[2].0 .1);
  assert!(shapes[3].0 .0 > 1);
  // the same single-bin width as chunk 2, after a multi-bin chunk overwrote
  // the scratch's offset widths
  assert_eq!(shapes[4].0, shapes[2].0);
  assert!(shapes[5].1);
  Ok(())
}

#[test]
fn test_concurrent_pages_from_shared_chunk() -> PcoResult<()> {
  let nums = pseudo_random(20_000, 5, 1 << 24);
  let chunk = Chunk {
    nums,
    config: ChunkConfig {
      paging_spec: PagingSpec::EqualPagesUpTo(1000),
      ..Default::default()
    },
  };
  let mut compressed = Vec::new();
  test_wrapped_compress(std::slice::from_ref(&chunk), &mut compressed)?;

  let (fd, src) = FileDecompressor::new(compressed.as_slice())?;
  let (cd, mut src) = fd.chunk_decompressor::<u32, _>(src)?;
  let mut scratch = DecompressorScratch::new();
  let mut pages = Vec::new();
  let mut page_start = 0;
  for page_n in chunk.config.paging_spec.n_per_page(chunk.nums.len())? {
    // find where this page ends by decompressing it once
    let mut pd = cd.page_decompressor_with_scratch(&mut scratch, src, page_n)?;
    decompress_by_batch(&mut pd, page_n)?;
    let rest = pd.into_src();
    pages.push((
      &src[..src.len() - rest.len()],
      page_start,
      page_n,
    ));
    src = rest;
    page_start += page_n;
  }

  let cd = &cd;
  let pages = &pages;
  let nums = &chunk.nums;
  std::thread::scope(|s| {
    for t in 0..8 {
      s.spawn(move || {
        let mut scratch = DecompressorScratch::new();
        for i in 0..pages.len() {
          let (page, page_start, page_n) = pages[(i + t) % pages.len()];
          let mut pd = cd
            .page_decompressor_with_scratch(&mut scratch, page, page_n)
            .unwrap();
          let page_nums = decompress_by_batch(&mut pd, page_n).unwrap();
          assert_eq!(
            &page_nums,
            &nums[page_start..page_start + page_n]
          );
        }
      });
    }
  });
  Ok(())
}
