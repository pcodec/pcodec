use std::cmp::min;
use std::fs::File;
use std::io::Write;

use better_io::{BetterBufRead, BetterBufReader};

use crate::chunk_config::{ChunkConfig, DeltaSpec};
use crate::errors::PcoResult;
use crate::wrapped::{FileCompressor, FileDecompressor, PageDecompressor};
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
    dst = cc.write_chunk_meta(dst)?;
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
