use std::cmp::min;
use std::fs::File;
use std::io::Write;

use better_io::{BetterBufRead, BetterBufReader};

use crate::chunk_config::{ChunkConfig, DeltaSpec, ModeSpec};
use crate::data_types::Number;
use crate::errors::{ErrorKind, PcoResult};
use crate::macros::match_latent_enum;
use crate::metadata::{DeltaEncoding, DynBins, Mode};
use crate::wrapped::ChunkDecompressor;
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

fn config(mode_spec: ModeSpec, delta_spec: DeltaSpec, page_n: usize) -> ChunkConfig {
  ChunkConfig {
    mode_spec,
    delta_spec,
    paging_spec: PagingSpec::EqualPagesUpTo(page_n),
    ..Default::default()
  }
}

// (chunk decompressor, [(page bytes, page's numbers)])
type ChunkWithPages<'a, T> = (ChunkDecompressor<T>, Vec<(Vec<u8>, &'a [T])>);

fn chunk_with_pages<'a, T: Number>(
  nums: &'a [T],
  config: &ChunkConfig,
) -> PcoResult<ChunkWithPages<'a, T>> {
  let fc = FileCompressor::default();
  let header = fc.write_header(Vec::new())?;
  let mut cc = fc.chunk_compressor(nums, config)?;
  let meta = cc.write_meta(Vec::new())?;
  let mut pages = Vec::new();
  let mut page_start = 0;
  for (page_idx, &page_n) in cc.n_per_page().iter().enumerate() {
    let page = cc.write_page(page_idx, Vec::new())?;
    pages.push((page, &nums[page_start..page_start + page_n]));
    page_start += page_n;
  }
  let (fd, _) = FileDecompressor::new(header.as_slice())?;
  let (cd, _) = fd.chunk_decompressor(meta.as_slice())?;
  Ok((cd, pages))
}

// one scratch, made for the first chunk, for every page of every chunk
fn decompress_reusing_scratch(chunks: &[Chunk]) -> PcoResult<Vec<ChunkDecompressor<u32>>> {
  let mut scratch = None;
  let mut cds = Vec::new();
  for chunk in chunks {
    let (cd, pages) = chunk_with_pages(&chunk.nums, &chunk.config)?;
    let scratch = scratch.get_or_insert_with(|| cd.new_scratch());
    for (page, nums) in pages {
      let mut pd = cd.page_decompressor_with_scratch(scratch, page.as_slice(), nums.len())?;
      assert_eq!(
        decompress_by_batch(&mut pd, nums.len())?,
        nums
      );
    }
    cds.push(cd);
  }
  Ok(cds)
}

#[test]
fn test_scratch_reused_across_chunks() -> PcoResult<()> {
  // multi-bin batches (full or partial) leave offset widths in the scratch
  //   -> the next single-bin chunk must not read them
  let classic = |nums, page_n| Chunk {
    nums,
    config: config(ModeSpec::Classic, DeltaSpec::NoOp, page_n),
  };
  let mixed = (0..1000)
    .map(|i| if i % 7 == 0 { i << 20 } else { i % 8 })
    .collect::<Vec<_>>();
  let cds = decompress_reusing_scratch(&[
    classic(pseudo_random(1000, 1, 1 << 4), 300),
    classic(mixed[..768].to_vec(), 256),
    classic(pseudo_random(1000, 2, 1 << 20), 300),
    classic(mixed, 300),
    classic(pseudo_random(1000, 3, 1 << 20), 300),
  ])?;
  let multi_bin = |cd: &ChunkDecompressor<u32>| {
    match_latent_enum!(
      &cd.meta().per_latent_var.primary.bins,
      DynBins<L>(bins) => { bins.len() > 1 }
    )
  };
  assert_eq!(
    cds.iter().map(multi_bin).collect::<Vec<_>>(),
    [false, true, false, true, false]
  );

  // big lookback window / dict, then a small one with partial final batches
  //   -> the small one must not see the big one's lookbacks / indices
  let lookback = |nums, page_n| Chunk {
    nums,
    config: config(
      ModeSpec::Classic,
      DeltaSpec::TryLookback,
      page_n,
    ),
  };
  let cds = decompress_reusing_scratch(&[
    lookback(
      pseudo_random(2000, 11, 1 << 30).repeat(10),
      20000,
    ),
    lookback(
      pseudo_random(3, 12, 1 << 30).repeat(100),
      150,
    ),
  ])?;
  assert!(cds
    .iter()
    .all(|cd| cd.meta().per_latent_var.delta.is_some()));
  let dict = |dict: Vec<u32>, n, page_n| Chunk {
    nums: (0..n).map(|i| dict[(i * 7919) % dict.len()]).collect(),
    config: config(ModeSpec::TryDict, DeltaSpec::NoOp, page_n),
  };
  let cds = decompress_reusing_scratch(&[
    dict(
      pseudo_random(1000, 21, 1 << 30),
      20000,
      20000,
    ),
    dict(pseudo_random(4, 22, 1 << 30), 300, 150),
  ])?;
  assert!(cds.iter().all(|cd| matches!(cd.meta().mode, Mode::Dict(_))));

  // a secondary latent var, and delta encodings without a delta latent var
  let int_mult = |seed| Chunk {
    nums: pseudo_random(1000, seed, 1 << 20)
      .iter()
      .map(|&x| x * 1000 + x % 3)
      .collect(),
    config: config(
      ModeSpec::TryIntMult(1000),
      DeltaSpec::NoOp,
      300,
    ),
  };
  let cds = decompress_reusing_scratch(&[int_mult(31), int_mult(32)])?;
  assert!(cds
    .iter()
    .all(|cd| cd.meta().per_latent_var.secondary.is_some()));
  let smooth = |seed, delta_spec| Chunk {
    nums: pseudo_random(1000, seed, 1 << 4)
      .iter()
      .enumerate()
      .map(|(i, &x)| (i * i) as u32 + x)
      .collect(),
    config: config(ModeSpec::Classic, delta_spec, 300),
  };
  let consecutive = DeltaSpec::TryConsecutive(2);
  let cds =
    decompress_reusing_scratch(&[smooth(41, consecutive.clone()), smooth(42, consecutive)])?;
  assert!(cds.iter().all(|cd| matches!(
    cd.meta().delta_encoding,
    DeltaEncoding::Consecutive { .. }
  )));
  let conv1 = DeltaSpec::TryConv1(2);
  let cds = decompress_reusing_scratch(&[smooth(43, conv1.clone()), smooth(44, conv1)])?;
  assert!(cds.iter().all(|cd| matches!(
    cd.meta().delta_encoding,
    DeltaEncoding::Conv1(_)
  )));
  Ok(())
}

#[test]
fn test_scratch_compatibility() -> PcoResult<()> {
  // same number type, other mode / delta -> other latent vars, and Auto specs
  // pick them per chunk
  let random = pseudo_random(1000, 41, 1 << 30);
  let u64s = random.iter().map(|&x| x as u64).collect::<Vec<_>>();
  let f64s = random.iter().map(|&x| x as f64 / 7.0).collect::<Vec<_>>();
  let mults = u64s.iter().map(|&x| x * 1000 + x % 3).collect::<Vec<_>>();
  let few = (0..1000).map(|i| u64s[i % 8] << 20).collect::<Vec<_>>();
  let repeats = u64s[..40].repeat(25);
  let classic = config(ModeSpec::Classic, DeltaSpec::NoOp, 1000);
  let (u64_cd, _) = chunk_with_pages(&u64s, &classic)?;
  let (f64_cd, f64_pages) = chunk_with_pages(&f64s, &classic)?;
  let (u32_cd, _) = chunk_with_pages(&random, &classic)?;
  let (lookback, _) = chunk_with_pages(
    &repeats,
    &config(
      ModeSpec::Classic,
      DeltaSpec::TryLookback,
      1000,
    ),
  )?;
  let (int_mult, _) = chunk_with_pages(
    &mults,
    &config(
      ModeSpec::TryIntMult(1000),
      DeltaSpec::NoOp,
      1000,
    ),
  )?;
  let (dict, _) = chunk_with_pages(
    &few,
    &config(ModeSpec::TryDict, DeltaSpec::NoOp, 1000),
  )?;
  // make sure the chunks have the latent vars this test is about
  assert!(lookback.meta().per_latent_var.delta.is_some());
  assert!(int_mult.meta().per_latent_var.secondary.is_some());
  // dict mode's primary latents are u32 indices even for u64 numbers
  assert!(matches!(dict.meta().mode, Mode::Dict(_)));

  // classic u64 scratch -> f64 (u64 latents too): fits
  //                     -> u32, or other latent vars: rejected
  let mut scratch = u64_cd.new_scratch();
  assert!(f64_cd.is_compatible_with_scratch(&scratch));
  let mut nums = vec![0.0; 1000];
  f64_cd
    .page_decompressor_with_scratch(&mut scratch, f64_pages[0].0.as_slice(), 1000)?
    .read(&mut nums)?;
  assert_eq!(nums, f64s);
  assert!(!u32_cd.is_compatible_with_scratch(&scratch));
  for cd in [&lookback, &int_mult, &dict] {
    assert!(!cd.is_compatible_with_scratch(&scratch));
    assert!(!u64_cd.is_compatible_with_scratch(&cd.new_scratch()));
    let res = cd.page_decompressor_with_scratch(&mut scratch, &[][..], 1000);
    assert!(matches!(
      res.err().map(|err| err.kind),
      Some(ErrorKind::InvalidArgument)
    ));
  }
  Ok(())
}

#[test]
fn test_concurrent_pages_from_shared_chunk() -> PcoResult<()> {
  let nums = pseudo_random(20_000, 5, 1 << 24);
  let (cd, pages) = chunk_with_pages(
    &nums,
    &config(ModeSpec::Auto, DeltaSpec::Auto, 1000),
  )?;
  let (cd, pages) = (&cd, &pages);
  std::thread::scope(|s| {
    for t in 0..8 {
      s.spawn(move || {
        let mut scratch = cd.new_scratch();
        for i in 0..pages.len() {
          let (page, nums) = pages[(i + t) % pages.len()].clone();
          let mut pd = cd
            .page_decompressor_with_scratch(&mut scratch, page.as_slice(), nums.len())
            .unwrap();
          assert_eq!(
            decompress_by_batch(&mut pd, nums.len()).unwrap(),
            nums
          );
        }
      });
    }
  });
  Ok(())
}
