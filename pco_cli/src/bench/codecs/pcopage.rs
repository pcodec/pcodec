use clap::Parser;
use pco::wrapped::{FileCompressor, FileDecompressor};
use pco::{ChunkConfig, PagingSpec};

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;

// This is designed to be a dumb format that just tests Pco's pagination
// capabilities and performance. It isn't stable.
#[derive(Clone, Debug, Parser)]
pub struct PaginatedPcoConfig {
  #[arg(long, default_value_t = pco::DEFAULT_COMPRESSION_LEVEL)]
  pub level: usize,
  #[arg(long, default_value_t = pco::DEFAULT_MAX_PAGE_N)]
  pub chunk_n: usize,
  #[arg(long, default_value_t = pco::DEFAULT_MAX_PAGE_N)]
  pub page_n: usize,
}

impl CodecInternal for PaginatedPcoConfig {
  fn name(&self) -> &'static str {
    "pcopage"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![
      ("level", self.level.to_string()),
      ("chunk-n", self.chunk_n.to_string()),
      ("page-n", self.page_n.to_string()),
    ]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let n = nums.len();
    let chunk_ns = PagingSpec::EqualPagesUpTo(self.chunk_n)
      .n_per_page(n)
      .unwrap();
    let config = ChunkConfig::default()
      .with_compression_level(self.level)
      .with_paging_spec(PagingSpec::EqualPagesUpTo(self.page_n));

    let fc = FileCompressor::default();
    let mut dst = Vec::new();
    dst.extend((n as u64).to_le_bytes());
    dst.extend((chunk_ns.len() as u32).to_le_bytes());
    fc.write_header(&mut dst).unwrap();

    let mut start = 0;
    for chunk_n in chunk_ns {
      let end = start + chunk_n;

      let mut cc = fc
        .chunk_compressor::<T>(&nums[start..end], &config)
        .unwrap();

      let n_per_page = cc.n_per_page();
      let n_pages = n_per_page.len();
      let additional_size_est = 4
        + cc.chunk_meta_size_hint()
        + (0..n_pages)
          .map(|page_i| 4 + cc.page_size_hint(page_i))
          .sum::<usize>();
      dst.reserve(additional_size_est);
      dst.extend((n_pages as u32).to_le_bytes());
      cc.write_chunk_meta(&mut dst).unwrap();
      for (page_i, page_n) in n_per_page.into_iter().enumerate() {
        dst.extend((page_n as u32).to_le_bytes());
        cc.write_page(page_i, &mut dst).unwrap();
      }

      start = end;
    }

    dst
  }

  fn decompress<T: PcoNumber>(&self, bytes: &[u8]) -> Vec<T> {
    let mut src = bytes;
    let n = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
    src = &src[8..];
    let n_chunks = u32::from_le_bytes(src[0..4].try_into().unwrap()) as usize;
    src = &src[4..];

    let mut dst = Vec::with_capacity(n);
    unsafe {
      dst.set_len(n);
    }

    let (fd, rest) = FileDecompressor::new(src).unwrap();
    src = rest;

    let mut i = 0;
    for _ in 0..n_chunks {
      let n_pages = u32::from_le_bytes(src[0..4].try_into().unwrap()) as usize;
      src = &src[4..];
      let (mut cd, rest) = fd.chunk_decompressor(src).unwrap();
      src = rest;

      for _ in 0..n_pages {
        let page_n = u32::from_le_bytes(src[0..4].try_into().unwrap()) as usize;
        src = &src[4..];
        let mut pd = cd.page_decompressor(src, page_n).unwrap();
        pd.decompress(&mut dst[i..]).unwrap();
        i += page_n;
        src = pd.into_src();
      }
    }

    dst
  }
}
