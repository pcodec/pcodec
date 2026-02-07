use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use pco::standalone::{DecompressorItem, FileDecompressor};

use crate::inspect::handler::{CompressionSummary, Output};
use crate::{core_handlers, utils};

pub mod handler;

/// Print metadata about a standalone .pco file.
#[derive(Clone, Debug, Parser)]
pub struct InspectOpt {
  pub path: PathBuf,
}

fn trivial_inspect(opt: &InspectOpt, src: &[u8]) -> Result<()> {
  let start_len = src.len();
  let (fd, src) = FileDecompressor::new(src)?;
  let header_size = start_len - src.len();
  let no_cd = fd.chunk_decompressor::<i32, _>(src)?;
  let src = match no_cd {
    DecompressorItem::Chunk(_) => unreachable!("file was supposed to be trivial"),
    DecompressorItem::EndOfData(src) => src,
  };

  let summary = Output {
    filename: opt.path.to_str().unwrap().to_string(),
    number_type: "<none>".to_string(),
    format_version: fd.format_version().to_string(),
    n: 0,
    n_chunks: 0,
    uncompressed_size: 0,
    compressed: CompressionSummary {
      ratio: 0.0,
      total_size: start_len - src.len(),
      header_size,
      meta_size: 0,
      page_size: 0,
      footer_size: 1,
      unknown_trailing_bytes: src.len(),
    },
    chunk: Vec::new(),
  };
  println!("{}", toml::to_string_pretty(&summary)?);
  Ok(())
}

pub fn inspect(opt: InspectOpt) -> Result<()> {
  let bytes = fs::read(&opt.path)?;
  let Some(dtype) = utils::get_standalone_dtype(&bytes)? else {
    return trivial_inspect(&opt, &bytes);
  };
  let handler = core_handlers::from_dtype(dtype);
  handler.inspect(&opt, &bytes)
}
