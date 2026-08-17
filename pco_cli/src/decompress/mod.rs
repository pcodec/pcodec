use std::fs::OpenOptions;
use std::io::{ErrorKind, Read};
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use pco::data_types::NumberType;
use pco::match_number_enum;

use std::cmp::min;

use better_io::BetterBufReader;
use pco::standalone::{DecompressorItem, FileDecompressor};
use pco::FULL_BATCH_N;

use crate::dtypes::PcoNumber;
use crate::utils;

pub mod column_writers;

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum OutputKind {
  Txt,
  Binary,
}

/// Decompress from standalone .pco into stdout.
#[derive(Clone, Debug, Parser)]
pub struct DecompressOpt {
  #[arg(long)]
  pub limit: Option<usize>,
  #[arg(short, long, default_value = "txt")]
  pub output: OutputKind,

  pub path: PathBuf,
}

pub fn decompress_generic<T: PcoNumber>(opt: &DecompressOpt) -> Result<()> {
  let file = OpenOptions::new().read(true).open(&opt.path)?;
  let src = BetterBufReader::from_read_simple(file);
  let (fd, mut src) = FileDecompressor::new(src)?;

  let mut writer = column_writers::new::<T>(opt)?;
  let mut remaining_limit = opt.limit.unwrap_or(usize::MAX);
  let mut nums = Vec::new();

  loop {
    if remaining_limit == 0 {
      break;
    }

    if let DecompressorItem::Chunk(mut cd) = fd.chunk_decompressor::<T, _>(src)? {
      let n = cd.n();
      let batch_size = min(n, remaining_limit);
      // how many pco should decompress
      let pco_size = (1 + batch_size / FULL_BATCH_N) * FULL_BATCH_N;
      nums.resize(pco_size, T::default());
      let _ = cd.read(&mut nums)?;
      src = cd.into_src();
      let arrow_nums = nums
        .iter()
        .take(batch_size)
        .map(|&x| T::to_arrow_native(x))
        .collect::<Vec<_>>();
      writer.write(arrow_nums)?;
      remaining_limit -= batch_size;
    } else {
      break;
    }
  }

  writer.close()?;
  Ok(())
}

pub fn decompress(opt: DecompressOpt) -> Result<()> {
  let mut initial_bytes = vec![0; pco::standalone::guarantee::header_size() + 1];
  match OpenOptions::new()
    .read(true)
    .open(&opt.path)?
    .read_exact(&mut initial_bytes)
  {
    Ok(()) => (),
    Err(e) if matches!(e.kind(), ErrorKind::UnexpectedEof) => (),
    other => other?,
  };
  let Some(dtype) = utils::get_standalone_dtype(&initial_bytes)? else {
    // file terminated; nothing to decompress
    return Ok(());
  };

  match_number_enum!(
    dtype,
    NumberType<T> => {
      decompress_generic::<T>(&opt)
    }
  )
}
