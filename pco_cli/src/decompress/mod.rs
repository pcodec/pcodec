use std::fs::OpenOptions;
use std::io::{ErrorKind, Read};
use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use pco::data_types::NumberType;
use pco::match_number_enum;

use crate::utils;

pub mod handler;

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
      handler::decompress::<T>(&opt)
    }
  )
}
