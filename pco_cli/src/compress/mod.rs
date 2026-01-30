use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use crate::input::{InputColumnOpt, InputFileOpt};
use crate::{arrow_handlers, chunk_config_opt, input, utils};

pub mod handler;

/// Compress from a different format into standalone .pco
#[derive(Clone, Debug, Parser)]
pub struct CompressOpt {
  /// Overwrite the output path (if it exists) instead of failing.
  #[arg(long)]
  pub overwrite: bool,
  #[command(flatten)]
  pub input_file: InputFileOpt,
  #[command(flatten)]
  pub input_column: InputColumnOpt,
  #[command(flatten)]
  pub chunk_config: chunk_config_opt::ChunkConfigOpt,

  /// Output .pco path to write to.
  pub path: PathBuf,
}

pub fn compress(opt: CompressOpt) -> Result<()> {
  let schema = input::get_schema(&opt.input_column, &opt.input_file)?;
  let col_idx = utils::find_col_idx(
    &schema,
    opt.input_column.col_idx,
    &opt.input_column.col_name,
  )?;
  let dtype = schema.field(col_idx).data_type();
  let handler = arrow_handlers::from_dtype(dtype)?;
  handler.compress(opt, &schema)
}
