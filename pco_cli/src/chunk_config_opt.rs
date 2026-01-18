use clap::Parser;

use pco::{ChunkConfig, DeltaSpec, ModeSpec, PagingSpec};

use crate::parse;

#[derive(Clone, Debug, Parser)]
pub struct ChunkConfigOpt {
  /// Compression level.
  #[arg(long, default_value = "8")]
  pub level: usize,
  /// Can be "Auto", "NoOp", "Consecutive@<order>", "Lookback", or
  /// "Conv1@<order>".
  #[arg(long, default_value = "Auto", value_parser = parse::delta_spec)]
  pub delta: DeltaSpec,
  /// Can be "Auto", "Classic", "Dict", "FloatMult@<base>", "FloatQuant@<k>", or
  /// "IntMult@<base>".
  ///
  /// Specs other than Auto and Classic will try the given mode and fall back to
  /// classic if the given mode is especially bad.
  #[arg(long, default_value = "Auto", value_parser = parse::mode_spec)]
  pub mode: ModeSpec,
  #[arg(long, default_value_t = pco::DEFAULT_MAX_PAGE_N)]
  pub chunk_n: usize,
  // We can't use default value here because clap will force it to be a no-args
  // flag, and that doesn't work with the k=v args we use in bench
  /// Can be "t" or "f".
  ///
  /// This must be "t" to enable compression of 8-bit data types.
  #[arg(long, value_parser = parse::boolean)]
  enable_8_bit: Option<bool>,
}

impl ChunkConfigOpt {
  pub fn enable_8_bit(&self) -> bool {
    self.enable_8_bit.unwrap_or(false)
  }
}

impl From<ChunkConfigOpt> for ChunkConfig {
  fn from(opt: ChunkConfigOpt) -> Self {
    let enable_8_bit = opt.enable_8_bit();
    ChunkConfig::default()
      .with_compression_level(opt.level)
      .with_delta_spec(opt.delta)
      .with_mode_spec(opt.mode)
      .with_paging_spec(PagingSpec::EqualPagesUpTo(opt.chunk_n))
      .with_enable_8_bit(enable_8_bit)
  }
}
