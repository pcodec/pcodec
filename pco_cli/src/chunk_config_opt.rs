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
}

impl ChunkConfigOpt {
  pub fn into_chunk_config(self, enable_8_bit: bool) -> ChunkConfig {
    ChunkConfig::default()
      .with_compression_level(self.level)
      .with_delta_spec(self.delta)
      .with_mode_spec(self.mode)
      .with_paging_spec(PagingSpec::EqualPagesUpTo(self.chunk_n))
      .with_enable_8_bit(enable_8_bit)
  }
}
