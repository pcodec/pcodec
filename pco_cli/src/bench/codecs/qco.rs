use clap::Parser;

use q_compress::CompressorConfig;

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;

#[derive(Clone, Debug, Parser)]
pub struct QcoConfig {
  /// Compression level.
  #[arg(long, default_value = "8")]
  level: usize,
  /// If specified, uses a fixed delta encoding order. Defaults to automatic detection.
  #[arg(long = "delta-order")]
  delta_encoding_order: Option<usize>,
  #[arg(long)]
  use_gcds: bool,
}

impl CodecInternal for QcoConfig {
  fn name(&self) -> &'static str {
    "qco"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![
      ("level", self.level.to_string()),
      (
        "delta-order",
        self
          .delta_encoding_order
          .map(|order| order.to_string())
          .unwrap_or("auto".to_string()),
      ),
      ("use-gcds", self.use_gcds.to_string()),
    ]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let qco_nums = T::nums_to_qco(nums);
    let delta_order = self.delta_encoding_order.unwrap_or_else(|| {
      q_compress::auto_compressor_config(&qco_nums, self.level).delta_encoding_order
    });
    let c_config = CompressorConfig::default()
      .with_compression_level(self.level)
      .with_use_gcds(self.use_gcds)
      .with_delta_encoding_order(delta_order);
    q_compress::standalone::Compressor::<T::Qco>::from_config(c_config).simple_compress(&qco_nums)
  }

  fn decompress<T: PcoNumber>(&self, bytes: &[u8]) -> Vec<T> {
    let qco_nums = q_compress::auto_decompress::<T::Qco>(bytes).expect("could not decompress");
    T::qco_to_nums(qco_nums)
  }
}
