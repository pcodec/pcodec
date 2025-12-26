use pco::{ChunkConfig, DeltaSpec, ModeSpec};

use crate::bench::codecs::CodecInternal;
use crate::chunk_config_opt::ChunkConfigOpt;
use crate::dtypes::PcoNumber;

fn unparse_delta_spec(spec: &DeltaSpec) -> String {
  match spec {
    DeltaSpec::Auto => "Auto".to_string(),
    DeltaSpec::NoOp => "NoOp".to_string(),
    DeltaSpec::TryConsecutive(order) => format!("Consecutive@{}", order),
    DeltaSpec::TryLookback => "Lookback".to_string(),
    _ => "Unknown".to_string(),
  }
}

fn unparse_mode_spec(spec: &ModeSpec) -> String {
  match spec {
    ModeSpec::Auto => "Auto".to_string(),
    ModeSpec::Classic => "Classic".to_string(),
    ModeSpec::TryFloatMult(base) => format!("FloatMult@{}", base),
    ModeSpec::TryFloatQuant(k) => format!("FloatQuant@{}", k),
    ModeSpec::TryIntMult(base) => format!("IntMult@{}", base),
    _ => "Unknown".to_string(),
  }
}

impl CodecInternal for ChunkConfigOpt {
  fn name(&self) -> &'static str {
    "pco"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![
      ("level", self.level.to_string()),
      ("delta", unparse_delta_spec(&self.delta)),
      ("mode", unparse_mode_spec(&self.mode)),
      ("chunk-n", self.chunk_n.to_string()),
    ]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let chunk_config = ChunkConfig::from(self);
    pco::standalone::simple_compress(nums, &chunk_config).expect("invalid config")
  }

  fn decompress<T: PcoNumber>(&self, bytes: &[u8]) -> Vec<T> {
    pco::standalone::simple_decompress::<T>(bytes).expect("could not decompress")
  }
}
