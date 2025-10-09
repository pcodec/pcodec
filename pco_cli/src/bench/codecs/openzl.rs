use crate::bench::codecs::{utils, CodecInternal};
use crate::dtypes::PcoNumber;
use clap::Parser;

#[derive(Clone, Debug, Parser)]
pub struct OpenZlConfig {}

impl CodecInternal for OpenZlConfig {
  fn name(&self) -> &'static str {
    "openzl"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    rust_openzl::compress_numeric(nums).unwrap()
  }

  fn decompress<T: PcoNumber>(&self, bytes: &[u8]) -> Vec<T> {
    rust_openzl::decompress_numeric(bytes).unwrap()
  }
}
