use once_cell::sync::Lazy;
use vortex::compressor::CompactCompressor;
use vortex::io::runtime::current::CurrentThreadRuntime;

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;
use clap::{Parser, ValueEnum};
use tokio::runtime::Runtime;
use vortex::arrays::PrimitiveArray;
use vortex::buffer::{Buffer, ByteBuffer};
use vortex::file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};
use vortex::io::runtime::tokio::TokioRuntime;
use vortex::stream::ArrayStreamExt;
use vortex::validity::Validity;
use vortex::ToCanonical;

static RUNTIME: Lazy<Runtime> =
  Lazy::new(|| Runtime::new().expect("Failed to create Tokio runtime"));

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum Strategy {
  Btrblocks,
  Compact,
}

#[derive(Clone, Debug, Parser)]
pub struct VortexConfig {
  #[arg(long, default_value = "btrblocks")]
  strategy: Strategy,
}

impl CodecInternal for VortexConfig {
  fn name(&self) -> &'static str {
    "vortex"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![(
      "strategy",
      format!("{:?}", self.strategy).to_lowercase(),
    )]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let vx_arr = PrimitiveArray::new(
      // can't figure out a way to avoid copying the numbers here
      Buffer::copy_from(nums),
      Validity::NonNullable,
    );

    let mut strategy = WriteStrategyBuilder::default();
    match self.strategy {
      Strategy::Btrblocks => (),
      Strategy::Compact => strategy = strategy.with_compressor(CompactCompressor::default()),
    };
    let options = VortexWriteOptions::default()
      .with_strategy(strategy.build())
      .blocking::<CurrentThreadRuntime>();

    // unfortunately vortex only has an async API
    // By default, writing an array will decompress it back to its canonical
    // form and then recompress it, so there's no need to compress it ahead of
    // time.
    let mut res = Vec::new();
    options
      .write(&mut res, vx_arr.to_array_iterator())
      .expect("vortex failed to write");
    res
  }

  fn decompress<T: PcoNumber>(&self, src: &[u8]) -> Vec<T> {
    let handle = TokioRuntime::current();
    let file = RUNTIME
      .block_on(
        VortexOpenOptions::new()
          .with_handle(handle.clone())
          // again, we must copy the compressed data
          .open(ByteBuffer::copy_from(src)),
      )
      .unwrap();
    let vx_arr = RUNTIME
      .block_on(
        file
          .scan()
          .unwrap()
          .with_handle(handle)
          .into_array_stream()
          .unwrap()
          .read_all(),
      )
      .unwrap();
    // aaand one last copy for the numbers
    vx_arr.to_primitive().buffer::<T>().to_vec()
  }
}
