use once_cell::sync::Lazy;
use vortex::compressor::CompactCompressor;

use crate::bench::codecs::{utils, CodecInternal};
use crate::dtypes::PcoNumber;
use clap::{Parser, ValueEnum};
use pco::data_types::NumberType;
use pco::match_number_enum;
use tokio::runtime::Runtime;
use vortex::arrays::PrimitiveArray;
use vortex::buffer::ByteBuffer;
use vortex::dtype::DType;
use vortex::file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};
use vortex::scalar::ScalarType;
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
    // can't figure out a way to avoid copying here
    let byte_buffer = ByteBuffer::copy_from(unsafe { utils::num_slice_to_bytes(nums) });
    let number_type = NumberType::from_descriminant(T::NUMBER_TYPE_BYTE).unwrap();
    let dtype = match_number_enum!(number_type,
      NumberType<T> => {
        T::dtype()
      }
    );
    let ptype = match dtype {
      DType::Primitive(ptype, _) => ptype,
      _ => unreachable!(),
    };
    let vortex_arr = PrimitiveArray::from_byte_buffer(byte_buffer, ptype, Validity::NonNullable);

    let mut strategy = WriteStrategyBuilder::default();
    match self.strategy {
      Strategy::Btrblocks => (),
      Strategy::Compact => strategy = strategy.with_compressor(CompactCompressor::default()),
    };
    let options = VortexWriteOptions::default().with_strategy(strategy.build());

    // unfortunately vortex only has an async API
    // By default, writing an array will decompress it back to its canonical
    // form and then recompress it, so there's no need to compress it ahead of
    // time.
    let res = RUNTIME
      .block_on(options.write(Vec::new(), vortex_arr.to_array_stream()))
      .expect("vortex failed to write");
    println!(
      "compressed to {:?} with {:?}",
      res.len(),
      self.strategy
    );
    res
  }

  fn decompress<T: PcoNumber>(&self, src: &[u8]) -> Vec<T> {
    // again, we must copy
    let file = VortexOpenOptions::in_memory()
      .open(ByteBuffer::from(src.to_vec()))
      .unwrap();
    let mut res = vec![];
    for array_result in file.scan().unwrap().into_array_iter().unwrap() {
      let array = array_result.unwrap().to_primitive();
      res.extend(array.buffer::<T>());
    }
    res
  }
}
