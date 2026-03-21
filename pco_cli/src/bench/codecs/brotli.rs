use crate::bench::codecs::{utils, CodecInternal};
use crate::dtypes::PcoNumber;
use brotli::enc::BrotliEncoderParams;
use brotli::BrotliCompress;
use clap::Parser;
use std::convert::TryInto;
use std::default::Default;
use std::{mem, slice};

#[derive(Clone, Debug, Parser)]
pub struct BrotliConfig {
  #[arg(long, default_value = "1")]
  quality: i32,
}

impl CodecInternal for BrotliConfig {
  fn name(&self) -> &'static str {
    "brotli"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![("quality", self.quality.to_string())]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let params = BrotliEncoderParams {
      quality: self.quality,
      ..Default::default()
    };
    let mut res = Vec::new();
    res.extend((nums.len() as u32).to_le_bytes());
    unsafe {
      BrotliCompress(
        &mut utils::num_slice_to_bytes(nums),
        &mut res,
        &params,
      )
      .unwrap();
    }
    res
  }

  fn decompress<T: PcoNumber>(&self, bytes: &[u8]) -> Vec<T> {
    let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut res = Vec::<T>::with_capacity(len);
    unsafe {
      let byte_len = len * mem::size_of::<T>();
      let mut bytes_out = slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u8, byte_len);
      brotli::BrotliDecompress(&mut &bytes[4..], &mut bytes_out).unwrap();
      res.set_len(len);
    }
    res
  }
}
