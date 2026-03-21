use std::convert::TryInto;
use std::{mem, slice};

use clap::Parser;

use crate::bench::codecs::{utils, CodecInternal};
use crate::dtypes::PcoNumber;

#[derive(Clone, Debug, Parser)]
pub struct ZstdConfig {
  #[arg(long, default_value = "3")]
  level: i32,
}

impl CodecInternal for ZstdConfig {
  fn name(&self) -> &'static str {
    "zstd"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![("level", self.level.to_string())]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let mut res = Vec::new();
    res.extend((nums.len() as u32).to_le_bytes());
    unsafe {
      zstd::stream::copy_encode(
        utils::num_slice_to_bytes(nums),
        &mut res,
        self.level,
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
      let bytes_out = slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u8, byte_len);
      zstd::stream::copy_decode(&bytes[4..], bytes_out).unwrap();
      res.set_len(len);
    }
    res
  }
}
