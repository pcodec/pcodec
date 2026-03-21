use std::convert::TryInto;
use std::io::{Read, Write};
use std::{mem, slice};

use clap::Parser;

use crate::bench::codecs::{utils, CodecInternal};
use crate::dtypes::PcoNumber;

#[derive(Clone, Debug, Parser)]
pub struct SnappyConfig {}

impl CodecInternal for SnappyConfig {
  fn name(&self) -> &'static str {
    "snappy"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let mut res = Vec::new();
    res.extend((nums.len() as u32).to_le_bytes());

    unsafe {
      let mut wtr = snap::write::FrameEncoder::new(&mut res);
      wtr.write_all(utils::num_slice_to_bytes(nums)).unwrap();
      wtr.flush().unwrap();
    }
    res
  }

  fn decompress<T: PcoNumber>(&self, bytes: &[u8]) -> Vec<T> {
    let len = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let mut res = Vec::<T>::with_capacity(len);
    let mut rdr = snap::read::FrameDecoder::new(&bytes[4..]);
    unsafe {
      let byte_len = len * mem::size_of::<T>();
      let bytes_out = slice::from_raw_parts_mut(res.as_mut_ptr() as *mut u8, byte_len);
      rdr.read_exact(bytes_out).unwrap();
      res.set_len(len);
    }
    res
  }
}
