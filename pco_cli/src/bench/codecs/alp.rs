use std::any;
use std::any::{Any, TypeId};
use std::ffi::{c_void, CString};
use std::mem;

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;
use alp::{ALPFloat, ALPRDFloat, RDEncoder};
use clap::Parser;

#[derive(Clone, Debug, Parser)]
pub struct AlpConfig {}

fn as_any<T: PcoNumber>(nums: &[T]) -> &dyn Any {
  unimplemented!()
  //nums
}

fn encode<R: ALPRDFloat>(values: &[R]) -> Vec<u8> {
  unimplemented!()
  //let sample = &values[..1000];
  //let encoder = RDEncoder::new(sample);
  //let split = encoder.split(values);
}

impl CodecInternal for AlpConfig {
  fn name(&self) -> &'static str {
    "alp"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let type_id = nums.type_id();
    if type_id == TypeId::of::<&[f32]>() {
      let nums = as_any(nums).downcast_ref::<&[f32]>().unwrap();
      encode(nums)
    } else if type_id == TypeId::of::<&[f64]>() {
      let nums = as_any(nums).downcast_ref::<&[f64]>().unwrap();
      encode(nums)
    } else {
      panic!("ALP doesn't support type {:?}", type_id)
    }
  }

  fn decompress<T: PcoNumber>(&self, compressed: &[u8]) -> Vec<T> {
    unimplemented!()
  }
}
