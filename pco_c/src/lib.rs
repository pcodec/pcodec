#![allow(clippy::missing_safety_doc)]

use std::ptr;

use libc::{c_uchar, c_uint, c_void, size_t};

use crate::PcoError::PcoInvalidType;
use pco::data_types::{Number, NumberType};
use pco::{match_number_enum, ChunkConfig};

#[repr(C)]
pub enum PcoError {
  PcoSuccess,
  PcoInvalidType,
  // TODO split this into the actual error kinds
  PcoCompressionError,
  PcoDecompressionError,
}

pco::define_number_enum!(
  #[derive()]
  NumVec(Vec)
);

#[repr(C)]
pub struct PcoFfiVec {
  ptr: *const c_void,
  len: size_t,
  raw_box: *const c_void,
}

impl PcoFfiVec {
  fn init_from_bytes(&mut self, v: Vec<u8>) {
    self.ptr = v.as_ptr() as *const c_void;
    self.len = v.len();
    self.raw_box = Box::into_raw(v.into_boxed_slice()) as *const c_void;
  }

  fn init_from_nums<T: Number>(&mut self, v: Vec<T>) {
    self.ptr = v.as_ptr() as *const c_void;
    self.len = v.len();
    self.raw_box = Box::into_raw(v.into_boxed_slice()) as *const c_void;
  }

  fn free(&mut self) {
    unsafe {
      drop(Box::from_raw(self.raw_box as *mut NumVec));
    }
    self.ptr = ptr::null();
    self.len = 0;
    self.raw_box = ptr::null();
  }
}

fn _simpler_compress<T: Number>(
  nums: *const c_void,
  len: size_t,
  level: c_uint,
  ffi_vec_ptr: *mut PcoFfiVec,
) -> PcoError {
  let slice = unsafe { std::slice::from_raw_parts(nums as *const T, len) };
  match pco::standalone::simple_compress(
    slice,
    &ChunkConfig::default().with_compression_level(level as usize),
  ) {
    Err(_) => PcoError::PcoCompressionError,
    Ok(v) => {
      unsafe { (*ffi_vec_ptr).init_from_bytes(v) };
      PcoError::PcoSuccess
    }
  }
}

fn _simple_decompress<T: Number>(
  compressed: *const c_void,
  len: size_t,
  ffi_vec_ptr: *mut PcoFfiVec,
) -> PcoError {
  let slice = unsafe { std::slice::from_raw_parts(compressed as *const u8, len) };
  match pco::standalone::simple_decompress::<T>(slice) {
    Err(_) => PcoError::PcoDecompressionError,
    Ok(v) => {
      unsafe { (*ffi_vec_ptr).init_from_nums(v) };
      PcoError::PcoSuccess
    }
  }
}

#[no_mangle]
pub extern "C" fn pco_simpler_compress(
  nums: *const c_void,
  len: size_t,
  dtype: c_uchar,
  level: c_uint,
  dst: *mut PcoFfiVec,
) -> PcoError {
  let Some(dtype) = NumberType::from_descriminant(dtype) else {
    return PcoInvalidType;
  };

  match_number_enum!(
    dtype,
    NumberType<T> => {
      _simpler_compress::<T>(nums, len, level, dst)
    }
  )
}

#[no_mangle]
pub extern "C" fn pco_simple_decompress(
  compressed: *const c_void,
  len: size_t,
  dtype: c_uchar,
  dst: *mut PcoFfiVec,
) -> PcoError {
  let Some(dtype) = NumberType::from_descriminant(dtype) else {
    return PcoInvalidType;
  };

  match_number_enum!(
    dtype,
    NumberType<T> => {
      _simple_decompress::<T>(compressed, len, dst)
    }
  )
}

#[no_mangle]
pub unsafe extern "C" fn pco_free_pcovec(ffi_vec: *mut PcoFfiVec) -> PcoError {
  unsafe { (*ffi_vec).free() };
  PcoError::PcoSuccess
}
