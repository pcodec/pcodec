use std::mem;

use clap::Parser;

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;

#[derive(Clone, Debug, Parser)]
pub struct SprintzConfig {
  #[arg(long, default_value = "false")]
  xff: bool,
}

impl CodecInternal for SprintzConfig {
  fn name(&self) -> &'static str {
    "sprintz"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![("xff", self.xff.to_string())]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let len = nums.len();
    let mut res = Vec::new();
    res.extend((len as u32).to_le_bytes());

    let buf_size = sprintz_sys::compress_buf_size(len * mem::size_of::<T>());
    res.reserve(buf_size);
    let pos = res.len();
    let clen = unsafe {
      let dest = res.as_mut_ptr().add(pos);
      match mem::size_of::<T>() {
        1 => {
          let src = nums.as_ptr() as *const u8;
          if self.xff {
            sprintz_sys::sprintz_c_compress_xff_8b(src, len as u32, dest as *mut i8, 1)
          } else {
            sprintz_sys::sprintz_c_compress_delta_8b(src, len as u32, dest as *mut i8, 1)
          }
        }
        2 => {
          let src = nums.as_ptr() as *const u16;
          if self.xff {
            sprintz_sys::sprintz_c_compress_xff_16b(src, len as u32, dest as *mut i16, 1)
          } else {
            sprintz_sys::sprintz_c_compress_delta_16b(src, len as u32, dest as *mut i16, 1)
          }
        }
        _ => panic!("sprintz only supports 1- and 2-byte types"),
      }
    };
    assert!(clen >= 0, "sprintz compress failed with code {}", clen);
    unsafe { res.set_len(pos + clen as usize) };
    res
  }

  fn decompress<T: PcoNumber>(&self, compressed: &[u8]) -> Vec<T> {
    let len = u32::from_le_bytes(compressed[..4].try_into().unwrap()) as usize;
    let compressed = &compressed[4..];

    let mut out = Vec::<T>::with_capacity(len + 64);
    let dlen = unsafe {
      let dest = out.as_mut_ptr();
      match mem::size_of::<T>() {
        1 => {
          let src = compressed.as_ptr() as *const i8;
          if self.xff {
            sprintz_sys::sprintz_c_decompress_xff_8b(src, dest as *mut u8)
          } else {
            sprintz_sys::sprintz_c_decompress_delta_8b(src, dest as *mut u8)
          }
        }
        2 => {
          let src = compressed.as_ptr() as *const i16;
          if self.xff {
            sprintz_sys::sprintz_c_decompress_xff_16b(src, dest as *mut u16)
          } else {
            sprintz_sys::sprintz_c_decompress_delta_16b(src, dest as *mut u16)
          }
        }
        _ => panic!("sprintz only supports 1- and 2-byte types"),
      }
    };
    assert!(dlen >= 0, "sprintz decompress failed with code {}", dlen);
    unsafe { out.set_len(len) };
    out
  }
}
