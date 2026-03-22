use clap::Parser;

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;

#[derive(Clone, Debug, clap::ValueEnum)]
enum Filter {
  Delta,
  Xff,
}

#[derive(Clone, Debug, Parser)]
pub struct SprintzConfig {
  #[arg(long, default_value = "delta")]
  filter: Filter,
}

impl CodecInternal for SprintzConfig {
  fn name(&self) -> &'static str {
    "sprintz"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    let filter = match self.filter {
      Filter::Delta => "delta",
      Filter::Xff => "xff",
    };
    vec![("filter", filter.to_string())]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let len = nums.len() as u32;
    let mut res = Vec::new();
    res.extend((len as u32).to_le_bytes());

    let buf_size = sprintz_sys::compress_buf_size(size_of_val(nums));
    res.reserve(buf_size);
    let pos = res.len();
    let clen = unsafe {
      let dst = res.as_mut_ptr().add(pos);
      match (size_of::<T>(), &self.filter) {
        (1, Filter::Delta) => {
          sprintz_sys::sprintz_c_compress_delta_8b(nums.as_ptr() as _, len, dst, 1)
        }
        (1, Filter::Xff) => {
          sprintz_sys::sprintz_c_compress_xff_8b(nums.as_ptr() as _, len, dst as _, 1)
        }
        (2, Filter::Delta) => {
          sprintz_sys::sprintz_c_compress_delta_16b(nums.as_ptr() as _, len, dst as _, 1)
        }
        (2, Filter::Xff) => {
          sprintz_sys::sprintz_c_compress_xff_16b(nums.as_ptr() as _, len, dst as _, 1)
        }
        _ => panic!("sprintz only supports 1- and 2-byte types"),
      }
    };
    assert!(
      clen >= 0,
      "sprintz compress failed with code {}",
      clen
    );
    // If we don't add +2 here, the last decompressed number will be wrong.
    // Seems like a bug on their side.
    unsafe { res.set_len(pos + (clen as usize) * size_of::<T>() + 2) };
    res
  }

  fn decompress<T: PcoNumber>(&self, compressed: &[u8]) -> Vec<T> {
    let len = u32::from_le_bytes(compressed[..4].try_into().unwrap()) as usize;
    let compressed = &compressed[4..];

    let mut out = Vec::<T>::with_capacity(len + 64);
    let dlen = unsafe {
      let dst = out.as_mut_ptr();
      match (size_of::<T>(), &self.filter) {
        (1, Filter::Delta) => {
          sprintz_sys::sprintz_c_decompress_delta_8b(compressed.as_ptr() as _, dst as _)
        }
        (1, Filter::Xff) => {
          sprintz_sys::sprintz_c_decompress_xff_8b(compressed.as_ptr() as _, dst as _)
        }
        (2, Filter::Delta) => {
          sprintz_sys::sprintz_c_decompress_delta_16b(compressed.as_ptr() as _, dst as _)
        }
        (2, Filter::Xff) => {
          sprintz_sys::sprintz_c_decompress_xff_16b(compressed.as_ptr() as _, dst as _)
        }
        _ => panic!("sprintz only supports 1- and 2-byte types"),
      }
    };
    assert!(
      dlen >= 0,
      "sprintz decompress failed with code {}",
      dlen
    );
    unsafe { out.set_len(len) };
    out
  }
}
