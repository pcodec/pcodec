use std::cmp::{max, min};
use std::convert::TryInto;

use clap::Parser;

use crate::bench::codecs::{utils, CodecInternal};
use crate::dtypes::PcoNumber;

#[derive(Clone, Debug, Parser)]
pub struct SpdpConfig {
  #[arg(long, default_value = "5")]
  level: u8,
}

impl CodecInternal for SpdpConfig {
  fn name(&self) -> &'static str {
    "spdp"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![("level", self.level.to_string())]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let mut dst = Vec::new();
    dst.push(self.level);
    dst.extend((nums.len() as u32).to_le_bytes());
    let mut src = unsafe { utils::num_slice_to_bytes(nums) };
    while !src.is_empty() {
      // SPDP modifies the input buffer, so we copy the batch
      let src_batch_length = min(src.len(), 1 << 23);
      let mut src_batch = src[..src_batch_length].to_vec();

      // write uncompressed size and a placeholder for compressed size,
      // to be filled in later
      dst.extend((src_batch_length as u32).to_le_bytes());
      dst.extend(&[0; 4]);

      let dst_buffer_size = 2 * src_batch_length + 9;
      dst.reserve(dst_buffer_size);

      let pos = dst.len();
      unsafe {
        let csize = spdp_sys::spdp_compress_batch(
          self.level,
          src_batch_length,
          src_batch.as_mut_ptr(),
          dst[pos..].as_mut_ptr(),
        );
        dst[pos - 4..pos].copy_from_slice(&(csize as u32).to_le_bytes());
        dst.set_len(pos + csize);
      };
      src = &src[src_batch_length..];
    }
    dst
  }

  fn decompress<T: PcoNumber>(&self, mut src: &[u8]) -> Vec<T> {
    let level = src[0];
    let total_count = u32::from_le_bytes(src[1..5].try_into().unwrap()) as usize;
    src = &src[5..];
    // Empirically, we need to add some extra buffer room at the end of the
    // decompressed buffer to avoid segfaults. I haven't looked at their source
    // code enough to know exactly how much is necessary.
    let mut dst = Vec::with_capacity(2 * total_count + 9);
    let mut dst_bytes = unsafe {
      dst.set_len(total_count);
      utils::num_slice_to_bytes_mut(&mut dst)
    };

    while !src.is_empty() {
      let dst_batch_length = u32::from_le_bytes(src[..4].try_into().unwrap()) as usize;
      let csize = u32::from_le_bytes(src[4..8].try_into().unwrap()) as usize;
      src = &src[8..];
      // SPDP modifies the input buffer and requires the input data be at least
      // as long as the uncompressed output data, so we copy the batch.
      let mut src_batch = vec![0; max(csize, dst_batch_length)];
      src_batch[..csize].copy_from_slice(&src[..csize]);
      unsafe {
        spdp_sys::spdp_decompress_batch(
          level,
          csize,
          src_batch.as_mut_ptr(),
          dst_bytes.as_mut_ptr(),
        )
      };
      src = &src[csize..];
      dst_bytes = &mut dst_bytes[dst_batch_length..];
    }
    dst
  }
}
