use std::ffi::{c_void, CString};
use std::mem;

use crate::bench::codecs::CodecInternal;
use crate::dtypes::PcoNumber;
use clap::Parser;

// gotta divide each compress/decompress call into chunks of byte size less than
// this or else blosc can't handle them
const CHUNK_SIZE: usize = 1 << 30;

#[derive(Clone, Debug, Parser)]
pub struct BloscConfig {
  #[arg(long, default_value = "1048576")]
  block_size: i32,
  #[arg(long, default_value = "blosclz")]
  cname: String,
  #[arg(long, default_value = "9")]
  clevel: u8,
}

impl BloscConfig {
  unsafe fn create_ctx(&self, typesize: i32) -> *mut blosc2_src::blosc2_context {
    let compressor_name = CString::new(self.cname.to_string()).unwrap();
    let compcode = blosc2_src::blosc2_compname_to_compcode(compressor_name.as_ptr());
    let mut filters = [0; 6]; // no filters
    filters[0] = 1; // byte shuffle
    let cparams = blosc2_src::blosc2_cparams {
      compcode: compcode as u8,
      compcode_meta: 0,
      clevel: self.clevel,
      use_dict: 0,
      typesize,
      nthreads: 1,
      blocksize: self.block_size,
      splitmode: 0,
      schunk: std::ptr::null_mut(),
      filters,
      filters_meta: [0; 6],
      prefilter: None,
      preparams: std::ptr::null_mut(),
      tuner_params: std::ptr::null_mut(),
      tuner_id: 0,
      instr_codec: false,
      codec_params: std::ptr::null_mut(),
      filter_params: [std::ptr::null_mut(); 6],
    };
    blosc2_src::blosc2_create_cctx(cparams)
  }

  unsafe fn create_dctx(&self) -> *mut blosc2_src::blosc2_context {
    let dparams = blosc2_src::blosc2_dparams {
      nthreads: 1,
      schunk: std::ptr::null_mut(),
      postfilter: None,
      postparams: std::ptr::null_mut(),
    };
    blosc2_src::blosc2_create_dctx(dparams)
  }
}

impl CodecInternal for BloscConfig {
  fn name(&self) -> &'static str {
    "blosc"
  }

  fn get_confs(&self) -> Vec<(&'static str, String)> {
    vec![
      ("block-size", self.block_size.to_string()),
      ("cname", self.cname.to_string()),
      ("level", self.clevel.to_string()),
    ]
  }

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8> {
    let n_bytes = mem::size_of_val(nums);
    let n_chunks = n_bytes.div_ceil(CHUNK_SIZE);
    let type_size = mem::size_of::<T>();
    let mut dst =
      Vec::with_capacity(4 + n_bytes + n_chunks * blosc2_src::BLOSC2_MAX_OVERHEAD as usize);
    dst.extend((n_chunks as u32).to_le_bytes());
    let ctx = unsafe { self.create_ctx(type_size as i32) };
    for chunk in nums.chunks(CHUNK_SIZE / type_size) {
      let chunk_n_bytes = mem::size_of_val(chunk);
      let dst_available = (dst.capacity() - dst.len()).min(i32::MAX as usize) as i32;
      unsafe {
        let src = chunk.as_ptr() as *const c_void;
        let compressed_size = blosc2_src::blosc2_compress_ctx(
          ctx,
          src,
          chunk_n_bytes as i32,
          dst.as_mut_ptr().add(dst.len()) as *mut c_void,
          dst_available,
        );
        dst.set_len(dst.len() + compressed_size as usize);
      }
    }

    unsafe {
      blosc2_src::blosc2_free_ctx(ctx);
    }
    dst
  }

  fn decompress<T: PcoNumber>(&self, mut compressed: &[u8]) -> Vec<T> {
    let n_chunks = u32::from_le_bytes(compressed[0..4].try_into().unwrap()) as usize;
    compressed = &compressed[4..];
    let mut res = Vec::<T>::new();

    let ctx = unsafe { self.create_dctx() };
    for _ in 0..n_chunks {
      let mut uncompressed_size = 0;
      let mut compressed_size = 0;
      let mut block_size = 0;
      unsafe {
        let src = compressed.as_ptr() as *const c_void;
        blosc2_src::blosc2_cbuffer_sizes(
          src,
          &mut uncompressed_size as *mut i32,
          &mut compressed_size as *mut i32,
          &mut block_size as *mut i32,
        );
        let chunk_n = uncompressed_size as usize / mem::size_of::<T>();
        res.reserve(chunk_n);
        let dst = res.as_mut_ptr().add(res.len()) as *mut c_void;
        blosc2_src::blosc2_decompress_ctx(
          ctx,
          src,
          compressed.len().min(i32::MAX as usize) as i32,
          dst,
          uncompressed_size,
        );
        res.set_len(res.len() + chunk_n);
        compressed = &compressed[compressed_size as usize..];
      }
    }

    unsafe {
      blosc2_src::blosc2_free_ctx(ctx);
    }
    res
  }
}
