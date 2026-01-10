use crate::chunk_latent_compressor::ChunkLatentCompressor;
use crate::metadata::PerLatentVar;
use crate::page_latent_decompressor::{DynPageLatentDecompressor, PageLatentDecompressor};
use crate::wrapped::{ChunkCompressor, ChunkDecompressor, PageDecompressor};
use std::mem;

#[test]
fn test_stack_sizes() {
  // Some of our structs get pretty large on the stack, so it's good to be
  // aware of that. Hopefully we can minimize this in the future.

  assert_eq!(
    mem::size_of::<ChunkLatentCompressor<u64>>(),
    144
  );
  assert_eq!(mem::size_of::<ChunkDecompressor<u64>>(), 168);
  assert_eq!(mem::size_of::<ChunkCompressor>(), 624);

  // decompression
  assert_eq!(
    mem::size_of::<PageLatentDecompressor<u64>>(),
    4288
  );
  assert_eq!(
    mem::size_of::<DynPageLatentDecompressor>(),
    16
  );
  assert_eq!(
    mem::size_of::<PerLatentVar<DynPageLatentDecompressor>>(),
    48
  );
  assert_eq!(
    mem::size_of::<PageDecompressor<u64, &[u8]>>(),
    240
  );
}
