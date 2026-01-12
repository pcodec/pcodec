use crate::chunk_latent_compressor::{ChunkLatentCompressor, DynChunkLatentCompressor};
use crate::chunk_latent_decompressor::ChunkLatentDecompressor;
use crate::page_latent_decompressor::{DynPageLatentDecompressor, PageLatentDecompressor};
use crate::wrapped::{ChunkCompressor, ChunkDecompressor, PageDecompressor};

#[test]
fn test_stack_sizes() {
  // Some of our structs get pretty large on the stack, so it's good to be
  // aware of that. Hopefully we can minimize this in the future.

  assert_eq!(size_of::<ChunkLatentCompressor<u64>>(), 3264);
  assert_eq!(size_of::<DynChunkLatentCompressor>(), 16);
  assert_eq!(size_of::<ChunkCompressor>(), 264);

  // decompression
  assert_eq!(size_of::<PageLatentDecompressor<u64>>(), 48);
  assert_eq!(size_of::<DynPageLatentDecompressor>(), 56);
  assert_eq!(
    size_of::<PageDecompressor<u64, &[u8]>>(),
    240
  );
  assert_eq!(
    size_of::<ChunkLatentDecompressor<u64>>(),
    4224
  );
  assert_eq!(size_of::<DynChunkLatentCompressor>(), 16);
  assert_eq!(size_of::<ChunkDecompressor<u64>>(), 240);
}
