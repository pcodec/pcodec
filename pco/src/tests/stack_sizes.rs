use crate::latent_batch_dissector::LatentBatchDissector;
use crate::latent_chunk_compressor::LatentChunkCompressor;
use crate::latent_page_decompressor::{DynLatentPageDecompressor, LatentPageDecompressor};
use crate::metadata::PerLatentVar;
use crate::wrapped::{ChunkCompressor, ChunkDecompressor, PageDecompressor};
use std::mem;

#[test]
fn test_stack_sizes() {
  // Some of our structs get pretty large on the stack, so it's good to be
  // aware of that. Hopefully we can minimize this in the future.

  // compression
  assert_eq!(
    mem::size_of::<LatentBatchDissector<u64>>(),
    3088
  );
  assert_eq!(
    mem::size_of::<LatentChunkCompressor<u64>>(),
    136
  );
  assert_eq!(mem::size_of::<ChunkDecompressor<u64>>(), 168);
  assert_eq!(mem::size_of::<ChunkCompressor>(), 624);

  // decompression
  assert_eq!(
    mem::size_of::<LatentPageDecompressor<u64>>(),
    4248
  );
  assert_eq!(
    mem::size_of::<DynLatentPageDecompressor>(),
    16
  );
  assert_eq!(
    mem::size_of::<PerLatentVar<DynLatentPageDecompressor>>(),
    48
  );
  assert_eq!(
    mem::size_of::<PageDecompressor<u64, &[u8]>>(),
    256
  );
}
