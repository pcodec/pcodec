use std::io::Write;

use crate::base_decompressor::DecompressorConfig;
use crate::data_types::NumberLike;
use crate::errors::{ErrorKind, PcoResult};
use crate::standalone::{Compressor, DecompressedItem, Decompressor};
use crate::CompressorConfig;

#[test]
fn test_low_level_short() -> PcoResult<()> {
  let nums = vec![vec![0], vec![10, 11], vec![20, 21, 22]];
  assert_lowest_level_behavior(nums)
}

#[test]
fn test_low_level_long() -> PcoResult<()> {
  let nums = vec![(0..100).collect::<Vec<_>>()];
  assert_lowest_level_behavior(nums)
}

#[test]
fn test_low_level_sparse() -> PcoResult<()> {
  let mut nums = vec![0; 1000];
  nums.push(1);
  nums.resize(2000, 0);
  assert_lowest_level_behavior(vec![nums])
}

fn assert_lowest_level_behavior<T: NumberLike>(chunks: Vec<Vec<T>>) -> PcoResult<()> {
  for delta_encoding_order in [0, 7] {
    let debug_info = format!("delta order={}", delta_encoding_order);
    let mut compressor = Compressor::<T>::from_config(CompressorConfig {
      delta_encoding_order,
      ..Default::default()
    });
    compressor.header()?;
    let mut metadatas = Vec::new();
    for nums in &chunks {
      metadatas.push(compressor.chunk(nums)?);
    }
    compressor.footer()?;

    let bytes = compressor.drain_bytes();

    let mut decompressor = Decompressor::<T>::from_config(DecompressorConfig {
      numbers_limit_per_item: 2,
      ..Default::default()
    });
    decompressor.write_all(&bytes).unwrap();
    let flags = decompressor.header()?;
    assert_eq!(&flags, compressor.flags(), "{}", debug_info);
    let mut chunk_idx = 0;
    let mut chunk_nums = Vec::<T>::new();
    let mut terminated = false;
    for maybe_item in &mut decompressor {
      let item = maybe_item?;
      match item {
        DecompressedItem::Flags(_) => panic!("already read flags"),
        DecompressedItem::ChunkMetadata(meta) => {
          assert!(!terminated);
          assert_eq!(&meta, &metadatas[chunk_idx]);
          if chunk_idx > 0 {
            assert_eq!(&chunk_nums, &chunks[chunk_idx - 1]);
            chunk_nums = Vec::new();
          }
          chunk_idx += 1;
        }
        DecompressedItem::Numbers(nums) => {
          assert!(!terminated);
          chunk_nums.extend(&nums);
        }
        DecompressedItem::Footer => {
          assert!(!terminated);
          terminated = true;
        }
      }
    }
    assert_eq!(
      &chunk_nums,
      chunks.last().unwrap(),
      "{}",
      debug_info
    );

    let terminated_err = decompressor.chunk_metadata().unwrap_err();
    assert!(
      matches!(
        terminated_err.kind,
        ErrorKind::InvalidArgument
      ),
      "{}",
      debug_info
    );
    assert!(terminated, "{}", debug_info);
  }
  Ok(())
}
