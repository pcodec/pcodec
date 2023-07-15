use std::io::Write;

use crate::errors::ErrorKind;
use crate::standalone::{simple_compress, Decompressor};
use crate::CompressorConfig;

#[test]
fn test_errors_do_not_mutate_decompressor() {
  let nums = vec![1, 2, 3, 4, 5];
  let compressed = simple_compress(&nums, CompressorConfig::default()).unwrap();
  let mut decompressor = Decompressor::<i32>::default();

  // header shouldn't leave us in a dirty state
  let mut i = 0;
  while i < compressed.len() + 1 {
    match decompressor.header() {
      Ok(_) => break,
      Err(e) if matches!(e.kind, ErrorKind::InsufficientData) => (),
      Err(e) => panic!("{}", e),
    };
    decompressor.write_all(&compressed[i..i + 1]).unwrap();
    i += 1;
  }

  // chunk metadata shouldn't leave us in a dirty state
  while i < compressed.len() + 1 {
    match decompressor.chunk_metadata() {
      Ok(_) => break,
      Err(e) if matches!(e.kind, ErrorKind::InsufficientData) => (),
      Err(e) => panic!("{}", e),
    };
    decompressor.write_all(&compressed[i..i + 1]).unwrap();
    i += 1;
  }

  // reading the chunk shouldn't leave us in a dirty state
  let mut rec_nums = vec![0; nums.len()];
  while i < compressed.len() + 1 {
    match decompressor.chunk_body(&mut rec_nums) {
      Ok(_) => {
        break;
      }
      Err(e) if matches!(e.kind, ErrorKind::InsufficientData) => (),
      Err(e) => panic!("{}", e),
    };
    decompressor.write_all(&compressed[i..i + 1]).unwrap();
    i += 1;
  }

  assert_eq!(rec_nums, nums);
}
