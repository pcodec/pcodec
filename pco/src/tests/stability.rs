use crate::chunk_config::{ChunkConfig, DeltaSpec};
use crate::data_types::Number;
use crate::errors::{ErrorKind, PcoResult};
use crate::metadata::chunk::ChunkMeta;
use crate::standalone::{simple_decompress, FileCompressor};
use crate::ModeSpec;

fn assert_panic_safe<T: Number>(nums: Vec<T>) -> PcoResult<ChunkMeta> {
  let fc = FileCompressor::default();
  let config = ChunkConfig {
    mode_spec: ModeSpec::Classic,
    delta_spec: DeltaSpec::NoOp,
    ..Default::default()
  };
  let mut cc = fc.chunk_compressor(&nums, &config)?;
  let meta = cc.meta().clone();
  let mut compressed = Vec::new();
  fc.write_header(&mut compressed)?;
  cc.write(&mut compressed)?;
  fc.write_footer(&mut compressed)?;

  for i in 0..compressed.len() - 1 {
    match simple_decompress::<T>(&compressed[0..i]) {
      Err(e) if matches!(e.kind, ErrorKind::InsufficientData) => (), // good
      Ok(_) => panic!("expected decompressor to notice insufficient data (got Ok)"),
      Err(e) => panic!(
        "expected decompressor to notice insufficient data (got {})",
        e
      ),
    }
  }

  Ok(meta)
}

#[test]
fn test_insufficient_data_short_bins() -> PcoResult<()> {
  let mut nums = Vec::new();
  for _ in 0..50 {
    nums.push(0);
  }
  for _ in 0..50 {
    nums.push(1000);
  }

  let meta = assert_panic_safe(nums)?;
  assert!(meta.per_latent_var.delta.is_none());
  assert!(meta.per_latent_var.secondary.is_none());
  assert_eq!(
    meta
      .per_latent_var
      .primary
      .bins
      .downcast_ref::<u32>()
      .unwrap()
      .len(),
    2
  );
  Ok(())
}

#[test]
fn test_insufficient_data_sparse() -> PcoResult<()> {
  let mut nums = vec![0];
  for _ in 0..(1 << 16) + 1 {
    nums.push(1);
  }

  let meta = assert_panic_safe(nums)?;
  assert!(meta.per_latent_var.delta.is_none());
  assert!(meta.per_latent_var.secondary.is_none());
  assert_eq!(
    meta
      .per_latent_var
      .primary
      .bins
      .downcast_ref::<u32>()
      .unwrap()
      .len(),
    2
  );
  Ok(())
}

#[test]
fn test_insufficient_data_long_offsets() -> PcoResult<()> {
  let n = 1000;
  let mut nums = Vec::new();
  for i in 0..n {
    nums.push((u64::MAX / n) * i);
  }

  let meta = assert_panic_safe(nums)?;
  let bins = meta
    .per_latent_var
    .primary
    .bins
    .downcast_ref::<u64>()
    .unwrap();
  assert!(meta.per_latent_var.delta.is_none());
  assert!(meta.per_latent_var.secondary.is_none());
  assert_eq!(bins.len(), 1);
  assert_eq!(bins[0].offset_bits, 64);
  Ok(())
}
