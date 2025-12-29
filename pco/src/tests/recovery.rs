use half::f16;
use rand::Rng;
use rand_xoshiro::rand_core::SeedableRng;

use crate::chunk_config::{ChunkConfig, DeltaSpec};
use crate::constants::Bitlen;
use crate::data_types::{LatentType, Number};
use crate::errors::PcoResult;
use crate::metadata::{ChunkMeta, DeltaEncoding, DynLatent, DynLatents, Mode};
use crate::standalone::{simple_compress, simple_decompress, FileCompressor};
use crate::ModeSpec;

fn compress_w_meta<T: Number>(nums: &[T], config: &ChunkConfig) -> PcoResult<(Vec<u8>, ChunkMeta)> {
  let mut compressed = Vec::new();
  let fc = FileCompressor::default();
  fc.write_header(&mut compressed)?;
  let cd = fc.chunk_compressor(nums, config)?;
  let meta = cd.meta().clone();
  cd.write_chunk(&mut compressed)?;
  fc.write_footer(&mut compressed)?;

  Ok((compressed, meta))
}

fn assert_nums_eq<T: Number>(decompressed: &[T], expected: &[T], name: &str) -> PcoResult<()> {
  let debug_info = format!("name={}", name,);
  // We can't do assert_eq on the whole vector because even bitwise identical
  // floats sometimes aren't equal by ==.
  assert_eq!(
    decompressed.len(),
    expected.len(),
    "{}",
    debug_info
  );
  for (i, (x, y)) in decompressed.iter().zip(expected).enumerate() {
    assert_eq!(
      x.to_latent_ordered(),
      y.to_latent_ordered(),
      "at {}; {}",
      i,
      debug_info,
    );
  }
  Ok(())
}

fn assert_recovers<T: Number>(nums: &[T], compression_level: usize, name: &str) -> PcoResult<()> {
  for delta_encoding_order in [0, 1, 7] {
    for mode_spec in [ModeSpec::Classic, ModeSpec::Auto] {
      let config = ChunkConfig {
        compression_level,
        delta_spec: DeltaSpec::TryConsecutive(delta_encoding_order),
        mode_spec,
        ..Default::default()
      };
      let compressed = simple_compress(nums, &config)?;
      let decompressed = simple_decompress(&compressed)?;
      assert_nums_eq(
        &decompressed,
        nums,
        &format!(
          "{} delta order={}",
          name, delta_encoding_order
        ),
      )?;
    }
  }
  Ok(())
}

#[test]
fn test_edge_cases() -> PcoResult<()> {
  assert_recovers(&[u64::MIN, u64::MAX], 0, "u64 extremes - 0")?;
  assert_recovers(&[f64::MIN, f64::MAX], 0, "f64 extremes - 0")?;
  assert_recovers(&[1.2_f32], 0, "f32 - 0")?;
  assert_recovers(&[1.2_f32], 1, "f32 - 1")?;
  assert_recovers(&[1.2_f32], 2, "f32 - 2")?;
  assert_recovers(&Vec::<u32>::new(), 6, "empty u32 - 6")?;
  assert_recovers(&Vec::<u32>::new(), 0, "empty u32 - 0")?;
  assert_recovers(&Vec::<u16>::new(), 6, "empty u16 - 6")?;
  assert_recovers(
    &[
      f16::NEG_INFINITY,
      f16::MIN,
      f16::NEG_ONE,
      f16::NEG_ZERO,
      f16::NAN,
      f16::ZERO,
      f16::ONE,
      f16::MAX,
      f16::INFINITY,
    ],
    5,
    "f16 - 5",
  )?;

  Ok(())
}

#[test]
fn test_moderate_data() -> PcoResult<()> {
  let mut v = Vec::new();
  for i in -50000..50000 {
    v.push(i);
  }
  assert_recovers(&v, 3, "moderate data")
}

#[test]
fn test_sparse() -> PcoResult<()> {
  let mut v = Vec::new();
  for _ in 0..10000 {
    v.push(1);
  }
  v.push(0);
  v.push(0);
  v.push(1);
  assert_recovers(&v, 1, "sparse")
}

#[test]
fn test_u16_codec() -> PcoResult<()> {
  assert_recovers(&[0_u16, u16::MAX, 2, 3, 4, 5], 1, "u16s")
}

#[test]
fn test_u32_codec() -> PcoResult<()> {
  assert_recovers(&[0_u32, u32::MAX, 3, 4, 5], 1, "u32s")
}

#[test]
fn test_u64_codec() -> PcoResult<()> {
  assert_recovers(&[0_u64, u64::MAX, 3, 4, 5], 1, "u64s")
}

#[test]
fn test_i16_codec() -> PcoResult<()> {
  assert_recovers(
    &[0_i16, -1, i16::MAX, i16::MIN, 7],
    1,
    "i16s",
  )
}

#[test]
fn test_i32_codec() -> PcoResult<()> {
  assert_recovers(
    &[0_i32, -1, i32::MAX, i32::MIN, 7],
    1,
    "i32s",
  )
}

#[test]
fn test_i64_codec() -> PcoResult<()> {
  assert_recovers(
    &[0_i64, -1, i64::MAX, i64::MIN, 7],
    1,
    "i64s",
  )
}

#[test]
fn test_f16_codec() -> PcoResult<()> {
  assert_recovers(
    &[
      f16::MAX,
      f16::MIN,
      f16::NAN,
      f16::NEG_INFINITY,
      f16::INFINITY,
      f16::from_f32(-0.0),
      f16::from_f32(0.0),
      f16::from_f32(77.7),
    ],
    1,
    "f16s",
  )
}

#[test]
fn test_f32_codec() -> PcoResult<()> {
  assert_recovers(
    &[
      f32::MAX,
      f32::MIN,
      f32::NAN,
      f32::NEG_INFINITY,
      f32::INFINITY,
      -0.0,
      0.0,
      77.7,
    ],
    1,
    "f32s",
  )
}

#[test]
fn test_f64_codec() -> PcoResult<()> {
  assert_recovers(
    &[
      f64::MAX,
      f64::MIN,
      f64::NAN,
      f64::NEG_INFINITY,
      f64::INFINITY,
      -0.0,
      0.0,
      77.7,
    ],
    1,
    "f64s",
  )
}

#[test]
fn test_multi_chunk() -> PcoResult<()> {
  let config = ChunkConfig::default();
  let fc = FileCompressor::default();
  let mut compressed = Vec::new();
  fc.write_header(&mut compressed)?;
  fc.chunk_compressor(&[1_i64, 2, 3], &config)?
    .write_chunk(&mut compressed)?;
  fc.chunk_compressor(&[11_i64, 12, 13], &config)?
    .write_chunk(&mut compressed)?;
  fc.write_footer(&mut compressed)?;

  let res = simple_decompress::<i64>(&compressed)?;
  assert_nums_eq(&res, &[1, 2, 3, 11, 12, 13], "multi chunk")?;
  Ok(())
}

fn recover_with_alternating_nums(offset_bits: Bitlen, name: &str) -> PcoResult<()> {
  let nums = [0_u64, 1 << (offset_bits - 1)].repeat(50);
  let (compressed, meta) = compress_w_meta(
    &nums,
    &ChunkConfig {
      delta_spec: DeltaSpec::NoOp,
      compression_level: 0,
      ..Default::default()
    },
  )?;
  assert!(meta.per_latent_var.delta.is_none());
  assert!(meta.per_latent_var.secondary.is_none());
  let latent_var = &meta.per_latent_var.primary;
  let bins = latent_var.bins.downcast_ref::<u64>().unwrap();
  assert_eq!(bins.len(), 1);
  assert_eq!(bins[0].offset_bits, offset_bits);
  let decompressed = simple_decompress(&compressed)?;
  assert_nums_eq(&decompressed, &nums, name)
}

#[test]
fn test_56_bit_offsets() -> PcoResult<()> {
  recover_with_alternating_nums(56, "56 bit offsets")
}

#[test]
fn test_57_bit_offsets() -> PcoResult<()> {
  recover_with_alternating_nums(57, "57 bit offsets")
}

#[test]
fn test_64_bit_offsets() -> PcoResult<()> {
  recover_with_alternating_nums(64, "64 bit offsets")
}

#[test]
fn test_with_int_mult() -> PcoResult<()> {
  let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);
  let mut nums = Vec::new();
  for _ in 0..300 {
    nums.push(rng.gen_range(-1000..1000) * 8 - 1);
  }
  let (compressed, meta) = compress_w_meta(
    &nums,
    &ChunkConfig {
      delta_spec: DeltaSpec::NoOp,
      ..Default::default()
    },
  )?;
  assert_eq!(
    meta.mode,
    Mode::IntMult(DynLatent::U32(8_u32))
  );
  let decompressed = simple_decompress(&compressed)?;
  assert_nums_eq(&decompressed, &nums, "sparse w gcd")?;
  Ok(())
}

#[test]
fn test_sparse_islands() -> PcoResult<()> {
  let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);
  let mut nums = Vec::new();
  // sparse - one common island of [0, 8) and one rare of [1000, 1008)
  for _ in 0..20 {
    for _ in 0..99 {
      nums.push(rng.gen_range(0..8))
    }
    nums.push(rng.gen_range(1000..1008))
  }
  assert_recovers(&nums, 4, "sparse islands")
}

#[test]
fn test_decimals() -> PcoResult<()> {
  let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);
  let mut nums = Vec::new();
  let n = 300;

  pub fn plus_epsilons(a: f64, epsilons: i64) -> f64 {
    f64::from_latent_ordered(a.to_latent_ordered().wrapping_add(epsilons as u64))
  }

  for _ in 0..n {
    let unadjusted_num = (rng.gen_range(-1..100) as f64) * 0.01;
    let adj = rng.gen_range(-1..2);
    nums.push(plus_epsilons(unadjusted_num, adj));
  }
  // add some big numbers just to test losslessness
  nums.resize(2 * n, f64::INFINITY);
  // Each regular number should take only 7 bits for offset and 2 bits for
  // adjustment, plus some overhead. Each infinity should take 1 bit plus maybe
  // 2 for adjustment.
  let overhead_bytes = 100;
  let (compressed, meta) = compress_w_meta(&nums, &ChunkConfig::default())?;
  assert!(compressed.len() < (9 * n + 3 * n) / 8 + overhead_bytes);
  assert_eq!(meta.mode, Mode::float_mult(1.0 / 100.0));

  assert_recovers(&nums, 2, "decimals")
}

#[test]
fn test_f16_mult() -> PcoResult<()> {
  let nums = [100.1, 299.9, 200.0].repeat(100);
  let config = ChunkConfig {
    mode_spec: ModeSpec::TryFloatMult(100.0),
    ..Default::default()
  };
  let (_, meta) = compress_w_meta(&nums, &config)?;
  assert_eq!(meta.mode, Mode::float_mult(100.0));

  assert_recovers(&nums, 1, "f16 mult mode")
}

#[test]
fn test_trivial_first_latent_var() -> PcoResult<()> {
  let mut nums = Vec::new();
  for i in 0..100 {
    nums.push(i as f32);
  }
  nums[77] += 0.0001;
  let (compressed, meta) = compress_w_meta(&nums, &ChunkConfig::default())?;
  assert_eq!(meta.mode, Mode::float_mult(1.0_f32));
  let decompressed = simple_decompress(&compressed)?;
  assert_nums_eq(&decompressed, &nums, "trivial_first_latent")?;
  Ok(())
}

#[test]
fn test_lookback_delta_encoding() -> PcoResult<()> {
  let mut nums = Vec::new();
  for i in 0..100 {
    nums.push(i % 9);
  }
  let (compressed, meta) = compress_w_meta(
    &nums,
    &ChunkConfig::default().with_delta_spec(DeltaSpec::TryLookback),
  )?;
  assert!(matches!(
    meta.delta_encoding,
    DeltaEncoding::Lookback { .. }
  ));
  let decompressed = simple_decompress(&compressed)?;
  assert_nums_eq(
    &decompressed,
    &nums,
    "lookback delta encoding",
  )?;
  Ok(())
}

#[test]
fn test_dict() -> PcoResult<()> {
  let mut nums = Vec::<i64>::new();
  for i in 0..2000 {
    for _ in 0..5 {
      nums.push(i * i);
    }
  }
  let (compressed, meta) = compress_w_meta(
    &nums,
    &ChunkConfig::default()
      .with_mode_spec(ModeSpec::TryDict)
      .with_delta_spec(DeltaSpec::NoOp),
  )?;
  let Mode::Dict(DynLatents::U64(dict)) = &meta.mode else {
    panic!("expected to compress with a dictionary of u64s");
  };
  assert!(matches!(
    meta.per_latent_var.primary.latent_type(),
    LatentType::U32
  ));
  assert!(matches!(meta.per_latent_var.secondary, None));
  assert_eq!(dict.len(), 2000); // this many unique values
  let decompressed = simple_decompress(&compressed)?;
  assert_nums_eq(&decompressed, &nums, "dict mode")?;
  Ok(())
}
