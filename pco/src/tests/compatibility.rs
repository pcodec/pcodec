use crate::data_types::{Number, NumberType};
use crate::errors::PcoResult;
use crate::standalone::FileCompressor;
use crate::{standalone, ChunkConfig, DeltaSpec, ModeSpec};
use half::f16;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

fn get_asset_dir() -> PathBuf {
  PathBuf::from_str(env!("CARGO_MANIFEST_DIR"))
    .unwrap()
    .join("assets")
}

fn get_pco_path(version: &str, name: &str) -> PathBuf {
  get_asset_dir().join(format!(
    "v{}_{}.pco",
    version.replace('.', "_"),
    name,
  ))
}

fn assert_nums_eq<T: Number>(x: &[T], y: &[T]) {
  assert_eq!(x.len(), y.len());
  for (i, (x, y)) in x.iter().zip(y).enumerate() {
    assert_eq!(
      x.to_latent_ordered(),
      y.to_latent_ordered(),
      "{} != {} at {}",
      x,
      y,
      i
    );
  }
}

fn assert_compatible<T: Number>(version: &str, name: &str, expected: &[T]) -> PcoResult<()> {
  let pco_path = get_pco_path(version, name);

  let compressed = fs::read(pco_path)?;
  let decompressed = standalone::simple_decompress::<T>(&compressed)?;

  assert_nums_eq(&decompressed, expected);
  Ok(())
}

fn needs_write(version: &str, path: &Path) -> bool {
  version == env!("CARGO_PKG_VERSION") && !path.exists()
}

fn simple_write_if_version_matches<T: Number>(
  version: &str,
  name: &str,
  nums: &[T],
  config: &ChunkConfig,
) -> PcoResult<()> {
  let path = get_pco_path(version, name);
  if !needs_write(version, &path) {
    return Ok(());
  }

  fs::write(
    path,
    standalone::simple_compress(nums, config)?,
  )?;
  Ok(())
}

#[test]
fn v0_0_0_classic() -> PcoResult<()> {
  let version = "0.0.0";
  let name = "classic";
  let nums = (0_i32..1000).chain(2000..3000).collect::<Vec<_>>();
  let config = ChunkConfig {
    delta_spec: DeltaSpec::NoOp,
    ..Default::default()
  };
  simple_write_if_version_matches(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v0_0_0_delta_float_mult() -> PcoResult<()> {
  let version = "0.0.0";
  let name = "delta_float_mult";
  let mut nums = (0..2000).map(|i| i as f32).collect::<Vec<_>>();
  nums[1337] += 1.001;
  let config = ChunkConfig {
    delta_spec: DeltaSpec::TryConsecutive(1),
    ..Default::default()
  };
  simple_write_if_version_matches(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v0_1_0_delta_int_mult() -> PcoResult<()> {
  // starting at 0.1.0 because 0.0.0 had GCD mode (no longer supported)
  // instead of int mult
  let version = "0.1.0";
  let name = "delta_int_mult";
  let mut nums = (0..2000).map(|i| i * 1000).collect::<Vec<_>>();
  nums[1337] -= 1;
  let config = ChunkConfig {
    delta_spec: DeltaSpec::TryConsecutive(1),
    ..Default::default()
  };
  simple_write_if_version_matches(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v0_1_1_classic() -> PcoResult<()> {
  // v0.1.1 introduced standalone versioning, separate from wrapped versioning
  let version = "0.1.1";
  let name = "standalone_versioned";
  let nums = vec![];
  let config = ChunkConfig::default();
  simple_write_if_version_matches::<f32>(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

fn generate_pseudorandom_f16s() -> Vec<f16> {
  // makes a variety of floats approximately uniformly distributed
  // between (-2.0, -1.0] U [1.0, 2.0)
  let mut num = 0.1_f32;
  let mut nums = vec![];
  for _ in 0..2000 {
    num = ((num * 77.7) + 0.1) % 2.0;
    if num < 1.0 {
      nums.push(f16::from_f32(-1.0 - num));
    } else {
      nums.push(f16::from_f32(num));
    }
  }
  nums
}

#[test]
fn v0_3_0_f16() -> PcoResult<()> {
  // v0.3.0 introduced 16-bit number types, including f16, which requires the
  // half crate
  let version = "0.3.0";
  let name = "f16";
  let config = ChunkConfig::default();
  let nums = generate_pseudorandom_f16s();
  simple_write_if_version_matches::<f16>(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v0_3_0_float_quant() -> PcoResult<()> {
  // v0.3.0 introduced float quantization mode
  let version = "0.3.0";
  let name = "float_quant";
  let nums = generate_pseudorandom_f16s()
    .into_iter()
    .map(|x| {
      let x = x.to_f32();
      if x.abs() < 1.1 {
        f32::from_bits(x.to_bits() + 1)
      } else {
        x
      }
    })
    .collect::<Vec<_>>();
  let config = ChunkConfig::default().with_mode_spec(ModeSpec::TryFloatQuant(
    f32::MANTISSA_DIGITS - f16::MANTISSA_DIGITS,
  ));
  simple_write_if_version_matches::<f32>(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v0_4_0_lookback_delta() -> PcoResult<()> {
  // v0.4.0 introduced lookback delta encoding
  let version = "0.4.0";
  let name = "lookback_delta";

  // randomly generated ahead of time
  let nums: Vec<u32> = vec![
    1121827092, 729032807, 3968137854, 2875434067, 3775328080, 431649926, 1048116090, 1906978350,
    14752788, 1180462487,
  ]
  .repeat(100);
  let config = ChunkConfig::default().with_delta_spec(DeltaSpec::TryLookback);
  simple_write_if_version_matches(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v0_4_5_uniform_type() -> PcoResult<()> {
  // v0.4.5 introduced optional uniform types in standalone
  let version = "0.4.5";
  let name = "uniform_type";

  // we write as two chunks for good measure
  let nums: Vec<u32> = vec![1, 2, 3, 4, 5];
  let config = ChunkConfig::default();

  let path = get_pco_path(version, name);
  if needs_write(version, &path) {
    let mut dst = vec![];
    let fc = FileCompressor::default().with_uniform_type(Some(NumberType::U32));
    fc.write_header(&mut dst)?;
    fc.chunk_compressor(&nums[0..3], &config)?
      .write_chunk(&mut dst)?;
    fc.chunk_compressor(&nums[3..5], &config)?
      .write_chunk(&mut dst)?;
    fc.write_footer(&mut dst)?;
    fs::write(path, dst)?;
  }

  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v0_4_8_minor_version() -> PcoResult<()> {
  // v0.4.8 introduced the minor (wrapped) format version
  let version = "0.4.8";
  let name = "minor_version";

  let nums: Vec<u32> = vec![1, 2, 3, 4, 5];
  let config = ChunkConfig::default();

  let path = get_pco_path(version, name);
  if needs_write(version, &path) {
    let mut dst = vec![];
    let fc = FileCompressor::default().with_max_supported_version();
    fc.write_header(&mut dst)?;
    fc.chunk_compressor(&nums, &config)?.write_chunk(&mut dst)?;
    fc.write_footer(&mut dst)?;
    fs::write(path, dst)?;
  }

  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v1_0_0_dict() -> PcoResult<()> {
  // v1.0.0 introduced dict mode
  let version = "1.0.0";
  let name = "dict";
  let nums = vec![8924659283, 234897984367, 9827358920].repeat(1000);
  let config = ChunkConfig::default()
    .with_mode_spec(ModeSpec::TryDict)
    .with_delta_spec(DeltaSpec::NoOp);
  simple_write_if_version_matches::<u64>(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v1_0_0_conv1() -> PcoResult<()> {
  // v1.0.0 introduced conv1 delta encoding
  let version = "1.0.0";
  let name = "conv1";
  let mut xm1 = 0.0;
  let mut xm2 = 0.0;
  let mut nums = vec![];
  for i in 0..2000 {
    let x = (xm1 as f32) * 1.99 - (xm2 as f32) + ((i * 47) % 77 - 38) as f32;
    nums.push((x + 10000.0) as i32);
    xm2 = xm1;
    xm1 = x;
  }
  let config = ChunkConfig::default().with_delta_spec(DeltaSpec::TryConv1(2));
  simple_write_if_version_matches::<i32>(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v1_0_0_u8() -> PcoResult<()> {
  // v1.0.0 introduced 8-bit unsigned integer type
  let version = "1.0.0";
  let name = "u8";
  let config = ChunkConfig::default().with_enable_8_bit(true);
  let nums = (0_u8..=64).chain(192..=255).collect::<Vec<_>>();
  simple_write_if_version_matches::<u8>(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}

#[test]
fn v1_0_0_i8() -> PcoResult<()> {
  // v1.0.0 introduced 8-bit signed integer type
  let version = "1.0.0";
  let name = "i8";
  let config = ChunkConfig::default().with_enable_8_bit(true);
  let nums = (-128_i8..=-64).chain(64..=127).collect::<Vec<_>>();
  simple_write_if_version_matches::<i8>(version, name, &nums, &config)?;
  assert_compatible(version, name, &nums)?;
  Ok(())
}
