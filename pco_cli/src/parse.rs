use anyhow::anyhow;
use arrow::datatypes::{DataType, TimeUnit};

use pco::{DeltaSpec, ModeSpec};

pub fn delta_spec(s: &str) -> anyhow::Result<DeltaSpec> {
  let spec = match s.to_lowercase().as_str() {
    "auto" => DeltaSpec::Auto,
    "noop" => DeltaSpec::NoOp,
    "lookback" => DeltaSpec::TryLookback,
    other => {
      let mut parts = other.split('@');
      let name = parts.next().unwrap();
      let err = || anyhow!("invalid delta spec: {}", s);
      let value = parts.next().ok_or_else(err)?;
      match name {
        "consecutive" => DeltaSpec::TryConsecutive(value.parse()?),
        "conv1" => DeltaSpec::TryConv1(value.parse()?),
        _ => return Err(err()),
      }
    }
  };
  Ok(spec)
}

pub fn mode_spec(s: &str) -> anyhow::Result<ModeSpec> {
  let spec = match s.to_lowercase().as_str() {
    "auto" => ModeSpec::Auto,
    "classic" => ModeSpec::Classic,
    "dict" => ModeSpec::TryDict,
    other => {
      let mut parts = other.split('@');
      let name = parts.next().unwrap();
      let err = || anyhow!("invalid mode spec: {}", s);
      let value = parts.next().ok_or_else(err)?;
      match name {
        "floatmult" => ModeSpec::TryFloatMult(value.parse()?),
        "floatquant" => ModeSpec::TryFloatQuant(value.parse()?),
        "intmult" => ModeSpec::TryIntMult(value.parse()?),
        _ => return Err(err()),
      }
    }
  };
  Ok(spec)
}

pub fn arrow_dtype(s: &str) -> anyhow::Result<DataType> {
  let name_pairs = [
    ("f16", DataType::Float16),
    ("f32", DataType::Float32),
    ("f64", DataType::Float64),
    ("i16", DataType::Int16),
    ("i32", DataType::Int32),
    ("i64", DataType::Int64),
    ("u16", DataType::UInt16),
    ("u32", DataType::UInt32),
    ("u64", DataType::UInt64),
    (
      "seconds",
      DataType::Timestamp(TimeUnit::Second, None),
    ),
    (
      "millis",
      DataType::Timestamp(TimeUnit::Millisecond, None),
    ),
    (
      "micros",
      DataType::Timestamp(TimeUnit::Microsecond, None),
    ),
    (
      "nanos",
      DataType::Timestamp(TimeUnit::Nanosecond, None),
    ),
  ];

  let lower = s.to_lowercase();
  for (name, dtype) in &name_pairs {
    if name == &lower {
      return Ok(dtype.clone());
    }
  }

  Err(anyhow!(
    "invalid data type: {}. Expected one of: {:?}",
    lower,
    name_pairs
      .iter()
      .map(|(name, _)| name.to_string())
      .collect::<Vec<_>>()
  ))
}
