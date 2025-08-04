use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow::array::{ArrayRef, Float32Array, Int16Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use hound::{SampleFormat, WavReader};

pub fn get_wav_schema(path: &Path) -> Result<Schema> {
  // this is excessively slow, but easy for now
  let reader = WavReader::open(path)?;
  let header = reader.spec();
  let dtype = match (header.sample_format, header.bits_per_sample) {
    (SampleFormat::Int, 8 | 16) => Ok(DataType::Int16),
    (SampleFormat::Int, 24 | 32) => Ok(DataType::Int32),
    (SampleFormat::Float, 32) => Ok(DataType::Float32),
    _ => Err(anyhow!(
      "audio format {:?} with {} bits per sample not supported",
      header.sample_format,
      header.bits_per_sample
    )),
  }?;
  let name = path
    .file_stem()
    .unwrap()
    .to_str()
    .expect("somehow not unicode");

  let fields: Vec<Field> = (0..header.channels)
    .map(|i| {
      Field::new(
        format!("{}_channel_{}", name, i),
        dtype.clone(),
        false,
      )
    })
    .collect();
  Ok(Schema::new(fields))
}

pub struct WavColumnReader {
  path: PathBuf,
  dtype: DataType,
  channel_idx: usize,
  did_read: bool,
}

impl WavColumnReader {
  pub fn new(schema: &Schema, path: &Path, col_idx: usize) -> Result<Self> {
    let dtype = schema.field(col_idx).data_type().clone();
    Ok(WavColumnReader {
      path: PathBuf::from(path),
      dtype,
      channel_idx: col_idx,
      did_read: false,
    })
  }
}

fn filter_to_channel<T>(data: Vec<T>, channel_idx: usize, channel_count: u16) -> Vec<T> {
  data
    .into_iter()
    .skip(channel_idx)
    .step_by(channel_count as usize)
    .collect()
}

impl WavColumnReader {
  fn get_array(&self) -> Result<ArrayRef> {
    let mut reader = WavReader::open(&self.path)?;
    let header = reader.spec();

    macro_rules! make_channel_array {
      ($reader:ident, $t:ty, $array_type:ty) => {{
        let data = reader.samples::<$t>().collect::<Result<Vec<_>, _>>()?;
        Arc::new(<$array_type>::from(filter_to_channel(
          data,
          self.channel_idx,
          header.channels,
        )))
      }};
    }

    let array: ArrayRef = match self.dtype {
      DataType::Int16 => make_channel_array!(reader, i16, Int16Array),
      DataType::Int32 => make_channel_array!(reader, i32, Int32Array),
      DataType::Float32 => make_channel_array!(reader, f32, Float32Array),
      _ => {
        return Err(anyhow!(
          "Unsupported data type: {:?}",
          self.dtype
        ))
      }
    };
    Ok(array)
  }
}

impl Iterator for WavColumnReader {
  type Item = Result<ArrayRef>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.did_read {
      return None;
    }

    self.did_read = true;
    Some(self.get_array())
  }
}
