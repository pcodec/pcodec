use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Result};
use arrow::array::{ArrayRef, Float32Array, Int16Array, Int32Array, UInt16Array};
use arrow::datatypes::{DataType, Field, Schema};
use wav::BitDepth;

pub fn get_wav_schema(path: &Path) -> Result<Schema> {
  // this is excessively slow, but easy for now
  let mut file = File::open(path)?;
  let (header, _) = wav::read(&mut file)?;
  let dtype = match (header.audio_format, header.bits_per_sample) {
    (wav::WAV_FORMAT_PCM, 8 | 16) => Ok(DataType::Int16),
    (wav::WAV_FORMAT_PCM, 24 | 32) => Ok(DataType::Int32),
    (wav::WAV_FORMAT_IEEE_FLOAT, 32) => Ok(DataType::Float32),
    _ => Err(anyhow!(
      "audio format {} with {} bits per sample not supported",
      header.audio_format,
      header.bits_per_sample
    )),
  }?;
  let name = path
    .file_stem()
    .unwrap()
    .to_str()
    .expect("somehow not unicode");

  let fields: Vec<Field> = (0..header.channel_count)
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
    let mut inp_file = File::open(&self.path)?;
    let (header, data) = wav::read(&mut inp_file)?;

    macro_rules! make_channel_array {
      ($data:ident, $array_type:ty) => {
        Arc::new(<$array_type>::from(filter_to_channel(
          $data,
          self.channel_idx,
          header.channel_count,
        )))
      };
    }

    let array: ArrayRef = match data {
      BitDepth::Eight(u8s) => {
        let u16s = u8s.into_iter().map(|x| x as u16).collect::<Vec<_>>();
        make_channel_array!(u16s, UInt16Array)
      }
      BitDepth::Sixteen(i16s) => make_channel_array!(i16s, Int16Array),
      BitDepth::TwentyFour(i32s) => make_channel_array!(i32s, Int32Array),
      BitDepth::ThirtyTwoFloat(f32s) => make_channel_array!(f32s, Float32Array),
      BitDepth::Empty => {
        if self.dtype == DataType::Int32 {
          Arc::new(Int32Array::from(Vec::<i32>::new()))
        } else if self.dtype == DataType::Float32 {
          Arc::new(Float32Array::from(Vec::<f32>::new()))
        } else {
          Arc::new(Int16Array::from(Vec::<i16>::new()))
        }
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
