use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use arrow::array::{
  ArrayData, ArrayRef, Float16Array, Float32Array, Float64Array, Int16Array, Int32Array,
  Int64Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::buffer::Buffer;
use arrow::csv;
use arrow::csv::Reader as CsvReader;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatchReader;
use clap::Parser;
use half::f16;
use parquet::arrow::arrow_reader::{ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder};
use parquet::arrow::ProjectionMask;

use pco::data_types::NumberType;
use pco::standalone::simple_decompress;

use crate::{dtypes, parse, utils};

#[cfg(feature = "audio")]
mod audio;

const MAX_INFER_SCHEMA_RECORDS: usize = 1000;

#[derive(Clone, Debug, Default, Parser)]
pub struct InputColumnOpt {
  /// A specific data type to interpret the column as. Only useful for data
  /// coming from CSVs where the type is ambiguous.
  #[arg(long, value_parser = parse::arrow_dtype)]
  pub dtype: Option<DataType>,
  /// Either this or col-idx must be specified.
  #[arg(long)]
  pub col_name: Option<String>,
  /// Either this or col-name must be specified.
  #[arg(long)]
  pub col_idx: Option<usize>,
}

#[derive(clap::ValueEnum, Clone, Copy, Debug, Parser)]
pub enum Format {
  Binary,
  Csv,
  Parquet,
  Pco,
  Wav,
}

#[derive(Clone, Debug, Parser)]
pub struct InputFileOpt {
  /// File or directory to be used as input.
  #[arg(short, long)]
  pub input: Option<PathBuf>,
  #[arg(long)]
  pub input_format: Option<Format>,

  #[arg(long)]
  pub csv_has_header: bool,
  #[arg(long, default_value = ",")]
  pub csv_delimiter: char,
}

impl InputFileOpt {
  fn format(&self) -> Result<Format> {
    if let Some(format) = self.input_format {
      return Ok(format);
    }

    let ext = self
      .input
      .as_ref()
      .and_then(|path| path.extension())
      .and_then(|ext| ext.to_str());
    let format = match ext {
      Some("csv") => Format::Csv,
      Some("parquet") => Format::Parquet,
      Some("pco") => Format::Pco,
      Some("wav") => Format::Wav,
      Some("bin") => Format::Binary,
      _ => {
        return Err(anyhow!(
          "failed to infer input format; consider passing --input-format"
        ))
      }
    };
    Ok(format)
  }
}

fn schema_from_field_paths(mut field_paths: Vec<(Field, PathBuf)>) -> Result<Schema> {
  // To support treating a whole directory of files as a multi-column input for
  // single-column formats, we attach some metadata to the schema to be read
  // back later.
  field_paths.sort_by_key(|(field, _)| field.name().to_string());
  let mut metadata = HashMap::new();
  for (i, (_, path)) in field_paths.iter().enumerate() {
    metadata.insert(
      i.to_string(),
      path.to_str().unwrap().to_string(),
    );
  }
  let fields = field_paths.into_iter().map(|(f, _)| f).collect::<Vec<_>>();
  Ok(Schema::new_with_metadata(fields, metadata))
}

fn single_column_or_filtered_dir_schema<F: Fn(&Path) -> Result<Option<Field>>>(
  path: &Path,
  get_field: F,
) -> Result<Schema> {
  let mut field_paths = Vec::new();

  if path.is_file() {
    if let Some(field) = get_field(path)? {
      field_paths.push((field, path.to_path_buf()));
    }
  } else {
    for entry in fs::read_dir(path)? {
      let file = entry?.path();

      if !file.is_file() {
        continue;
      }

      if let Some(field) = get_field(&file).with_context(|| format!("while reading {:?}", file))? {
        field_paths.push((field, file));
      }
    }
  }

  schema_from_field_paths(field_paths)
}

fn get_binary_field(path: &Path) -> Result<Option<Field>> {
  let no_ext = path
    .file_stem()
    .unwrap()
    .to_str()
    .expect("somehow not unicode");
  let mut split = no_ext.split('_');
  let invalid_filename = || {
    anyhow!(
      "filename must be of the format <DTYPE>_<NAME>, but was {:?}",
      path
    )
  };
  let dtype_str = split.next().ok_or_else(invalid_filename)?;
  let dtype = parse::arrow_dtype(dtype_str)?;
  let name = split.collect::<Vec<_>>().join("_");
  Ok(Some(Field::new(name, dtype, false)))
}

fn infer_binary_schema(dir: &Path) -> Result<Schema> {
  single_column_or_filtered_dir_schema(dir, get_binary_field)
}

fn infer_csv_schema(col_opt: &InputColumnOpt, file_opt: &InputFileOpt) -> Result<Schema> {
  // arrow API is kinda bad right now, so we have to convert the paths
  // back to strings
  let inferred_schema = csv::infer_schema_from_files(
    &[file_opt
      .input
      .as_ref()
      .unwrap()
      .to_str()
      .unwrap()
      .to_string()],
    file_opt.csv_delimiter as u8,
    Some(MAX_INFER_SCHEMA_RECORDS),
    file_opt.csv_has_header,
  )?;

  let Some(dtype) = &col_opt.dtype else {
    return Ok(inferred_schema);
  };

  let mut fields = Vec::new();
  for (col_idx, field) in inferred_schema.fields().iter().enumerate() {
    let new_field = match (&col_opt.col_name, &col_opt.col_idx) {
      (Some(name), None) if name == field.name() => Field::new(name, dtype.clone(), false),
      (None, Some(idx)) if *idx == col_idx => Field::new(field.name(), dtype.clone(), false),
      _ => field.as_ref().clone(),
    };
    fields.push(new_field);
  }
  Ok(Schema::new(fields))
}

fn infer_parquet_schema(col_opt: &InputColumnOpt, path: &Path) -> Result<Schema> {
  let file = File::open(path)?;
  let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
  let schema = reader.schema().as_ref().clone();

  if let Some(arrow_dtype) = &col_opt.dtype {
    let col_idx = utils::find_col_idx(&schema, col_opt.col_idx, &col_opt.col_name)?;
    let field = schema.field(col_idx);
    if field.data_type() != arrow_dtype {
      return Err(anyhow!(
        "optionally specified dtype {:?} did not match parquet schema {:?}",
        arrow_dtype,
        field.data_type(),
      ));
    }
  }
  Ok(schema)
}

fn get_pco_field(path: &Path) -> Result<Option<Field>> {
  // horribly inefficient, but we're not making performance a concern here yet
  let compressed = fs::read(path)?;
  let field = utils::get_standalone_dtype(&compressed)?.map(|dtype| {
    let name = path.file_stem().unwrap().to_str().unwrap();
    Field::new(name, dtypes::to_arrow(dtype), false)
  });
  Ok(field)
}

fn infer_pco_schema(path: &Path) -> Result<Schema> {
  single_column_or_filtered_dir_schema(path, get_pco_field)
}

#[cfg(feature = "audio")]
fn infer_wav_schema(path: &Path) -> Result<Schema> {
  audio::get_wav_schema(path)
}

#[cfg(not(feature = "audio"))]
fn infer_wav_schema(_path: &Path) -> Result<Schema> {
  Err(anyhow!("not compiled with audio feature"))
}

pub fn get_schema(col_opt: &InputColumnOpt, file_opt: &InputFileOpt) -> Result<Schema> {
  let path = file_opt
    .input
    .as_ref()
    .ok_or_else(|| anyhow!("no input was provided"))?;
  match file_opt.format()? {
    // maybe one day I should structure this better
    Format::Binary => infer_binary_schema(path),
    Format::Csv => infer_csv_schema(col_opt, file_opt),
    Format::Parquet => infer_parquet_schema(col_opt, path),
    Format::Pco => infer_pco_schema(path),
    Format::Wav => infer_wav_schema(path),
  }
  .with_context(|| "while inferring schema")
}

#[cfg(feature = "audio")]
fn new_wav_reader(
  schema: &Schema,
  path: &Path,
  col_idx: usize,
) -> Result<Box<dyn Iterator<Item = Result<ArrayRef>>>> {
  Ok(Box::new(audio::WavColumnReader::new(
    schema, path, col_idx,
  )?))
}

#[cfg(not(feature = "audio"))]
fn new_wav_reader(
  _schema: &Schema,
  _path: &Path,
  _col_idx: usize,
) -> Result<Box<dyn Iterator<Item = Result<ArrayRef>>>> {
  Err(anyhow!("not compiled with audio feature"))
}

pub fn new_column_reader(
  schema: &Schema,
  col_idx: usize,
  opt: &InputFileOpt,
) -> Result<Box<dyn Iterator<Item = Result<ArrayRef>>>> {
  let path = opt.input.as_ref().unwrap();
  let res: Box<dyn Iterator<Item = Result<ArrayRef>>> = match opt.format()? {
    Format::Binary => Box::new(BinaryColumnReader::new(schema, col_idx)?),
    Format::Csv => Box::new(CsvColumnReader::new(
      schema, path, col_idx, opt,
    )?),
    Format::Parquet => Box::new(ParquetColumnReader::new(
      schema, path, col_idx,
    )?),
    Format::Pco => Box::new(PcoColumnReader::new(schema, col_idx)?),
    Format::Wav => new_wav_reader(schema, path, col_idx)?,
  };
  Ok(res)
}

struct BinaryColumnReader {
  col_path: PathBuf,
  dtype: DataType,
  did_read: bool,
}

impl BinaryColumnReader {
  fn new(schema: &Schema, col_idx: usize) -> Result<Self> {
    let col_path = PathBuf::from(schema.metadata.get(&col_idx.to_string()).unwrap());
    let dtype = schema.field(col_idx).data_type().clone();
    Ok(BinaryColumnReader {
      col_path,
      dtype,
      did_read: false,
    })
  }
}

impl BinaryColumnReader {
  fn get_array(&self) -> Result<ArrayRef> {
    let bytes = fs::read(&self.col_path)?;
    let n_bytes = bytes.len();
    let buffer = Buffer::from_vec(bytes);

    let array_data = ArrayData::builder(self.dtype.clone())
      .add_buffer(buffer)
      .len(n_bytes / self.dtype.primitive_width().unwrap())
      .build()?;
    let array = arrow::array::make_array(array_data);

    Ok(array)
  }
}

impl Iterator for BinaryColumnReader {
  type Item = Result<ArrayRef>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.did_read {
      return None;
    }

    self.did_read = true;
    Some(self.get_array())
  }
}

struct ParquetColumnReader(ParquetRecordBatchReader);

impl ParquetColumnReader {
  fn new(schema: &Schema, path: &Path, col_idx: usize) -> Result<Self> {
    let file = File::open(path)?;
    let batch_reader_builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let parquet_schema = parquet::arrow::arrow_to_parquet_schema(schema)?;
    let batch_reader = batch_reader_builder
      .with_projection(ProjectionMask::leaves(
        &parquet_schema,
        vec![col_idx],
      ))
      .build()?;
    Ok(Self(batch_reader))
  }
}

impl Iterator for ParquetColumnReader {
  type Item = Result<ArrayRef>;

  fn next(&mut self) -> Option<Result<ArrayRef>> {
    self.0.next().map(|batch_result| {
      let batch = batch_result?;
      // 0 because we told arrow to only read the exact column we want
      Ok(batch.column(0).clone())
    })
  }
}

struct CsvColumnReader {
  csv_reader: CsvReader<File>,
  col_idx: usize,
}

impl CsvColumnReader {
  fn new(schema: &Schema, path: &Path, col_idx: usize, opt: &InputFileOpt) -> Result<Self> {
    let csv_reader = csv::ReaderBuilder::new(SchemaRef::new(schema.clone()))
      .with_header(opt.csv_has_header)
      .with_delimiter(opt.csv_delimiter as u8)
      .build(File::open(path)?)?;

    Ok(Self {
      csv_reader,
      col_idx,
    })
  }
}

impl Iterator for CsvColumnReader {
  type Item = Result<ArrayRef>;

  fn next(&mut self) -> Option<Result<ArrayRef>> {
    self.csv_reader.next().map(|batch_result| {
      let batch = batch_result?;
      Ok(batch.column(self.col_idx).clone())
    })
  }
}

struct PcoColumnReader {
  col_path: PathBuf,
  dtype: NumberType,
  did_read: bool,
}

impl PcoColumnReader {
  fn new(schema: &Schema, col_idx: usize) -> Result<Self> {
    let col_path = PathBuf::from(schema.metadata.get(&col_idx.to_string()).unwrap());
    let dtype = dtypes::from_arrow(&schema.field(col_idx).data_type().clone())?;
    Ok(PcoColumnReader {
      col_path,
      dtype,
      did_read: false,
    })
  }
}

impl PcoColumnReader {
  fn get_array(&self) -> Result<ArrayRef> {
    use NumberType::*;

    let compressed = fs::read(&self.col_path)?;
    let array: ArrayRef = match self.dtype {
      F16 => Arc::new(Float16Array::from(simple_decompress::<f16>(
        &compressed,
      )?)),
      F32 => Arc::new(Float32Array::from(simple_decompress::<f32>(
        &compressed,
      )?)),
      F64 => Arc::new(Float64Array::from(simple_decompress::<f64>(
        &compressed,
      )?)),
      I16 => Arc::new(Int16Array::from(simple_decompress::<i16>(
        &compressed,
      )?)),
      I32 => Arc::new(Int32Array::from(simple_decompress::<i32>(
        &compressed,
      )?)),
      I64 => Arc::new(Int64Array::from(simple_decompress::<i64>(
        &compressed,
      )?)),
      U16 => Arc::new(UInt16Array::from(simple_decompress::<u16>(
        &compressed,
      )?)),
      U32 => Arc::new(UInt32Array::from(simple_decompress::<u32>(
        &compressed,
      )?)),
      U64 => Arc::new(UInt64Array::from(simple_decompress::<u64>(
        &compressed,
      )?)),
      other => {
        return Err(anyhow!(
          "number type {:?} not yet supported in pco_cli",
          other
        ))
      }
    };
    Ok(array)
  }
}

impl Iterator for PcoColumnReader {
  type Item = Result<ArrayRef>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.did_read {
      return None;
    }

    self.did_read = true;
    Some(self.get_array())
  }
}
