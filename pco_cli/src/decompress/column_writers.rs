use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::PrimitiveArray;
use arrow::csv::WriterBuilder as CsvWriterBuilder;
use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};
use arrow::record_batch::RecordBatch;

use crate::decompress::DecompressOpt;
use crate::decompress::OutputKind::*;
use crate::dtypes::PcoNumber;

pub fn new<T: PcoNumber>(opt: &DecompressOpt) -> Result<Box<dyn ColumnWriter<T>>> {
  // eventually we'll likely have a txt writer and a parquet writer, etc.
  let writer: Box<dyn ColumnWriter<T>> = match opt.output {
    Txt => Box::<TxtWriter<T>>::default(),
    Binary => Box::<BinaryWriter<T>>::default(),
  };
  Ok(writer)
}

pub trait ColumnWriter<T: PcoNumber> {
  fn write(&mut self, nums: Vec<<T::Arrow as ArrowPrimitiveType>::Native>) -> Result<()>;
  fn close(&mut self) -> Result<()>;
}

#[derive(Default)]
struct TxtWriter<T: PcoNumber> {
  phantom: PhantomData<T>,
}

impl<T: PcoNumber> ColumnWriter<T> for TxtWriter<T> {
  fn write(&mut self, arrow_natives: Vec<<T::Arrow as ArrowPrimitiveType>::Native>) -> Result<()> {
    let schema = Schema::new(vec![Field::new("c0", T::ARROW_DTYPE, false)]);
    let c0 = PrimitiveArray::<T::Arrow>::from_iter_values(arrow_natives);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(c0)])?;
    let mut stdout_bytes = Vec::<u8>::new();
    {
      let mut writer = CsvWriterBuilder::new()
        .with_header(false)
        .build(&mut stdout_bytes);
      writer.write(&batch)?;
    }
    print!("{}", String::from_utf8(stdout_bytes)?);
    Ok(())
  }

  fn close(&mut self) -> Result<()> {
    Ok(())
  }
}

#[derive(Default)]
struct BinaryWriter<T: PcoNumber> {
  phantom: PhantomData<T>,
}

impl<T: PcoNumber> ColumnWriter<T> for BinaryWriter<T> {
  fn write(&mut self, arrow_natives: Vec<<T::Arrow as ArrowPrimitiveType>::Native>) -> Result<()> {
    let mut out = std::io::stdout();
    for &x in &arrow_natives {
      out.write_all(&T::arrow_native_to_bytes(x))?;
    }
    Ok(())
  }

  fn close(&mut self) -> Result<()> {
    std::io::stdout().flush()?;
    Ok(())
  }
}
