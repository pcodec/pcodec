use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use arrow::datatypes::*;

use crate::bench::handler::BenchHandler;
use crate::compress::handler::CompressHandler;
use crate::dtypes::ArrowNumber;

fn new_boxed_handler<P: ArrowNumber>() -> Box<dyn ArrowHandler> {
  Box::new(ArrowHandlerImpl {
    phantom: PhantomData::<P>,
  })
}

pub fn from_dtype(dtype: &DataType) -> Result<Box<dyn ArrowHandler>> {
  use DataType::*;

  macro_rules! match_dtype {
    {$($name:pat => $t:ty,)+} => {
      match dtype {
        $(&$name => Ok(new_boxed_handler::<$t>()),)+
        _ => Err(anyhow!("unsupported arrow dtype: {:?}", dtype))
      }
    }
  }

  match_dtype!(
    Float16 => Float16Type,
    Float32 => Float32Type,
    Float64 => Float64Type,
    Int8 => Int8Type,
    Int16 => Int16Type,
    Int32 => Int32Type,
    Int64 => Int64Type,
    UInt8 => UInt8Type,
    UInt16 => UInt16Type,
    UInt32 => UInt32Type,
    UInt64 => UInt64Type,
    Timestamp(TimeUnit::Second, _) => TimestampSecondType,
    Timestamp(TimeUnit::Millisecond, _) => TimestampMillisecondType,
    Timestamp(TimeUnit::Microsecond, _) => TimestampMicrosecondType,
    Timestamp(TimeUnit::Nanosecond, _) => TimestampNanosecondType,
    Date32 => Date32Type,
    Date64 => Date64Type,
  )
}

pub trait ArrowHandler: CompressHandler + BenchHandler {}

#[derive(Clone, Debug, Default)]
pub struct ArrowHandlerImpl<P> {
  phantom: PhantomData<P>,
}

impl<P: ArrowNumber> ArrowHandler for ArrowHandlerImpl<P> {}
