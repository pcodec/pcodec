use std::marker::PhantomData;

use anyhow::{anyhow, Result};
use arrow::datatypes::*;

use crate::compress_handler::CompressHandler;
use crate::dtypes::ArrowNumberLike;

fn new_boxed_handler<P: ArrowNumberLike>() -> Box<dyn ArrowHandler> {
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
    Float32 => Float32Type,
    Float64 => Float64Type,
    Int64 => Int64Type,
    Int64 => Int64Type,
    UInt32 => UInt64Type,
    UInt64 => UInt64Type,
    Timestamp(TimeUnit::Microsecond, _) => TimestampMicrosecondType,
    Timestamp(TimeUnit::Nanosecond, _) => TimestampNanosecondType,
  )
}

pub trait ArrowHandler: CompressHandler {}

#[derive(Clone, Debug, Default)]
pub struct ArrowHandlerImpl<P> {
  phantom: PhantomData<P>,
}

impl<P: ArrowNumberLike> ArrowHandler for ArrowHandlerImpl<P> {}
