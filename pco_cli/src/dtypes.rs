use std::{any, mem};

use anyhow::anyhow;
use anyhow::Result;
use arrow::datatypes as arrow_dtypes;
use arrow::datatypes::DataType as ArrowDataType;
use arrow::datatypes::{ArrowPrimitiveType, DataType};
use half::f16;
use pco::data_types::{Number, NumberType};

use crate::num_vec::NumVec;

pub trait Parquetable: Sized {
  const PARQUET_DTYPE_STR: &'static str;
  const TRANSMUTABLE: bool = true;

  type Parquet: parquet::data_type::DataType;

  fn transmute_nums_to_parquet(
    _nums: &[Self],
  ) -> &[<Self::Parquet as parquet::data_type::DataType>::T] {
    panic!(
      "transmutation for Parquet {} type has not yet been implemented in Pco CLI",
      any::type_name::<Self>()
    )
  }
  fn copy_nums_to_parquet(
    _nums: &[Self],
  ) -> Vec<<Self::Parquet as parquet::data_type::DataType>::T> {
    panic!(
      "conversion for Parquet {} type has not yet been implemented in Pco CLI",
      any::type_name::<Self>()
    )
  }
  fn parquet_to_nums(vec: Vec<<Self::Parquet as parquet::data_type::DataType>::T>) -> Vec<Self>;
}

#[cfg(feature = "full_bench")]
pub trait QCompressable: Sized {
  type Qco: q_compress::data_types::NumberLike;

  fn nums_to_qco(nums: &[Self]) -> &[Self::Qco];
  fn qco_to_nums(vec: Vec<Self::Qco>) -> Vec<Self>;
}

#[cfg(feature = "full_bench")]
pub trait TurboPforable: Sized {
  unsafe fn encode(src: &mut [Self], dst: &mut [u8]) -> usize;
  unsafe fn decode(src: &mut [u8], n: usize, dst: &mut [Self]);
}

#[cfg(feature = "full_bench")]
pub trait PcoNumber:
  Number + Parquetable + QCompressable + TurboPforable + vortex::dtype::NativePType
{
  const ARROW_DTYPE: DataType;

  type Arrow: ArrowPrimitiveType;

  fn to_arrow_native(self) -> <Self::Arrow as ArrowPrimitiveType>::Native;
  fn make_num_vec(nums: Vec<Self>) -> NumVec;
  fn arrow_native_to_bytes(x: <Self::Arrow as ArrowPrimitiveType>::Native) -> Vec<u8>;
}

#[cfg(not(feature = "full_bench"))]
pub trait PcoNumber: Number + Parquetable {
  const ARROW_DTYPE: DataType;

  type Arrow: ArrowPrimitiveType;

  fn to_arrow_native(self) -> <Self::Arrow as ArrowPrimitiveType>::Native;
  fn make_num_vec(nums: Vec<Self>) -> NumVec;
  fn arrow_native_to_bytes(x: <Self::Arrow as ArrowPrimitiveType>::Native) -> Vec<u8>;
}

pub trait ArrowNumber: ArrowPrimitiveType {
  type Pco: PcoNumber;

  fn native_to_pco(native: Self::Native) -> Self::Pco;

  fn native_vec_to_pco(native: Vec<Self::Native>) -> Vec<Self::Pco>;
}

macro_rules! parquetable {
  ($t: ty, $parq: ty, $parq_str: expr) => {
    impl Parquetable for $t {
      const PARQUET_DTYPE_STR: &'static str = $parq_str;

      type Parquet = $parq;

      fn transmute_nums_to_parquet(
        nums: &[Self],
      ) -> &[<Self::Parquet as parquet::data_type::DataType>::T] {
        nums
      }
      fn parquet_to_nums(
        vec: Vec<<Self::Parquet as parquet::data_type::DataType>::T>,
      ) -> Vec<Self> {
        vec
      }
    }
  };
}

macro_rules! trivial {
  ($t: ty, $name: ident, $p: ty) => {
    #[cfg(feature = "full_bench")]
    impl QCompressable for $t {
      type Qco = $t;

      fn nums_to_qco(nums: &[Self]) -> &[Self::Qco] {
        nums
      }
      fn qco_to_nums(vec: Vec<Self::Qco>) -> Vec<Self> {
        vec
      }
    }

    impl PcoNumber for $t {
      const ARROW_DTYPE: DataType = <$p as ArrowPrimitiveType>::DATA_TYPE;

      type Arrow = $p;

      fn to_arrow_native(self) -> <Self::Arrow as ArrowPrimitiveType>::Native {
        self as Self
      }

      fn make_num_vec(nums: Vec<Self>) -> NumVec {
        NumVec::$name(nums)
      }

      fn arrow_native_to_bytes(x: <Self::Arrow as ArrowPrimitiveType>::Native) -> Vec<u8> {
        x.to_le_bytes().to_vec()
      }
    }

    impl ArrowNumber for $p {
      type Pco = $t;

      fn native_to_pco(native: Self::Native) -> Self::Pco {
        native as Self::Pco
      }

      fn native_vec_to_pco(native: Vec<Self::Native>) -> Vec<Self::Pco> {
        native
      }
    }
  };
}

macro_rules! extra_arrow {
  ($t: ty, $p: ty) => {
    impl ArrowNumber for $p {
      type Pco = $t;

      fn native_to_pco(native: Self::Native) -> Self::Pco {
        native as Self::Pco
      }

      fn native_vec_to_pco(native: Vec<Self::Native>) -> Vec<Self::Pco> {
        native
      }
    }
  };
}

parquetable!(f32, parquet::data_type::FloatType, "FLOAT");
parquetable!(f64, parquet::data_type::DoubleType, "DOUBLE");
parquetable!(i32, parquet::data_type::Int32Type, "INT32");
parquetable!(i64, parquet::data_type::Int64Type, "INT64");

// For 16-bit types, we have no way to transmute into parquet types, so we need
// to copy.
impl Parquetable for f16 {
  const PARQUET_DTYPE_STR: &'static str = "FLOAT";
  const TRANSMUTABLE: bool = false;
  // Would love to use half float representation here, but rust arrow doesn't
  // have a good one yet.
  type Parquet = parquet::data_type::FloatType;

  fn copy_nums_to_parquet(nums: &[Self]) -> Vec<f32> {
    nums.iter().map(|x| x.to_f32()).collect()
  }
  fn parquet_to_nums(vec: Vec<f32>) -> Vec<Self> {
    vec.into_iter().map(f16::from_f32).collect()
  }
}

impl Parquetable for i16 {
  const PARQUET_DTYPE_STR: &'static str = "INT32";
  const TRANSMUTABLE: bool = false;
  type Parquet = parquet::data_type::Int32Type;

  fn copy_nums_to_parquet(nums: &[Self]) -> Vec<i32> {
    nums.iter().map(|&x| x as i32).collect()
  }
  fn parquet_to_nums(vec: Vec<i32>) -> Vec<Self> {
    vec.into_iter().map(|x| x as i16).collect()
  }
}

impl Parquetable for u16 {
  const PARQUET_DTYPE_STR: &'static str = "INT32";
  const TRANSMUTABLE: bool = false;
  type Parquet = parquet::data_type::Int32Type;

  fn copy_nums_to_parquet(nums: &[Self]) -> Vec<i32> {
    nums.iter().map(|&x| x as i32).collect()
  }
  fn parquet_to_nums(vec: Vec<i32>) -> Vec<Self> {
    vec.into_iter().map(|x| x as u16).collect()
  }
}

// Parquet doesn't have unsigned integer types, but to be as fair and fast as
// possible, we transmute here.
// Numerical value is not preserved, but Parquet's compression ratio is.
impl Parquetable for u32 {
  const PARQUET_DTYPE_STR: &'static str = "INT32";
  type Parquet = parquet::data_type::Int32Type;

  fn transmute_nums_to_parquet(
    nums: &[Self],
  ) -> &[<Self::Parquet as parquet::data_type::DataType>::T] {
    unsafe { mem::transmute(nums) }
  }
  fn parquet_to_nums(vec: Vec<<Self::Parquet as parquet::data_type::DataType>::T>) -> Vec<Self> {
    unsafe { mem::transmute(vec) }
  }
}

impl Parquetable for u64 {
  const PARQUET_DTYPE_STR: &'static str = "INT64";
  type Parquet = parquet::data_type::Int64Type;

  fn transmute_nums_to_parquet(
    nums: &[Self],
  ) -> &[<Self::Parquet as parquet::data_type::DataType>::T] {
    unsafe { mem::transmute(nums) }
  }
  fn parquet_to_nums(vec: Vec<<Self::Parquet as parquet::data_type::DataType>::T>) -> Vec<Self> {
    unsafe { mem::transmute(vec) }
  }
}

#[cfg(feature = "full_bench")]
impl QCompressable for f16 {
  type Qco = u16;

  fn nums_to_qco(nums: &[Self]) -> &[Self::Qco] {
    unsafe { mem::transmute(nums) }
  }
  fn qco_to_nums(vec: Vec<Self::Qco>) -> Vec<Self> {
    unsafe { mem::transmute(vec) }
  }
}

impl PcoNumber for f16 {
  const ARROW_DTYPE: DataType = arrow_dtypes::Float16Type::DATA_TYPE;

  type Arrow = arrow_dtypes::Float16Type;

  fn to_arrow_native(self) -> <Self::Arrow as ArrowPrimitiveType>::Native {
    self as Self
  }

  fn make_num_vec(nums: Vec<Self>) -> NumVec {
    NumVec::F16(nums)
  }

  fn arrow_native_to_bytes(x: <Self::Arrow as ArrowPrimitiveType>::Native) -> Vec<u8> {
    x.to_le_bytes().to_vec()
  }
}

trivial!(f32, F32, arrow_dtypes::Float32Type);
trivial!(f64, F64, arrow_dtypes::Float64Type);
trivial!(i16, I16, arrow_dtypes::Int16Type);
trivial!(i32, I32, arrow_dtypes::Int32Type);
trivial!(i64, I64, arrow_dtypes::Int64Type);
trivial!(u16, U16, arrow_dtypes::UInt16Type);
trivial!(u32, U32, arrow_dtypes::UInt32Type);
trivial!(u64, U64, arrow_dtypes::UInt64Type);

extra_arrow!(f16, arrow_dtypes::Float16Type);
extra_arrow!(i64, arrow_dtypes::TimestampSecondType);
extra_arrow!(i64, arrow_dtypes::TimestampMillisecondType);
extra_arrow!(i64, arrow_dtypes::TimestampMicrosecondType);
extra_arrow!(i64, arrow_dtypes::TimestampNanosecondType);
extra_arrow!(i32, arrow_dtypes::Date32Type);
extra_arrow!(i64, arrow_dtypes::Date64Type);

pub fn from_arrow(arrow_dtype: &ArrowDataType) -> Result<NumberType> {
  let res = match arrow_dtype {
    ArrowDataType::Float16 => NumberType::F16,
    ArrowDataType::Float32 => NumberType::F32,
    ArrowDataType::Float64 => NumberType::F64,
    ArrowDataType::Int16 => NumberType::I16,
    ArrowDataType::Int32 => NumberType::I32,
    ArrowDataType::Int64 => NumberType::I64,
    ArrowDataType::UInt16 => NumberType::U16,
    ArrowDataType::UInt32 => NumberType::U32,
    ArrowDataType::UInt64 => NumberType::U64,
    ArrowDataType::Timestamp(_, _) => NumberType::I64,
    ArrowDataType::Date32 => NumberType::I32,
    ArrowDataType::Date64 => NumberType::I64,
    _ => {
      return Err(anyhow!(
        "unable to convert arrow dtype {:?} to pco",
        arrow_dtype
      ))
    }
  };
  Ok(res)
}

pub fn to_arrow(dtype: NumberType) -> ArrowDataType {
  match dtype {
    NumberType::F16 => ArrowDataType::Float16,
    NumberType::F32 => ArrowDataType::Float32,
    NumberType::F64 => ArrowDataType::Float64,
    NumberType::I16 => ArrowDataType::Int16,
    NumberType::I32 => ArrowDataType::Int32,
    NumberType::I64 => ArrowDataType::Int64,
    NumberType::U16 => ArrowDataType::UInt16,
    NumberType::U32 => ArrowDataType::UInt32,
    NumberType::U64 => ArrowDataType::UInt64,
    other => panic!(
      "number type {:?} not yet supported in pco_cli",
      other
    ),
  }
}
