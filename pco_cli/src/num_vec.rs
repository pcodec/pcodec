use std::mem;

use pco::data_types::CoreDataType;
use pco::with_core_dtypes;

use crate::dtypes::PcoNumberLike;

fn cast_to_nums<T: PcoNumberLike>(bytes: Vec<u8>) -> Vec<T> {
  // Here we're assuming the bytes are in the right format for our data type.
  // For instance, chunks of 8 little-endian bytes on most platforms for
  // i64's.
  // This is fast and should work across platforms.
  let n = bytes.len() / mem::size_of::<T>();
  unsafe {
    let mut nums = mem::transmute::<_, Vec<T>>(bytes);
    nums.set_len(n);
    nums
  }
}

fn check_equal<T: PcoNumberLike>(recovered: &[T], original: &[T]) {
  assert_eq!(recovered.len(), original.len());
  for (i, (x, y)) in recovered.iter().zip(original.iter()).enumerate() {
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

macro_rules! impl_num_vec {
  {$($name:ident($lname:ident) => $t:ty,)+} => {
    pub enum NumVec {
      $($name(Vec<$t>),)+
    }

    impl NumVec {
      pub fn new(dtype: CoreDataType, raw_bytes: Vec<u8>) -> Self {
        match dtype {
          $(CoreDataType::$name => NumVec::$name(cast_to_nums(raw_bytes)),)+
        }
      }

      pub fn n(&self) -> usize {
        match self {
          $(NumVec::$name(nums) => nums.len(),)+
        }
      }

      pub fn truncated(&self, limit: usize) -> Self {
        match self {
          $(NumVec::$name(nums) => NumVec::$name(nums[..limit].to_vec()),)+
        }
      }

      pub fn dtype(&self) -> CoreDataType {
        match self {
          $(NumVec::$name(_) => CoreDataType::$name,)+
        }
      }

      pub fn check_equal(&self, other: &NumVec) {
        match (self, other) {
          $((NumVec::$name(x), NumVec::$name(y)) => check_equal(x, y),)+
          _ => unreachable!(),
        }
      }
    }
  };
}

with_core_dtypes!(impl_num_vec);
