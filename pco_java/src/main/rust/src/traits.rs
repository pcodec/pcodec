use crate::result::Result;
use half::f16;
use jni::objects::{JPrimitiveArray, TypeArray};
use jni::JNIEnv;
use pco::data_types::Number;
use std::mem;

pub trait JavaConversions: Number {
  type Java: TypeArray;
  fn get_region(env: &JNIEnv, src: &JPrimitiveArray<Self::Java>, dst: &mut [Self]) -> Result<()>;

  fn new_array<'a>(env: &JNIEnv<'a>, len: i32) -> Result<JPrimitiveArray<'a, Self::Java>>;
  fn set_region(env: &JNIEnv, src: &[Self], dst: &mut JPrimitiveArray<Self::Java>) -> Result<()>;
}

macro_rules! trivial_impl {
  ($t:ty, $get:ident, $new:ident, $set:ident) => {
    impl JavaConversions for $t {
      type Java = Self;
      fn get_region(env: &JNIEnv, src: &JPrimitiveArray<Self>, dst: &mut [Self]) -> Result<()> {
        env.$get(src, 0, dst)?;
        Ok(())
      }

      fn new_array<'a>(env: &JNIEnv<'a>, len: i32) -> Result<JPrimitiveArray<'a, Self::Java>> {
        Ok(env.$new(len)?)
      }
      fn set_region(env: &JNIEnv, src: &[Self], dst: &mut JPrimitiveArray<Self>) -> Result<()> {
        env.$set(dst, 0, src)?;
        Ok(())
      }
    }
  };
}

macro_rules! transmute_impl {
  ($t:ty, $j:ty) => {
    impl JavaConversions for $t {
      type Java = $j;
      fn get_region(
        env: &JNIEnv,
        src: &JPrimitiveArray<Self::Java>,
        dst: &mut [Self],
      ) -> Result<()> {
        let dst = unsafe { mem::transmute::<&mut [Self], &mut [Self::Java]>(dst) };
        Self::Java::get_region(env, src, dst)
      }

      fn new_array<'a>(env: &JNIEnv<'a>, len: i32) -> Result<JPrimitiveArray<'a, Self::Java>> {
        Self::Java::new_array(env, len)
      }
      fn set_region(
        env: &JNIEnv,
        src: &[Self],
        dst: &mut JPrimitiveArray<Self::Java>,
      ) -> Result<()> {
        let src = unsafe { mem::transmute::<&[Self], &[Self::Java]>(src) };
        Self::Java::set_region(env, src, dst)
      }
    }
  };
}

trivial_impl!(
  i16,
  get_short_array_region,
  new_short_array,
  set_short_array_region
);
trivial_impl!(
  i32,
  get_int_array_region,
  new_int_array,
  set_int_array_region
);
trivial_impl!(
  i64,
  get_long_array_region,
  new_long_array,
  set_long_array_region
);
trivial_impl!(
  f32,
  get_float_array_region,
  new_float_array,
  set_float_array_region
);
trivial_impl!(
  f64,
  get_double_array_region,
  new_double_array,
  set_double_array_region
);

transmute_impl!(f16, i16);
transmute_impl!(u16, i16);
transmute_impl!(u32, i32);
transmute_impl!(u64, i64);
