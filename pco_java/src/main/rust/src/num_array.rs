use jni::{
  objects::{JObject, JValueGen, JValueOwned},
  JNIEnv,
};
use pco::data_types::{Number, NumberType};

use crate::{result::Result, traits::JavaConversions};

const NUM_ARRAY: &'static str = "Lio/github/pcodec/NumArray;";

pub fn from_java<'a>(
  env: &mut JNIEnv<'a>,
  j_num_array: JObject,
) -> Result<(JObject<'a>, NumberType)> {
  let JValueOwned::Object(src) = env.get_field(&j_num_array, "nums", "Ljava/lang/Object;")? else {
    unreachable!();
  };
  let JValueOwned::Int(number_type_int) = env.get_field(&j_num_array, "numberTypeByte", "I")?
  else {
    unreachable!();
  };
  let number_type = NumberType::from_descriminant(number_type_int as u8).unwrap();
  Ok((src, number_type))
}

pub fn to_java<'a, T: Number + JavaConversions>(
  env: &mut JNIEnv<'a>,
  nums: &[T],
) -> Result<JObject<'a>> {
  let mut array = T::new_array(env, nums.len() as i32)?;
  T::set_region(env, &nums, &mut array)?;
  let num_array = env.new_object(
    NUM_ARRAY,
    "(Ljava/lang/Object;I)V",
    &[
      JValueGen::Object(&*array),
      JValueGen::Int(T::NUMBER_TYPE_BYTE as i32),
    ],
  )?;
  Ok(num_array)
}
