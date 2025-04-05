use jni::{
  objects::{JObject, JValueGen, JValueOwned},
  JNIEnv,
};
use pco::data_types::{Number, NumberType};

use crate::{result::Result, traits::JavaConversions};

const TYPE_SIGNATURE: &str = "Lio/github/pcodec/NumArray;";

pub fn from_java<'a>(
  env: &mut JNIEnv<'a>,
  j_num_array: JObject,
) -> Result<(JObject<'a>, NumberType)> {
  let JValueOwned::Object(src) = env.get_field(&j_num_array, "nums", "Ljava/lang/Object;")? else {
    unreachable!();
  };
  let JValueOwned::Byte(number_type_i8) = env.get_field(&j_num_array, "numberTypeByte", "B")?
  else {
    unreachable!();
  };
  let number_type = NumberType::from_descriminant(number_type_i8 as u8).unwrap();
  Ok((src, number_type))
}

pub fn to_java<'a, T: Number + JavaConversions>(
  env: &mut JNIEnv<'a>,
  nums: &[T],
) -> Result<JObject<'a>> {
  let mut array = T::new_array(env, nums.len() as i32)?;
  T::set_region(env, nums, &mut array)?;
  let num_array = env.new_object(
    TYPE_SIGNATURE,
    "(Ljava/lang/Object;B)V",
    &[
      JValueGen::Object(&*array),
      JValueGen::Byte(T::NUMBER_TYPE_BYTE as i8),
    ],
  )?;
  Ok(num_array)
}
