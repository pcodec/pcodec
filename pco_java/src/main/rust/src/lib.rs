mod result;
mod traits;

use crate::result::{Exception, ExceptionKind, Result};
use crate::traits::JavaConversions;
use jni::objects::{JClass, JObject, JPrimitiveArray, JValueGen, JValueOwned};
use jni::sys::*;
use jni::JNIEnv;
use pco::data_types::{Number, NumberType};
use pco::match_number_enum;
use pco::standalone::{FileDecompressor, MaybeChunkDecompressor};

fn handle_result(env: &mut JNIEnv, result: Result<jobject>) -> jobject {
  // We need a function that creates a fake instance of the return type, due
  // to unwinding issues:
  // https://github.com/jni-rs/jni-rs/issues/76
  match result {
    Ok(inner) => inner,
    Err(e) => {
      let descriptor = match e.kind {
        ExceptionKind::InvalidArgument => "java/lang/IllegalArgumentException",
        // probably not reachable since FFI only supports in-memory data
        ExceptionKind::Io => "java/io/IOException",
        ExceptionKind::Runtime => "java/lang/RuntimeException",
      };
      match env.throw_new(descriptor, &e.msg) {
          Ok(()) => (),
          Err(e) => eprintln!("Error when trying to raise Java exception. This is likely a bug in the pco java bindings: {}", e),
      };
      *JObject::null()
    }
  }
}

const NUM_ARRAY: &'static str = "Lio/github/pcodec/NumArray;";

fn simpler_compress_inner<'a>(
  env: &mut JNIEnv<'a>,
  num_array: jobject,
  level: jint,
) -> Result<jbyteArray> {
  let num_array = unsafe { JObject::from_raw(num_array) };
  let JValueOwned::Object(src) = env.get_field(&num_array, "nums", "Ljava/lang/Object;")? else {
    unreachable!();
  };
  let JValueOwned::Int(number_type_int) = env.get_field(&num_array, "numberTypeByte", "I")? else {
    unreachable!();
  };
  let number_type = NumberType::from_descriminant(number_type_int as u8).unwrap();

  let compressed = match_number_enum!(number_type, NumberType<T> => {
      let src = JPrimitiveArray::from(src);
      let src_len = env.get_array_length(&src)? as usize;
      let mut nums = Vec::with_capacity(src_len);
      unsafe {
          nums.set_len(src_len);
      }
      T::get_region(env, &src, &mut nums)?;
      // TODO is there a way to avoid copying here?
      pco::standalone::simpler_compress(&nums, level as usize)?
  });

  let compressed = env.byte_array_from_slice(&compressed)?;
  Ok(compressed.into_raw())
}

fn decompress_chunks<'a, T: Number + JavaConversions>(
  env: &mut JNIEnv<'a>,
  mut src: &[u8],
  file_decompressor: FileDecompressor,
) -> Result<jobject> {
  let n_hint = file_decompressor.n_hint();
  let mut res: Vec<T> = Vec::with_capacity(n_hint);
  while let MaybeChunkDecompressor::Some(mut chunk_decompressor) =
    file_decompressor.chunk_decompressor::<T, &[u8]>(src)?
  {
    let initial_len = res.len(); // probably always zero to start, since we just created res
    let remaining = chunk_decompressor.n();
    unsafe {
      res.set_len(initial_len + remaining);
    }
    let progress = chunk_decompressor.decompress(&mut res[initial_len..])?;
    assert!(progress.finished);
    src = chunk_decompressor.into_src();
  }
  let mut array = T::new_array(env, res.len() as i32)?;
  T::set_region(env, &res, &mut array)?;
  let num_array = env.new_object(
    NUM_ARRAY,
    "(Ljava/lang/Object;I)V",
    &[
      JValueGen::Object(&*array),
      JValueGen::Int(T::NUMBER_TYPE_BYTE as i32),
    ],
  )?;
  let optional = env.call_static_method(
    "Ljava/util/Optional;",
    "of",
    "(Ljava/lang/Object;)Ljava/util/Optional;",
    &[JValueGen::Object(&num_array)],
  )?;
  let JValueGen::Object(optional) = optional else {
    unreachable!()
  };
  Ok(optional.as_raw())
}

fn java_none<'a>(env: &mut JNIEnv<'a>) -> Result<jobject> {
  let optional = env.call_static_method("Ljava/util/Optional;", "empty", "", &[])?;
  let JValueGen::Object(optional) = optional else {
    unreachable!()
  };
  Ok(optional.as_raw())
}

fn simple_decompress_inner<'a>(env: &mut JNIEnv<'a>, src: jbyteArray) -> Result<jobject> {
  let src = unsafe { JPrimitiveArray::from_raw(src) };
  let src = env.convert_byte_array(src)?;
  let (file_decompressor, rest) = FileDecompressor::new(src.as_slice())?;
  let maybe_number_type = file_decompressor.peek_number_type_or_termination(&rest)?;

  use pco::standalone::NumberTypeOrTermination::*;
  match maybe_number_type {
    Known(number_type) => {
      match_number_enum!(
          number_type,
          NumberType<T> => {
              decompress_chunks::<T>(env, rest, file_decompressor)
          }
      )
    }
    Termination => java_none(env),
    Unknown(other) => Err(Exception {
      kind: ExceptionKind::Runtime,
      msg: format!(
        "unrecognized pco number type byte {:?}",
        other,
      ),
    }),
  }
}

#[no_mangle]
pub extern "system" fn Java_io_github_pcodec_Standalone_simpler_1compress<'a>(
  mut env: JNIEnv<'a>,
  _: JClass<'a>,
  num_array: jobject,
  level: jint,
) -> jbyteArray {
  let result = simpler_compress_inner(&mut env, num_array, level);
  handle_result(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_io_github_pcodec_Standalone_simple_1decompress<'a>(
  mut env: JNIEnv<'a>,
  _: JClass<'a>,
  src: jbyteArray,
) -> jobject {
  let result = simple_decompress_inner(&mut env, src);
  handle_result(&mut env, result)
}
