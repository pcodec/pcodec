#![allow(clippy::uninit_vec)]

mod config;
mod num_array;
mod result;
mod traits;

use crate::result::{Exception, ExceptionKind, Result};
use crate::traits::JavaConversions;
use jni::objects::{JClass, JObject, JPrimitiveArray, JValueGen};
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

fn simple_compress_inner(
  env: &mut JNIEnv,
  j_num_array: jobject,
  j_chunk_config: jobject,
) -> Result<jbyteArray> {
  let (j_num_array, j_chunk_config) = unsafe {
    (
      JObject::from_raw(j_num_array),
      JObject::from_raw(j_chunk_config),
    )
  };
  let (j_src, number_type) = num_array::from_java(env, j_num_array)?;
  let chunk_config = config::from_java(env, j_chunk_config)?;

  let compressed = match_number_enum!(number_type, NumberType<T> => {
      let j_src = JPrimitiveArray::from(j_src);
      let len = env.get_array_length(&j_src)? as usize;
      let mut nums = Vec::with_capacity(len);
      unsafe {
          nums.set_len(len);
      }
      T::get_region(env, &j_src, &mut nums)?;
      // TODO is there a way to avoid copying here?
      pco::standalone::simple_compress(&nums, &chunk_config)?
  });

  let compressed = env.byte_array_from_slice(&compressed)?;
  Ok(compressed.into_raw())
}

fn decompress_chunks<T: Number + JavaConversions>(
  env: &mut JNIEnv,
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
  let num_array = num_array::to_java(env, &res)?;
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

fn java_none(env: &mut JNIEnv) -> Result<jobject> {
  let optional = env.call_static_method("Ljava/util/Optional;", "empty", "", &[])?;
  let JValueGen::Object(optional) = optional else {
    unreachable!()
  };
  Ok(optional.as_raw())
}

fn simple_decompress_inner(env: &mut JNIEnv, src: jbyteArray) -> Result<jobject> {
  let src = unsafe { JPrimitiveArray::from_raw(src) };
  let src = env.convert_byte_array(src)?;
  let (file_decompressor, rest) = FileDecompressor::new(src.as_slice())?;
  let maybe_number_type = file_decompressor.peek_number_type_or_termination(rest)?;

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
pub extern "system" fn Java_io_github_pcodec_Standalone_simple_1compress<'a>(
  mut env: JNIEnv<'a>,
  _: JClass<'a>,
  j_num_array: jobject,
  j_chunk_config: jobject,
) -> jbyteArray {
  let result = simple_compress_inner(&mut env, j_num_array, j_chunk_config);
  handle_result(&mut env, result)
}

#[no_mangle]
pub extern "system" fn Java_io_github_pcodec_Standalone_simple_1decompress<'a>(
  mut env: JNIEnv<'a>,
  _: JClass<'a>,
  j_src: jbyteArray,
) -> jobject {
  let result = simple_decompress_inner(&mut env, j_src);
  handle_result(&mut env, result)
}
