use jni::{JNIEnv};
use jni::errors::{Error, Result};
use jni::objects::{JClass, JPrimitiveArray};
use jni::sys::*;
use pco::errors::PcoError;

fn pco_err_to_jni_err(err: PcoError) -> Error {
    panic!()
}

fn compress_inner<'a>(
    env: JNIEnv<'a>,
    src: jlongArray,
) -> Result<jbyteArray> {
    // TODO copy less
    let src = unsafe { JPrimitiveArray::from_raw(src) };
    let src_len = env.get_array_length(&src)? as usize;
    let mut nums = vec![0; src_len];
    env.get_long_array_region(&src, 0, &mut nums)?;
    let compressed = pco::standalone::simpler_compress(&nums, 8).map_err(pco_err_to_jni_err)?;
    let compressed = env.byte_array_from_slice(&compressed)?;
    Ok(compressed.into_raw())
}

#[no_mangle]
pub extern "system" fn Java_org_pcodec_Native_simpler_1compress_1i64<'a>(
    env: JNIEnv<'a>,
    _: JClass<'a>,
    src: jlongArray,
) -> jbyteArray {
    compress_inner(env, src).unwrap()
}
