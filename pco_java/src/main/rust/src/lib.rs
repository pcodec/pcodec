use jni::{JNIEnv};
use jni::errors::{Error as JniError};
use jni::objects::{JClass, JObject,JPrimitiveArray};
use jni::sys::*;
use pco::errors::{PcoError, ErrorKind as PcoErrorKind};

#[derive(Clone, Debug)]
enum ExceptionKind {
    InvalidArgument,
    Io,
    Runtime,
}

#[derive(Clone, Debug)]
struct Exception {
    kind: ExceptionKind,
    msg: String,
}

type Result<T> = std::result::Result<T, Exception>;

impl From<PcoError> for Exception {
    fn from(value: PcoError) -> Self {
        let msg = format!("{}", value);
        let kind = match value.kind {
            PcoErrorKind::Io(_) => ExceptionKind::Io,
            PcoErrorKind::InvalidArgument => ExceptionKind::InvalidArgument,
            _ => ExceptionKind::Runtime,
        };
        Exception { kind, msg }
    }
}

impl From<JniError> for Exception {
    fn from(value: JniError) -> Self {
        Exception { kind: ExceptionKind::Runtime, msg: format!("{}", value) }
    }
}

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
            env.throw_new(descriptor, &e.msg).unwrap();
            *JObject::null()
        }
    }
}

fn compress_inner<'a>(
    env: &JNIEnv<'a>,
    src: jlongArray,
    level: jint,
) -> Result<jbyteArray> {
    // TODO copy less
    let src = unsafe { JPrimitiveArray::from_raw(src) };
    let src_len = env.get_array_length(&src)? as usize;
    let mut nums = vec![0; src_len];
    env.get_long_array_region(&src, 0, &mut nums)?;
    let compressed = pco::standalone::simpler_compress(&nums, level as usize)?;
    let compressed = env.byte_array_from_slice(&compressed)?;
    Ok(compressed.into_raw())
}

#[no_mangle]
pub extern "system" fn Java_org_pcodec_Native_simpler_1compress_1i64<'a>(
    mut env: JNIEnv<'a>,
    _: JClass<'a>,
    src: jlongArray,
    level: jint,
) -> jbyteArray {
    let result = compress_inner(&env, src, level);
    handle_result(&mut env, result)
}
