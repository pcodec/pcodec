use crate::result::Result;
use jni::{
  objects::{JObject, JValueOwned},
  JNIEnv,
};
use pco::ChunkConfig;

pub fn from_java(env: &mut JNIEnv, j_chunk_config: JObject) -> Result<ChunkConfig> {
  let JValueOwned::Int(compression_level) =
    env.get_field(&j_chunk_config, "compressionLevel", "I")?
  else {
    unreachable!();
  };
  Ok(ChunkConfig::default().with_compression_level(compression_level as usize))
}
