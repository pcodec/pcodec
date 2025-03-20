use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::fs;
use std::mem;
use std::ops::Deref;
use std::str::FromStr;
use std::time::{Duration, Instant};

#[cfg(feature = "full_bench")]
use crate::bench::codecs::blosc::BloscConfig;
#[cfg(feature = "full_bench")]
use crate::bench::codecs::brotli::BrotliConfig;
use crate::bench::codecs::parquet::ParquetConfig;
#[cfg(feature = "full_bench")]
use crate::bench::codecs::qco::QcoConfig;
use crate::bench::codecs::snappy::SnappyConfig;
#[cfg(feature = "full_bench")]
use crate::bench::codecs::spdp::SpdpConfig;
#[cfg(feature = "full_bench")]
use crate::bench::codecs::turbo_pfor::TurboPforConfig;
use crate::bench::codecs::zstd::ZstdConfig;
use crate::bench::IterOpt;
use crate::bench::{BenchStat, Precomputed};
use crate::chunk_config_opt::ChunkConfigOpt;
use crate::dtypes::PcoNumber;
use crate::num_vec::NumVec;
use ::pco::data_types::NumberType;
use ::pco::match_number_enum;
use anyhow::{anyhow, Context, Result};
use clap::{CommandFactory, FromArgMatches};

#[cfg(feature = "full_bench")]
mod blosc;
#[cfg(feature = "full_bench")]
mod brotli;
mod parquet;
mod pco;
#[cfg(feature = "full_bench")]
mod qco;
mod snappy;
#[cfg(feature = "full_bench")]
mod spdp;
#[cfg(feature = "full_bench")]
mod turbo_pfor;
pub mod utils;
mod zstd;

// Unfortunately we can't make a Box<dyn this> because it has generic
// functions, so we use a wrapping trait (CodecSurface) to manually dynamic
// dispatch.
trait CodecInternal: Clone + CommandFactory + Debug + FromArgMatches + Send + Sync + 'static {
  fn name(&self) -> &'static str;
  fn get_confs(&self) -> Vec<(&'static str, String)>;

  fn compress<T: PcoNumber>(&self, nums: &[T]) -> Vec<u8>;
  fn decompress<T: PcoNumber>(&self, compressed: &[u8]) -> Vec<T>;

  fn compress_dynamic(&self, num_vec: &NumVec) -> Vec<u8> {
    match_number_enum!(
      num_vec,
      NumVec<T>(nums) => { self.compress(nums) }
    )
  }

  fn decompress_dynamic(&self, dtype: NumberType, compressed: &[u8]) -> NumVec {
    match_number_enum!(
      dtype,
      NumberType<T> => {
        NumVec::new(self.decompress::<T>(compressed)).unwrap()
      }
    )
  }
}

pub trait CodecSurface: Debug + Send + Sync {
  fn name(&self) -> &'static str;
  fn from_kv_args(kv_args: &[String]) -> Result<Box<dyn CodecSurface>>
  where
    Self: Sized;
  fn details(&self, explicit: bool) -> String;

  fn warmup_iter(
    &self,
    nums_vec: &NumVec,
    dataset: &str,
    opt: &IterOpt,
    thread_idx: usize,
  ) -> Result<Precomputed>;
  fn stats_iter(
    &self,
    nums_vec: &NumVec,
    precomputed: &Precomputed,
    opt: &IterOpt,
  ) -> Result<BenchStat>;

  fn clone_to_box(&self) -> Box<dyn CodecSurface>;
}

fn default_codec<C: CodecInternal>() -> C {
  let empty_args = Vec::<String>::new();
  let mut default_arg_matches = <C as CommandFactory>::command().get_matches_from(empty_args);
  <C as FromArgMatches>::from_arg_matches_mut(&mut default_arg_matches).unwrap()
}

impl<C: CodecInternal> CodecSurface for C {
  fn name(&self) -> &'static str {
    self.name()
  }

  fn from_kv_args(kv_args: &[String]) -> Result<Box<dyn CodecSurface>>
  where
    Self: Sized,
  {
    let mut matches = Self::command()
      .try_get_matches_from(kv_args)
      .map_err(|_| {
        let codec = default_codec::<Self>();
        let help_string =
          Self::command().render_help().to_string();
        let options_start = help_string.find("Options:").unwrap_or_default();
        anyhow!(
          "Configurations for {} codec not understood. As an example, the default configuration is \"{}{}\".\n\n{}",
          codec.name(),
          codec.name(),
          codec.details(true),
          &help_string[options_start..]
        )
      })?;
    let codec = Self::from_arg_matches_mut(&mut matches)?;
    Ok(Box::new(codec))
  }

  fn details(&self, explicit: bool) -> String {
    // use derived clap defaults
    let default = default_codec::<Self>();
    let default_confs: HashMap<&'static str, String> = default.get_confs().into_iter().collect();
    let mut res = String::new();
    for (k, v) in self.get_confs() {
      if explicit || &v != default_confs.get(k).unwrap() {
        res.push_str(&format!(":{}={}", k, v,));
      }
    }
    res
  }

  fn warmup_iter(
    &self,
    num_vec: &NumVec,
    dataset: &str,
    opt: &IterOpt,
    thread_idx: usize,
  ) -> Result<Precomputed> {
    let dtype = num_vec.dtype();

    // compress
    let compressed = self.compress_dynamic(num_vec);

    // write to disk
    if let Some(dir) = opt.save_dir.as_ref() {
      if thread_idx == 0 {
        let save_path = dir.join(format!(
          "{}{}.{}",
          &dataset,
          self.details(false),
          self.name(),
        ));
        fs::write(save_path, &compressed)?;
      }
    }

    // decompress
    if !opt.no_decompress {
      let rec_nums = self.decompress_dynamic(dtype, &compressed);

      if !opt.no_assertions {
        rec_nums.check_equal(num_vec).with_context(|| {
          format!(
            "on codec {}{}, dataset {}",
            self.name(),
            self.details(false),
            dataset
          )
        })?;
      }
    }

    Ok(Precomputed { compressed })
  }

  fn stats_iter(
    &self,
    num_vec: &NumVec,
    precomputed: &Precomputed,
    opt: &IterOpt,
  ) -> Result<BenchStat> {
    // compress
    let compress_dt = if !opt.no_compress {
      let t = Instant::now();
      let _ = self.compress_dynamic(num_vec);
      Instant::now() - t
    } else {
      Duration::ZERO
    };

    // decompress
    let decompress_dt = if !opt.no_decompress {
      let t = Instant::now();
      let _ = self.decompress_dynamic(num_vec.dtype(), &precomputed.compressed);
      Instant::now() - t
    } else {
      Duration::ZERO
    };

    let uncompressed_size = match_number_enum!(
      num_vec,
      NumVec<T>(inner) => {
        inner.len() * mem::size_of::<T>()
      }
    );

    Ok(BenchStat {
      compressed_size: precomputed.compressed.len(),
      uncompressed_size,
      compress_dt,
      decompress_dt,
    })
  }

  fn clone_to_box(&self) -> Box<dyn CodecSurface> {
    Box::new(self.clone())
  }
}

#[derive(Debug)]
pub struct CodecConfig(Box<dyn CodecSurface>);

impl FromStr for CodecConfig {
  type Err = anyhow::Error;

  fn from_str(s: &str) -> Result<Self> {
    let parts = s.split(':').collect::<Vec<_>>();
    let name = parts[0];
    // first argument gets ignored
    let mut clap_kv_args = vec!["".to_string()];
    for &part in &parts[1..] {
      let kv_vec = part.split('=').collect::<Vec<_>>();
      if kv_vec.len() != 2 {
        return Err(anyhow!(
          "codec config {} is not a key=value pair",
          part
        ));
      }
      clap_kv_args.push(format!("--{}={}", kv_vec[0], kv_vec[1]));
    }

    let codec: Result<Box<dyn CodecSurface>> = match name {
      #[cfg(feature = "full_bench")]
      "blosc" | "blosc2" => BloscConfig::from_kv_args(&clap_kv_args),
      #[cfg(feature = "full_bench")]
      "brotli" => BrotliConfig::from_kv_args(&clap_kv_args),
      "parquet" => ParquetConfig::from_kv_args(&clap_kv_args),
      "pco" | "pcodec" => ChunkConfigOpt::from_kv_args(&clap_kv_args),
      #[cfg(feature = "full_bench")]
      "qco" | "q_compress" => QcoConfig::from_kv_args(&clap_kv_args),
      "snap" | "snappy" => SnappyConfig::from_kv_args(&clap_kv_args),
      #[cfg(feature = "full_bench")]
      "spdp" => SpdpConfig::from_kv_args(&clap_kv_args),
      #[cfg(feature = "full_bench")]
      "tpfor" | "turbopfor" => TurboPforConfig::from_kv_args(&clap_kv_args),
      "zstd" | "zstandard" => ZstdConfig::from_kv_args(&clap_kv_args),
      _ => {
        return Err(anyhow!(
          "Unknown codec: {}. Perhaps rebuild with the full_bench feature?",
          name
        ))
      }
    };

    Ok(Self(codec?))
  }
}

impl Display for CodecConfig {
  fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
    write!(
      f,
      "{}{}",
      self.0.name(),
      self.0.details(false),
    )
  }
}

impl Clone for CodecConfig {
  fn clone(&self) -> Self {
    Self(self.0.clone_to_box())
  }
}

impl Deref for CodecConfig {
  type Target = Box<dyn CodecSurface>;

  fn deref(&self) -> &Self::Target {
    &self.0
  }
}
