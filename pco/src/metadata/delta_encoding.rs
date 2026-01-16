use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::constants::*;
use crate::data_types::number_priv::NumberPriv;
use crate::data_types::signed::Signed;
use crate::data_types::LatentType;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::format_version::FormatVersion;
use crate::metadata::per_latent_var::LatentVarKey;
use std::io::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeltaLookbackConfig {
  /// The log2 of the number of latents explicitly stored in page metadata
  /// to prepopulate the lookback window.
  pub state_n_log: Bitlen,
  /// The log2 of the maximum possible lookback.
  pub window_n_log: Bitlen,
}

impl DeltaLookbackConfig {
  pub(crate) fn state_n(&self) -> usize {
    1 << self.state_n_log
  }

  pub(crate) fn window_n(&self) -> usize {
    1 << self.window_n_log
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeltaConv1Config {
  pub quantization: Bitlen,
  // Avoiding exposing bias and weights because I think it's possible we'll
  // change their representation in the future; for now users will have to
  // satisfy themselves with the Debug string if they are curious
  bias: i64,
  weights: Vec<i64>,
}

impl DeltaConv1Config {
  pub(crate) fn new(quantization: Bitlen, bias: i64, weights: Vec<i64>) -> Self {
    Self {
      quantization,
      bias,
      weights,
    }
  }

  pub(crate) fn bias<S: Signed>(&self) -> S {
    S::from_i64(self.bias)
  }

  pub(crate) fn weights<S: Signed>(&self) -> Vec<S> {
    self.weights.iter().cloned().map(S::from_i64).collect()
  }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DeltaConv2Config {
  pub quantization: Bitlen,
  // 2D width.
  pub w: usize,
  // Kernel size in height dimension.
  pub kh: usize,
  // Kernel size in width dimension.
  pub kw: usize,
  // Avoiding exposing parameters for the same reason as Conv1
  bias: i64,
  weights: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum LatentVarDeltaEncoding {
  NoOp,
  Consecutive(usize),
  Lookback(DeltaLookbackConfig),
  Conv1(DeltaConv1Config),
  Conv2(DeltaConv2Config),
}

impl LatentVarDeltaEncoding {
  pub(crate) fn n_latents_per_state(&self) -> usize {
    match self {
      Self::NoOp => 0,
      Self::Consecutive(order) => *order,
      Self::Lookback(config) => 1 << config.state_n_log,
      Self::Conv1(config) => config.weights.len(),
      Self::Conv2(config) => config.kw - 1,
    }
  }
}

/// How Pco did
/// [delta encoding](https://en.wikipedia.org/wiki/Delta_encoding) on a chunk.
///
/// Delta encoding optionally takes differences between nearby numbers,
/// greatly reducing the entropy of the data distribution in some cases.
/// This stage of processing happens after applying the
/// [`Mode`][crate::metadata::Mode] during compression.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeltaEncoding {
  /// No delta encoding; the values are encoded as-is.
  ///
  /// This is best if your data is in random order.
  NoOp,
  /// Encodes the differences between consecutive values (or differences
  /// between those, etc.).
  ///
  /// This is best if your numbers have high variance overall, but adjacent
  /// numbers are close in value, e.g. an arithmetic sequence.
  Consecutive {
    order: usize,
    secondary_uses_delta: bool,
  },
  /// Encodes an extra "lookback" latent variable and the differences
  /// `x[i] - x[i - lookback[i]]` between values.
  ///
  /// This is best if your numbers have complex repeating patterns
  /// beyond just adjacent elements.
  /// It is in spirit similar to LZ77 compression, but only stores lookbacks
  /// (AKA match offsets) and no match lengths.
  Lookback {
    config: DeltaLookbackConfig,
    secondary_uses_delta: bool,
  },
  /// Encodes the difference between each value and a convolution of the
  /// preceding few elements.
  ///
  /// This is best if your numbers have local trends that aren't captured by
  /// simply taking differences.
  Conv1(DeltaConv1Config),
  /// Encodes the difference between each value and a convolution of nearby
  /// elements in 2D space.
  ///
  /// This is best for data like images.
  Conv2(DeltaConv2Config),
}

impl DeltaEncoding {
  pub(crate) const MAX_BIT_SIZE: usize = (BITS_TO_ENCODE_DELTA_ENCODING_VARIANT
    + BITS_TO_ENCODE_DELTA_CONV_QUANTIZATION
    + BITS_TO_ENCODE_DELTA_CONV_N_WEIGHTS) as usize
    + 64
    + MAX_CONV1_DELTA_ORDER * 32;

  unsafe fn read_from_pre_v3(reader: &mut BitReader) -> Self {
    let order = reader.read_usize(BITS_TO_ENCODE_DELTA_ENCODING_ORDER);
    match order {
      0 => Self::NoOp,
      _ => Self::Consecutive {
        order,
        secondary_uses_delta: false,
      },
    }
  }

  pub(crate) unsafe fn read_from(
    reader: &mut BitReader,
    version: &FormatVersion,
  ) -> PcoResult<Self> {
    if !version.supports_delta_variants() {
      return Ok(Self::read_from_pre_v3(reader));
    }

    let delta_encoding_variant = reader.read_bitlen(BITS_TO_ENCODE_DELTA_ENCODING_VARIANT);

    let res = match delta_encoding_variant {
      0 => Self::NoOp,
      1 => {
        let order = reader.read_usize(BITS_TO_ENCODE_DELTA_ENCODING_ORDER);
        if order == 0 {
          return Err(PcoError::corruption(
            "Consecutive delta encoding order must not be 0",
          ));
        } else {
          Self::Consecutive {
            order,
            secondary_uses_delta: reader.read_bool(),
          }
        }
      }
      2 => {
        let window_n_log = 1 + reader.read_bitlen(BITS_TO_ENCODE_DELTA_LOOKBACK_WINDOW_N_LOG);
        let state_n_log = reader.read_bitlen(BITS_TO_ENCODE_DELTA_LOOKBACK_STATE_N_LOG);
        if state_n_log > window_n_log {
          return Err(PcoError::corruption(format!(
            "LZ delta encoding state size log exceeded window size log: {} vs {}",
            state_n_log, window_n_log
          )));
        }
        Self::Lookback {
          config: DeltaLookbackConfig {
            window_n_log,
            state_n_log,
          },
          secondary_uses_delta: reader.read_bool(),
        }
      }
      3 => {
        let quantization = reader.read_bitlen(BITS_TO_ENCODE_DELTA_CONV_QUANTIZATION);
        let bias = i64::from_latent_ordered(reader.read_uint(64));
        let order = 1 + reader.read_usize(BITS_TO_ENCODE_DELTA_CONV_N_WEIGHTS);
        let mut weights = Vec::with_capacity(order);
        for _ in 0..order {
          weights.push(i32::from_latent_ordered(reader.read_uint(32)) as i64);
        }

        Self::Conv1(DeltaConv1Config {
          quantization,
          bias,
          weights,
        })
      }
      value => {
        return Err(PcoError::corruption(format!(
          "unknown delta encoding value: {}",
          value
        )))
      }
    };
    Ok(res)
  }

  pub(crate) unsafe fn write_to<W: Write>(&self, writer: &mut BitWriter<W>) {
    let variant = match self {
      Self::NoOp => 0,
      Self::Consecutive { .. } => 1,
      Self::Lookback { .. } => 2,
      Self::Conv1(_) => 3,
      Self::Conv2(_) => 4,
    };
    writer.write_bitlen(
      variant,
      BITS_TO_ENCODE_DELTA_ENCODING_VARIANT,
    );

    match self {
      Self::NoOp => (),
      Self::Consecutive {
        order,
        secondary_uses_delta,
      } => {
        writer.write_usize(*order, BITS_TO_ENCODE_DELTA_ENCODING_ORDER);
        writer.write_bool(*secondary_uses_delta);
      }
      Self::Lookback {
        config,
        secondary_uses_delta,
      } => {
        writer.write_bitlen(
          config.window_n_log - 1,
          BITS_TO_ENCODE_DELTA_LOOKBACK_WINDOW_N_LOG,
        );
        writer.write_bitlen(
          config.state_n_log,
          BITS_TO_ENCODE_DELTA_LOOKBACK_STATE_N_LOG,
        );
        writer.write_bool(*secondary_uses_delta);
      }
      Self::Conv1(config) => {
        writer.write_bitlen(
          config.quantization,
          BITS_TO_ENCODE_DELTA_CONV_QUANTIZATION,
        );
        writer.write_uint(config.bias.to_latent_ordered(), 64);
        writer.write_usize(
          config.weights.len() - 1,
          BITS_TO_ENCODE_DELTA_CONV_N_WEIGHTS,
        );
        for &weight in &config.weights {
          writer.write_uint((weight as i32).to_latent_ordered(), 32);
        }
      }
    }
  }

  pub(crate) fn latent_type(&self) -> Option<LatentType> {
    match self {
      Self::NoOp | Self::Consecutive { .. } | Self::Conv1(_) | Self::Conv2(_) => Option::None,
      Self::Lookback { .. } => Some(LatentType::U32),
    }
  }

  pub(crate) fn for_latent_var(&self, key: LatentVarKey) -> LatentVarDeltaEncoding {
    match (self, key) {
      (Self::NoOp, _) => LatentVarDeltaEncoding::NoOp,
      // We never recursively delta encode.
      (_, LatentVarKey::Delta) => LatentVarDeltaEncoding::NoOp,
      // We always apply the DeltaEncoding to the primary latents.
      (Self::Consecutive { order, .. }, LatentVarKey::Primary) => {
        LatentVarDeltaEncoding::Consecutive(*order)
      }
      (
        Self::Consecutive {
          order,
          secondary_uses_delta: true,
        },
        LatentVarKey::Secondary,
      ) => LatentVarDeltaEncoding::Consecutive(*order),
      (
        Self::Consecutive {
          secondary_uses_delta: false,
          ..
        },
        LatentVarKey::Secondary,
      ) => LatentVarDeltaEncoding::NoOp,
      (Self::Lookback { config, .. }, LatentVarKey::Primary) => {
        LatentVarDeltaEncoding::Lookback(*config)
      }
      (
        Self::Lookback {
          config,
          secondary_uses_delta: true,
        },
        LatentVarKey::Secondary,
      ) => LatentVarDeltaEncoding::Lookback(*config),
      (
        Self::Lookback {
          secondary_uses_delta: false,
          ..
        },
        LatentVarKey::Secondary,
      ) => LatentVarDeltaEncoding::NoOp,
      (Self::Conv1(config), LatentVarKey::Primary) => LatentVarDeltaEncoding::Conv1(config.clone()),
      (Self::Conv1(_), LatentVarKey::Secondary) => LatentVarDeltaEncoding::NoOp,
      (Self::Conv2(config), LatentVarKey::Primary) => LatentVarDeltaEncoding::Conv2(config.clone()),
      (Self::Conv2(_), LatentVarKey::Secondary) => LatentVarDeltaEncoding::NoOp,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::bit_writer::BitWriter;

  fn check_bit_size(encoding: DeltaEncoding) {
    let mut bytes = Vec::new();
    let mut writer = BitWriter::new(
      &mut bytes,
      DeltaEncoding::MAX_BIT_SIZE as usize, // this is like 8x more than we need
    );
    unsafe {
      encoding.write_to(&mut writer);
    }
    let true_bit_size = writer.bit_idx();
    assert!(true_bit_size <= DeltaEncoding::MAX_BIT_SIZE);
  }

  #[test]
  fn test_bit_size() {
    check_bit_size(DeltaEncoding::NoOp);
    check_bit_size(DeltaEncoding::Consecutive {
      order: 3,
      secondary_uses_delta: false,
    });
    check_bit_size(DeltaEncoding::Lookback {
      config: DeltaLookbackConfig {
        window_n_log: 8,
        state_n_log: 1,
      },
      secondary_uses_delta: true,
    });
    check_bit_size(DeltaEncoding::Conv1(DeltaConv1Config {
      quantization: 31,
      bias: i64::MAX,
      weights: vec![i64::MAX; MAX_CONV1_DELTA_ORDER],
    }));
  }
}
