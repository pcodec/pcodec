use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::constants::{
  Bitlen, BITS_TO_ENCODE_DELTA_ENCODING_ORDER, BITS_TO_ENCODE_DELTA_ENCODING_VARIANT,
  BITS_TO_ENCODE_LZ_DELTA_STATE_N_LOG, BITS_TO_ENCODE_LZ_DELTA_WINDOW_N_LOG,
};
use crate::data_types::LatentType;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::delta_encoding::DeltaEncoding::*;
use crate::metadata::format_version::FormatVersion;
use crate::metadata::per_latent_var::LatentVarKey;
use std::io::Write;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaConsecutiveConfig {
  /// The number of times consecutive deltas were taken.
  /// For instance, 2nd order delta encoding is delta-of-deltas.
  ///
  /// This is always positive, between 1 and 7.
  pub order: usize,
  pub secondary_uses_delta: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaLookbackConfig {
  /// The log2 of the number of latents explicitly stored in page metadata
  /// to prepopulate the lookback window.
  pub state_n_log: Bitlen,
  /// The log2 of the maximum possible lookback.
  pub window_n_log: Bitlen,
  pub secondary_uses_delta: bool,
}

impl DeltaLookbackConfig {
  pub(crate) fn state_n(&self) -> usize {
    1 << self.state_n_log
  }

  pub(crate) fn window_n(&self) -> usize {
    1 << self.window_n_log
  }
}

/// How Pco did
/// [delta encoding](https://en.wikipedia.org/wiki/Delta_encoding) on this
/// chunk.
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
  None,
  /// Encodes the differences between consecutive values (or differences
  /// between those, etc.).
  ///
  /// This is best if your numbers have high variance overall, but adjacent
  /// numbers are close in value, e.g. an arithmetic sequence.
  Consecutive(DeltaConsecutiveConfig),
  /// Encodes an extra "lookback" latent variable and the differences
  /// `x[i] - x[i - lookback[i]]` between values.
  ///
  /// This is best if your numbers have complex repeating patterns
  /// beyond just adjacent elements.
  /// It is in spirit similar to LZ77 compression, but only stores lookbacks
  /// (AKA match offsets) and no match lengths.
  Lookback(DeltaLookbackConfig),
}

impl DeltaEncoding {
  unsafe fn read_from_pre_v3(reader: &mut BitReader) -> Self {
    let order = reader.read_usize(BITS_TO_ENCODE_DELTA_ENCODING_ORDER);
    match order {
      0 => None,
      _ => Consecutive(DeltaConsecutiveConfig {
        order,
        secondary_uses_delta: false,
      }),
    }
  }

  pub(crate) unsafe fn read_from(
    version: &FormatVersion,
    reader: &mut BitReader,
  ) -> PcoResult<Self> {
    if !version.supports_delta_variants() {
      return Ok(Self::read_from_pre_v3(reader));
    }

    let delta_encoding_variant = reader.read_bitlen(BITS_TO_ENCODE_DELTA_ENCODING_VARIANT);

    let res = match delta_encoding_variant {
      0 => None,
      1 => {
        let order = reader.read_usize(BITS_TO_ENCODE_DELTA_ENCODING_ORDER);
        if order == 0 {
          return Err(PcoError::corruption(
            "Consecutive delta encoding order must not be 0",
          ));
        } else {
          Consecutive(DeltaConsecutiveConfig {
            order,
            secondary_uses_delta: reader.read_bool(),
          })
        }
      }
      2 => {
        let window_n_log = 1 + reader.read_bitlen(BITS_TO_ENCODE_LZ_DELTA_WINDOW_N_LOG);
        let state_n_log = reader.read_bitlen(BITS_TO_ENCODE_LZ_DELTA_STATE_N_LOG);
        if state_n_log > window_n_log {
          return Err(PcoError::corruption(format!(
            "LZ delta encoding state size log exceeded window size log: {} vs {}",
            state_n_log, window_n_log
          )));
        }
        Lookback(DeltaLookbackConfig {
          window_n_log,
          state_n_log,
          secondary_uses_delta: reader.read_bool(),
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
      None => 0,
      Consecutive(_) => 1,
      Lookback(_) => 2,
    };
    writer.write_bitlen(
      variant,
      BITS_TO_ENCODE_DELTA_ENCODING_VARIANT,
    );

    match self {
      None => (),
      Consecutive(config) => {
        writer.write_usize(
          config.order,
          BITS_TO_ENCODE_DELTA_ENCODING_ORDER,
        );
        writer.write_bool(config.secondary_uses_delta);
      }
      Lookback(config) => {
        writer.write_bitlen(
          config.window_n_log - 1,
          BITS_TO_ENCODE_LZ_DELTA_WINDOW_N_LOG,
        );
        writer.write_bitlen(
          config.state_n_log,
          BITS_TO_ENCODE_LZ_DELTA_STATE_N_LOG,
        );
        writer.write_bool(config.secondary_uses_delta);
      }
    }
  }

  pub(crate) fn latent_type(&self) -> Option<LatentType> {
    match self {
      None | Consecutive(_) => Option::None,
      Lookback(_) => Some(LatentType::U32),
    }
  }

  pub(crate) fn applies_to_latent_var(&self, key: LatentVarKey) -> bool {
    match (self, key) {
      // We never recursively delta encode.
      (_, LatentVarKey::Delta) => false,
      // We always apply the DeltaEncoding to the primary latents.
      (_, LatentVarKey::Primary) => true,
      (None, LatentVarKey::Secondary) => false,
      (Consecutive(config), LatentVarKey::Secondary) => config.secondary_uses_delta,
      (Lookback(config), LatentVarKey::Secondary) => config.secondary_uses_delta,
    }
  }

  pub(crate) fn for_latent_var(&self, key: LatentVarKey) -> DeltaEncoding {
    if self.applies_to_latent_var(key) {
      self.clone()
    } else {
      None
    }
  }

  pub(crate) fn n_latents_per_state(&self) -> usize {
    match self {
      None => 0,
      Consecutive(config) => config.order,
      Lookback(config) => 1 << config.state_n_log,
    }
  }

  pub(crate) fn exact_bit_size(&self) -> Bitlen {
    let payload_bits = match self {
      None => 0,
      // For nontrivial encodings, we have a +1 bit for whether the
      // secondary latent is delta-encoded or not.
      Consecutive(_) => BITS_TO_ENCODE_DELTA_ENCODING_ORDER + 1,
      Lookback(_) => BITS_TO_ENCODE_LZ_DELTA_WINDOW_N_LOG + BITS_TO_ENCODE_LZ_DELTA_STATE_N_LOG + 1,
    };
    BITS_TO_ENCODE_DELTA_ENCODING_VARIANT + payload_bits
  }
}

#[cfg(test)]
mod tests {
  use crate::bit_writer::BitWriter;
  use crate::metadata::delta_encoding::{DeltaConsecutiveConfig, DeltaLookbackConfig};
  use crate::metadata::DeltaEncoding;

  fn check_bit_size(encoding: DeltaEncoding) {
    let mut bytes = Vec::new();
    let mut writer = BitWriter::new(&mut bytes, 100);
    unsafe {
      encoding.write_to(&mut writer);
    }
    assert_eq!(
      encoding.exact_bit_size() as usize,
      writer.bit_idx(),
    );
  }

  #[test]
  fn test_bit_size() {
    check_bit_size(DeltaEncoding::None);
    check_bit_size(DeltaEncoding::Consecutive(
      DeltaConsecutiveConfig {
        order: 3,
        secondary_uses_delta: false,
      },
    ));
    check_bit_size(DeltaEncoding::Lookback(
      DeltaLookbackConfig {
        window_n_log: 8,
        state_n_log: 1,
        secondary_uses_delta: true,
      },
    ));
  }
}
