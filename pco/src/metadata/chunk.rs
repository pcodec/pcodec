use std::io::Write;

use better_io::BetterBufRead;

use crate::bit_reader::BitReaderBuilder;
use crate::bit_writer::BitWriter;
use crate::constants::{DeltaLookback, MAX_CONV1_DELTA_QUANTIZATION, OVERSHOOT_PADDING};
use crate::data_types::{Latent, LatentType};
use crate::errors::{PcoError, PcoResult};
use crate::macros::match_latent_enum;
use crate::metadata::chunk_latent_var::ChunkLatentVarMeta;
use crate::metadata::delta_encoding::DeltaEncoding;
use crate::metadata::format_version::FormatVersion;
use crate::metadata::per_latent_var::PerLatentVar;
use crate::metadata::{DynBins, Mode};

/// The metadata of a pco chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkMeta {
  /// The formula `pco` used to compress each number at a low level.
  pub mode: Mode,
  /// How delta encoding was applied.
  pub delta_encoding: DeltaEncoding,
  /// Metadata about the interleaved streams needed by `pco` to
  /// compress/decompress the inputs
  /// according to the formula used by `mode`.
  pub per_latent_var: PerLatentVar<ChunkLatentVarMeta>,
  _private: (),
}

impl ChunkMeta {
  pub fn new(
    mode: Mode,
    delta_encoding: DeltaEncoding,
    per_latent_var: PerLatentVar<ChunkLatentVarMeta>,
  ) -> PcoResult<Self> {
    // validate the delta encoding is compatible with everything else
    match &delta_encoding {
      DeltaEncoding::NoOp | DeltaEncoding::Consecutive { .. } => (),
      DeltaEncoding::Lookback { config, .. } => {
        let Some(latent_var) = &per_latent_var.delta else {
          unreachable!("Lookback delta encoding should always have a delta latent var");
        };

        let window_n = config.window_n() as DeltaLookback;
        let bins = latent_var.bins.downcast_ref::<DeltaLookback>().unwrap();
        let maybe_corrupt_bin = bins
          .iter()
          .find(|bin| bin.lower < 1 || bin.lower > window_n);
        if let Some(corrupt_bin) = maybe_corrupt_bin {
          return Err(PcoError::corruption(format!(
            "delta lookback bin had invalid lower bound of {} outside window [1, {}]",
            corrupt_bin.lower, window_n
          )));
        }
      }
      DeltaEncoding::Conv1(config) => {
        match &per_latent_var.primary.bins {
          DynBins::U16(_) | DynBins::U32(_) => (),
          DynBins::U64(_) => {
            return Err(PcoError::corruption(
              "Conv1 delta encodings are not supported on types larger than 32 bits",
            ))
          }
        }
        let (l_bits, conv_bits) = match_latent_enum!(
          &per_latent_var.primary.bins,
          DynBins<L>(_bins) => {
            (L::BITS, <L as Latent>::Conv::BITS)
          }
        );
        let max_quantization = MAX_CONV1_DELTA_QUANTIZATION.min(conv_bits - 1);
        if config.quantization > max_quantization {
          return Err(PcoError::corruption(format!(
            "Conv1 delta encoding quantization of {} exceeds max of {}",
            config.quantization, max_quantization
          )));
        }

        let max_pred = (config.bias::<i64>() as f64).abs()
          + 2.0_f64.powi(l_bits as i32)
            * config
              .weights::<i64>()
              .iter()
              .map(|w| w.abs() as f64)
              .sum::<f64>();
        if max_pred >= 2.0_f64.powi(conv_bits as i32 - 1) {
          return Err(PcoError::corruption(format!(
            "Conv1 delta encoding weights and bias risk overflowing as high as {}",
            max_pred,
          )));
        }
      }
    }

    Ok(Self {
      mode,
      delta_encoding,
      per_latent_var,
      _private: (),
    })
  }

  pub(crate) fn max_size(&self) -> usize {
    let bits_for_latent_vars = self
      .per_latent_var
      .as_ref()
      .map(|_, var_meta| var_meta.exact_bit_size())
      .sum();
    let n_bits = self.mode.max_bit_size() + DeltaEncoding::MAX_BIT_SIZE + bits_for_latent_vars;
    n_bits.div_ceil(8)
  }

  pub(crate) fn exact_page_meta_size(&self) -> usize {
    let bit_size = self
      .per_latent_var
      .as_ref()
      .map(|key, var_meta| {
        let delta_encoding = self.delta_encoding.for_latent_var(key);
        var_meta.exact_page_meta_bit_size(&delta_encoding)
      })
      .sum();
    bit_size.div_ceil(8)
  }

  pub(crate) fn read_from<R: BetterBufRead>(
    reader_builder: &mut BitReaderBuilder<R>,
    version: &FormatVersion,
    latent_type: LatentType,
  ) -> PcoResult<Self> {
    let mode = Mode::read_from(reader_builder, version, latent_type)?;
    let delta_encoding = reader_builder.with_reader(
      DeltaEncoding::MAX_BIT_SIZE.div_ceil(8) + OVERSHOOT_PADDING,
      |reader| unsafe { DeltaEncoding::read_from(reader, version) },
    )?;

    let delta = if let Some(delta_latent_type) = delta_encoding.latent_type() {
      Some(ChunkLatentVarMeta::read_from::<R>(
        reader_builder,
        delta_latent_type,
      )?)
    } else {
      None
    };

    let primary = ChunkLatentVarMeta::read_from::<R>(
      reader_builder,
      mode.primary_latent_type(latent_type),
    )?;

    let secondary = if let Some(secondary_latent_type) = mode.secondary_latent_type(latent_type) {
      Some(ChunkLatentVarMeta::read_from::<R>(
        reader_builder,
        secondary_latent_type,
      )?)
    } else {
      None
    };

    let per_latent_var = PerLatentVar {
      delta,
      primary,
      secondary,
    };

    reader_builder.with_reader(1, |reader| {
      reader.drain_empty_byte("nonzero bits in end of final byte of chunk metadata")
    })?;

    Self::new(mode, delta_encoding, per_latent_var)
  }

  pub(crate) unsafe fn write_to<W: Write>(&self, writer: &mut BitWriter<W>) -> PcoResult<()> {
    self.mode.write_to(writer);
    self.delta_encoding.write_to(writer);

    writer.flush()?;

    for (_, latents) in self.per_latent_var.as_ref().enumerated() {
      latents.write_to(writer)?;
    }

    writer.finish_byte();
    writer.flush()?;
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::constants::ANS_INTERLEAVING;
  use crate::data_types::Latent;
  use crate::macros::match_latent_enum;
  use crate::metadata::dyn_bins::DynBins;
  use crate::metadata::dyn_latents::DynLatents;
  use crate::metadata::page::PageMeta;
  use crate::metadata::page_latent_var::PageLatentVarMeta;
  use crate::metadata::{Bin, DynLatent};

  fn check_sizes(meta: &ChunkMeta) -> PcoResult<()> {
    let buffer_size = 8192;
    let mut dst = Vec::new();
    let mut writer = BitWriter::new(&mut dst, buffer_size);
    unsafe { meta.write_to(&mut writer)? };
    writer.flush()?;
    assert!(dst.len() <= meta.max_size());

    // page meta size
    let mut dst = Vec::new();
    let mut writer = BitWriter::new(&mut dst, buffer_size);
    let page_meta = PageMeta {
      per_latent_var: meta.per_latent_var.as_ref().map(|key, latent_var_meta| {
        let delta_encoding = meta.delta_encoding.for_latent_var(key);
        let delta_moments = match_latent_enum!(
          &latent_var_meta.bins,
          DynBins<L>(_bins) => {
            DynLatents::new(vec![L::ZERO; delta_encoding.n_latents_per_state()])
          }
        );
        PageLatentVarMeta {
          delta_state: delta_moments,
          ans_final_state_idxs: [0; ANS_INTERLEAVING],
        }
      }),
    };
    unsafe {
      page_meta.write_to(
        meta
          .per_latent_var
          .as_ref()
          .map(|_, var_meta| var_meta.ans_size_log),
        &mut writer,
      )
    };
    writer.flush()?;
    assert_eq!(meta.exact_page_meta_size(), dst.len());
    Ok(())
  }

  #[test]
  fn exact_size_binless() -> PcoResult<()> {
    let meta = ChunkMeta {
      mode: Mode::Classic,
      delta_encoding: DeltaEncoding::Consecutive {
        order: 5,
        secondary_uses_delta: false,
      },
      per_latent_var: PerLatentVar {
        delta: None,
        primary: ChunkLatentVarMeta {
          ans_size_log: 0,
          bins: DynBins::U32(vec![]),
        },
        secondary: None,
      },
      _private: (),
    };

    check_sizes(&meta)
  }

  #[test]
  fn exact_size_trivial() -> PcoResult<()> {
    let meta = ChunkMeta {
      mode: Mode::Classic,
      delta_encoding: DeltaEncoding::NoOp,
      per_latent_var: PerLatentVar {
        delta: None,
        primary: ChunkLatentVarMeta {
          ans_size_log: 0,
          bins: DynBins::U64(vec![Bin {
            weight: 1,
            lower: 77_u64,
            offset_bits: 0,
          }]),
        },
        secondary: None,
      },
      _private: (),
    };

    check_sizes(&meta)
  }

  #[test]
  fn exact_size_float_mult() -> PcoResult<()> {
    let meta = ChunkMeta {
      mode: Mode::FloatMult(DynLatent::U32(777_u32)),
      delta_encoding: DeltaEncoding::Consecutive {
        order: 3,
        secondary_uses_delta: false,
      },
      per_latent_var: PerLatentVar {
        delta: None,
        primary: ChunkLatentVarMeta {
          ans_size_log: 7,
          bins: DynBins::U32(vec![
            Bin {
              weight: 11,
              lower: 0_u32,
              offset_bits: 0,
            },
            Bin {
              weight: 117,
              lower: 1,
              offset_bits: 0,
            },
          ]),
        },
        secondary: Some(ChunkLatentVarMeta {
          ans_size_log: 3,
          bins: DynBins::U32(vec![
            Bin {
              weight: 3,
              lower: 0_u32,
              offset_bits: 0,
            },
            Bin {
              weight: 5,
              lower: 1,
              offset_bits: 0,
            },
          ]),
        }),
      },
      _private: (),
    };

    check_sizes(&meta)
  }
}
