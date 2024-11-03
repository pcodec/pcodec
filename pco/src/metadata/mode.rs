use crate::bit_reader::BitReader;
use crate::bit_writer::BitWriter;
use crate::constants::{Bitlen, BITS_TO_ENCODE_MODE_VARIANT, BITS_TO_ENCODE_QUANTIZE_K};
use crate::data_types::{Float, LatentType};
use crate::errors::{PcoError, PcoResult};
use crate::macros::match_latent_enum;
use crate::metadata::dyn_latent::DynLatent;
use crate::metadata::format_version::FormatVersion;
use crate::metadata::Mode::*;
use std::fmt::Debug;
use std::io::Write;

// Internally, here's how we should model each mode:
//
// Classic: The data is drawn from a smooth distribution.
//   Most natural data is like this.
//
// IntMult: The data is generated by 2 smooth distributions:
//   one whose outputs are multiplied by the base, and another whose outputs
//   are in the range [0, base). The 2nd process is often but not always
//   trivial.
//
// FloatMult: The data is generated by a smooth distribution
//   whose outputs get multiplied by the base and perturbed by floating point
//   errors.
//
// FloatQuant: The data is generated by first drawing from a smooth distribution
//   on low-precision floats, then widening the result by adding
//   less-significant bits drawn from a second, very low-entropy distribution
//   (e.g. in the common case, one that always produces zeros).
//
// Note the differences between int mult and float mult,
// which have equivalent formulas.

/// How Pco does the first step of processing for this chunk.
///
/// Each mode splits the vector of numbers into one or two vectors of latents,
/// with a different formula for how the split and (and eventual join during
/// decompression) is done.
/// Each of these vectors of latents is then passed through Pco's subsequent
/// processing steps (possible delta encoding and binning) to produce the final
/// compressed bytes.
///
/// We have delibrately written the formulas below in a slightly wrong way to
/// convey the correct intuition without dealing with implementation
/// complexities.
/// Slightly more rigorous formulas are in format.md.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Mode {
  /// Represents each number as a single latent: itself.
  ///
  /// Formula: `num = num`
  #[default]
  Classic,
  /// Given a `base`, represents each number as two latents: a multiplier
  /// on the base and an adjustment.
  ///
  /// Only applies to integers.
  ///
  /// Formula: `num = mode.base * mult + adjustment`
  IntMult(DynLatent),
  /// Given a float `base`, represents each number as two latents: a
  /// multiplier on the base and an ULPs (units-in-the-last-place) adjustment.
  ///
  /// Only applies to floats.
  ///
  /// Formula: `num = mode.base * mult + adjustment ULPs`
  FloatMult(DynLatent),
  /// Given a number of bits `k`, represents each number as two latents:
  /// quantums (effectively the first `TYPE_SIZE - k` bits) and an ULPs
  /// adjustment.
  ///
  /// Only applies to floats.
  ///
  /// Formula: `num = from_bits(quantums << k + adjustment)`
  /// (warning: this formula is especially simplified)
  FloatQuant(Bitlen),
}

impl Mode {
  pub(crate) unsafe fn read_from(
    reader: &mut BitReader,
    version: &FormatVersion,
    latent_type: LatentType,
  ) -> PcoResult<Self> {
    let read_latent = |reader| {
      match_latent_enum!(
        latent_type,
        LatentType<L> => {
          DynLatent::read_uncompressed_from::<L>(reader)
        }
      )
    };

    let mode = match reader.read_bitlen(BITS_TO_ENCODE_MODE_VARIANT) {
      0 => Classic,
      1 => {
        if version.used_old_gcds() {
          return Err(PcoError::compatibility(
            "unable to decompress data from v0.0.0 of pco with different GCD encoding",
          ));
        }

        let base = read_latent(reader);
        IntMult(base)
      }
      2 => {
        let base_latent = read_latent(reader);
        FloatMult(base_latent)
      }
      3 => {
        let k = reader.read_bitlen(BITS_TO_ENCODE_QUANTIZE_K);
        FloatQuant(k)
      }
      value => {
        return Err(PcoError::corruption(format!(
          "unknown mode value {}",
          value
        )))
      }
    };
    Ok(mode)
  }

  pub(crate) unsafe fn write_to<W: Write>(&self, writer: &mut BitWriter<W>) {
    let mode_value = match self {
      Classic => 0,
      IntMult(_) => 1,
      FloatMult { .. } => 2,
      FloatQuant { .. } => 3,
    };
    writer.write_bitlen(mode_value, BITS_TO_ENCODE_MODE_VARIANT);
    match self {
      Classic => (),
      IntMult(base) => {
        base.write_uncompressed_to(writer);
      }
      FloatMult(base_latent) => {
        base_latent.write_uncompressed_to(writer);
      }
      &FloatQuant(k) => {
        writer.write_uint(k, BITS_TO_ENCODE_QUANTIZE_K);
      }
    };
  }

  pub(crate) fn primary_latent_type(&self, number_latent_type: LatentType) -> LatentType {
    match self {
      Classic | FloatMult(_) | FloatQuant(_) | IntMult(_) => number_latent_type,
    }
  }

  pub(crate) fn secondary_latent_type(&self, number_latent_type: LatentType) -> Option<LatentType> {
    match self {
      Classic => None,
      FloatMult(_) | FloatQuant(_) | IntMult(_) => Some(number_latent_type),
    }
  }
  // pub(crate) fn has_secondary_latent_var(&self) -> bool {
  //   match self {
  //     Classic => false,
  //     FloatMult(_) | IntMult(_) | FloatQuant(_) => true,
  //   }
  // }

  pub(crate) fn float_mult<F: Float>(base: F) -> Self {
    Self::FloatMult(DynLatent::new(base.to_latent_ordered()).unwrap())
  }

  pub(crate) fn exact_bit_size(&self) -> Bitlen {
    let payload_bits = match self {
      Classic => 0,
      IntMult(base) | FloatMult(base) => base.bits(),
      FloatQuant(_) => BITS_TO_ENCODE_QUANTIZE_K,
    };
    BITS_TO_ENCODE_MODE_VARIANT + payload_bits
  }
}
