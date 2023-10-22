use crate::bits;

// must be u8 or larger
// u64+ seems less performant
// exposed in public API
pub(crate) type Bitlen = u32;
// must be u32 or larger
// exposed in public API
pub(crate) type Weight = u32;

// compatibility
pub const CURRENT_FORMAT_VERSION: u8 = 0; // .

// bit lengths
pub const BITS_TO_ENCODE_ANS_SIZE_LOG: Bitlen = 4;
pub const BITS_TO_ENCODE_DELTA_ENCODING_ORDER: Bitlen = 3;
pub const BITS_TO_ENCODE_MODE: Bitlen = 4;
pub const BITS_TO_ENCODE_N_BINS: Bitlen = 15;

// padding
pub const FULL_BIN_BATCH_SIZE: usize = 128;
const OVERSHOOT_PADDING: usize = MAX_SUPPORTED_PRECISION_BYTES + BYTES_PER_WORD + 1;
pub const MINIMAL_PADDING_BYTES: usize = 1 << 6;
pub const BIN_BATCH_PADDING: usize = FULL_BIN_BATCH_SIZE * (4 + 2 * MAX_SUPPORTED_PRECISION_BYTES) + OVERSHOOT_PADDING;
pub const PAGE_LATENT_META_PADDING: usize =
  MAX_DELTA_ENCODING_ORDER * MAX_SUPPORTED_PRECISION_BYTES + MAX_ANS_BYTES + OVERSHOOT_PADDING;
pub const LATENT_BATCH_PADDING: usize =
  FULL_BATCH_SIZE * (MAX_SUPPORTED_PRECISION_BYTES + MAX_ANS_BYTES) + OVERSHOOT_PADDING;
pub const DEFAULT_PADDING_BYTES: usize = 1 << 13;

// native architecture info
pub const WORD_SIZE: usize = usize::BITS as usize;
pub const WORD_BITLEN: Bitlen = usize::BITS as Bitlen;
pub const BYTES_PER_WORD: usize = WORD_SIZE / 8;

// cutoffs and legal parameter values
pub const AUTO_DELTA_LIMIT: usize = 1100;
pub const MAX_ANS_BITS: Bitlen = 14;
pub const MAX_ANS_BYTES: usize = bits::ceil_div(MAX_ANS_BITS as usize, 8);
pub const MAX_AUTO_DELTA_COMPRESSION_LEVEL: usize = 6;
pub const MAX_COMPRESSION_LEVEL: usize = 12;
pub const MAX_DELTA_ENCODING_ORDER: usize = 7;
pub const MAX_ENTRIES: usize = 1 << 24;
pub const MAX_SUPPORTED_PRECISION: Bitlen = 128;
pub const MAX_SUPPORTED_PRECISION_BYTES: usize = (MAX_SUPPORTED_PRECISION / 8) as usize;

// defaults
pub const DEFAULT_COMPRESSION_LEVEL: usize = 8;
pub const DEFAULT_MAX_PAGE_SIZE: usize = 1000000;

// important parts of the format specification
pub const ANS_INTERLEAVING: usize = 4;
pub const FULL_BATCH_SIZE: usize = 256;

#[cfg(test)]
mod tests {
  use crate::constants::*;

  fn bits_to_encode(max_value: usize) -> Bitlen {
    usize::BITS - max_value.leading_zeros()
  }

  fn assert_can_encode(n_bits: Bitlen, max_number: usize) {
    assert!(n_bits >= bits_to_encode(max_number));
  }

  #[test]
  fn test_bits_to_encode_delta_encoding_order() {
    assert_can_encode(
      BITS_TO_ENCODE_DELTA_ENCODING_ORDER,
      MAX_DELTA_ENCODING_ORDER,
    );
  }

  #[test]
  fn test_bin_batch_fits_in_default_padding() {
    let max_bytes_per_bin = 2 * MAX_SUPPORTED_PRECISION as usize / 8 + 4;
    assert!(FULL_BIN_BATCH_SIZE * max_bytes_per_bin < DEFAULT_PADDING_BYTES);
  }

  #[test]
  fn test_batch_fits_in_default_padding() {
    let max_bytes_per_num = MAX_SUPPORTED_PRECISION as usize / 8 + 2;
    assert!(FULL_BATCH_SIZE * max_bytes_per_num < DEFAULT_PADDING_BYTES);
  }
}
