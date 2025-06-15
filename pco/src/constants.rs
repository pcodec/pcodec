// must be u16 or larger
// u32 is performant because it aligns with other things and doesn't require
// masking
// exposed in public API
pub(crate) type Bitlen = u32;
// must be u32 or larger
// exposed in public API
pub(crate) type Weight = u32;
pub(crate) type DeltaLookback = u32;

// compatibility
pub const CURRENT_FORMAT_VERSION: u8 = 3;

// bit lengths
pub const BITS_TO_ENCODE_ANS_SIZE_LOG: Bitlen = 4;
pub const BITS_TO_ENCODE_MODE_VARIANT: Bitlen = 4;
pub const BITS_TO_ENCODE_DELTA_ENCODING_VARIANT: Bitlen = 4;
pub const BITS_TO_ENCODE_DELTA_ENCODING_ORDER: Bitlen = 3;
pub const BITS_TO_ENCODE_LZ_DELTA_WINDOW_N_LOG: Bitlen = 5;
pub const BITS_TO_ENCODE_LZ_DELTA_STATE_N_LOG: Bitlen = 4;
pub const BITS_TO_ENCODE_N_BINS: Bitlen = 15;
// conservative: wide enough to support quantizing float datasets with 255 unused bits of precision
pub const BITS_TO_ENCODE_QUANTIZE_K: Bitlen = 8;

// padding
pub const HEADER_PADDING: usize = 1;
// + 9 because we might read an extra u64 (8 bytes), plus 1 for good measure
pub const OVERSHOOT_PADDING: usize = MAX_SUPPORTED_PRECISION_BYTES + 9;
// Chunk meta padding is enough for one full batch of bins; this should also
// generously cover the data needed to read the other parts of chunk meta.
pub const CHUNK_META_PADDING: usize =
  FULL_BIN_BATCH_SIZE * (4 + 2 * MAX_SUPPORTED_PRECISION_BYTES) + OVERSHOOT_PADDING;
// Page padding is enough for one full batch of latents; this should also
// generously cover the data needed to read the page meta.
pub const PAGE_PADDING: usize =
  FULL_BATCH_N * (MAX_SUPPORTED_PRECISION_BYTES + MAX_ANS_BYTES) + OVERSHOOT_PADDING;

// cutoffs and legal parameter values
pub const MAX_ANS_BITS: Bitlen = 14;
pub const MAX_ANS_BYTES: usize = MAX_ANS_BITS.div_ceil(8) as usize;
pub const LIMITED_UNOPTIMIZED_BINS_LOG: Bitlen = 6;
pub const MAX_COMPRESSION_LEVEL: usize = 12;
pub const MAX_DELTA_ENCODING_ORDER: usize = 7;
pub const MAX_ENTRIES: usize = 1 << 24;
pub const MAX_SUPPORTED_PRECISION: Bitlen = 128;
pub const MAX_SUPPORTED_PRECISION_BYTES: usize = (MAX_SUPPORTED_PRECISION / 8) as usize;
pub const MULT_REQUIRED_BITS_SAVED_PER_NUM: f64 = 0.5;
pub const QUANT_REQUIRED_BITS_SAVED_PER_NUM: f64 = 1.5;
pub const CLASSIC_MEMORIZABLE_BINS_LOG: Bitlen = 8;

// defaults
pub const DEFAULT_COMPRESSION_LEVEL: usize = 8;
// if you modify default page size, update docs for PagingSpec
pub const DEFAULT_MAX_PAGE_N: usize = 1 << 18;

// important parts of the format specification
pub const ANS_INTERLEAVING: usize = 4;
/// The count of numbers per batch, the smallest unit of decompression.
///
/// Only the final batch in each page may have fewer numbers than this.
pub const FULL_BATCH_N: usize = 256;
pub const FULL_BIN_BATCH_SIZE: usize = 128;

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
  fn test_ans_interleaving_fits_in_u64() {
    assert!(ANS_INTERLEAVING * MAX_ANS_BITS as usize <= 57);
  }
}
