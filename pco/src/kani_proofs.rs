//! Kani proof harnesses for the unsafe decode path.
//!
//! The crate reads compressed data through raw-pointer loads in
//! `bit_reader` that have no bounds checks. Their safety rests on a
//! *padding contract*: the caller must guarantee the source slice has
//! enough bytes past the current index. These harnesses state that
//! contract explicitly and let Kani check it for every input, rather
//! than for the fixed corpus the unit tests use.

use crate::bit_reader::{read_uint_at, u32_at, u64_at, BitReader};
use crate::bits;
use crate::constants::{Bitlen, Weight, MAX_ANS_BITS};
use crate::read_write_uint::calc_max_bytes;

// Enough room that a nondeterministic byte_idx still has interesting
// slack, but small enough to keep the model tractable.
const SRC_LEN: usize = 32;

// ---------------------------------------------------------------------------
// bits::lowest_bits{,_fast}
// ---------------------------------------------------------------------------

// `lowest_bits_fast` documents "doesn't handle the case when n >= U::BITS".
// Below that bound it must be a plain mask, with no shift overflow.
#[kani::proof]
fn proof_lowest_bits_fast_u64_is_a_mask() {
  let x: u64 = kani::any();
  let n: Bitlen = kani::any();
  kani::assume(n < 64);

  let got = bits::lowest_bits_fast(x, n);
  let want = x & ((1u64 << n) - 1);
  assert_eq!(got, want);
}

#[kani::proof]
fn proof_lowest_bits_fast_u32_is_a_mask() {
  let x: u32 = kani::any();
  let n: Bitlen = kani::any();
  kani::assume(n < 32);

  let got = bits::lowest_bits_fast(x, n);
  let want = x & ((1u32 << n) - 1);
  assert_eq!(got, want);
}

// The checked variant must be total: no n at all may panic, and n >= BITS
// must be the identity.
#[kani::proof]
fn proof_lowest_bits_u64_is_total() {
  let x: u64 = kani::any();
  let n: Bitlen = kani::any();

  let got = bits::lowest_bits(x, n);
  if n >= 64 {
    assert_eq!(got, x);
  } else {
    assert_eq!(got, x & ((1u64 << n) - 1));
  }
}

// ---------------------------------------------------------------------------
// raw loads
// ---------------------------------------------------------------------------

// The safety contract of `u64_at` / `u32_at`: byte_idx + width must fit in the
// slice. Kani checks the raw pointer dereference itself, so a passing harness
// means the load is genuinely in bounds for every admissible index.
#[kani::proof]
fn proof_u64_at_in_bounds() {
  let src: [u8; SRC_LEN] = kani::any();
  let byte_idx: usize = kani::any();
  kani::assume(byte_idx <= SRC_LEN - 8);

  let got = unsafe { u64_at(&src, byte_idx) };
  // and it really is a little-endian load of those 8 bytes
  let mut want = 0u64;
  for i in 0..8 {
    want |= (src[byte_idx + i] as u64) << (8 * i as u32);
  }
  assert_eq!(got, want);
}

#[kani::proof]
fn proof_u32_at_in_bounds() {
  let src: [u8; SRC_LEN] = kani::any();
  let byte_idx: usize = kani::any();
  kani::assume(byte_idx <= SRC_LEN - 4);

  let got = unsafe { u32_at(&src, byte_idx) };
  let mut want = 0u32;
  for i in 0..4 {
    want |= (src[byte_idx + i] as u32) << (8 * i as u32);
  }
  assert_eq!(got, want);
}

// ---------------------------------------------------------------------------
// read_uint_at: the padding contract, per READ_BYTES arm
// ---------------------------------------------------------------------------
//
// The comment in bit_reader.rs justifies the three arms by hand:
//   0..=25  bits -> 4 bytes,  26..=57 bits -> 8 bytes,  58..=113 bits -> 15.
// `read_uint` picks the arm from `U::MAX_BYTES`, which for the types this
// crate actually instantiates works out to:
//   u8, u16 -> 4 bytes    u32 -> 8 bytes    u64, usize -> 15 bytes
// Each harness pins one arm: given `bits_past_byte < 8` (what `refill`
// establishes) and n within that arm's stated range, the read touches only
// READ_BYTES bytes past byte_idx and no shift overflows.

#[kani::proof]
fn proof_read_uint_at_4_stays_in_padding() {
  let src: [u8; SRC_LEN] = kani::any();
  let byte_idx: usize = kani::any();
  let bits_past_byte: Bitlen = kani::any();
  let n: Bitlen = kani::any();

  kani::assume(byte_idx <= SRC_LEN - 4);
  kani::assume(bits_past_byte < 8);
  kani::assume(n <= 25);

  let _: u16 = unsafe { read_uint_at::<u16, 4>(&src, byte_idx, bits_past_byte, n) };
}

#[kani::proof]
fn proof_read_uint_at_8_stays_in_padding() {
  let src: [u8; SRC_LEN] = kani::any();
  let byte_idx: usize = kani::any();
  let bits_past_byte: Bitlen = kani::any();
  let n: Bitlen = kani::any();

  kani::assume(byte_idx <= SRC_LEN - 8);
  kani::assume(bits_past_byte < 8);
  kani::assume(n <= 57);

  let _: u32 = unsafe { read_uint_at::<u32, 8>(&src, byte_idx, bits_past_byte, n) };
}

// The 15-byte arm is the delicate one: it does a second load at byte_idx + 7
// and left-shifts by `56 - bits_past_byte`, which underflows if
// bits_past_byte > 56. Only `refill`'s < 8 invariant keeps that safe.
// This is the arm u64/usize reads go through, i.e. the common case.
#[kani::proof]
fn proof_read_uint_at_15_stays_in_padding() {
  let src: [u8; SRC_LEN] = kani::any();
  let byte_idx: usize = kani::any();
  let bits_past_byte: Bitlen = kani::any();
  let n: Bitlen = kani::any();

  kani::assume(byte_idx <= SRC_LEN - 15);
  kani::assume(bits_past_byte < 8);
  kani::assume(n <= 64);

  let _: u64 = unsafe { read_uint_at::<u64, 15>(&src, byte_idx, bits_past_byte, n) };
}

// ---------------------------------------------------------------------------
// calc_max_bytes: does the advertised padding actually cover the reads?
// ---------------------------------------------------------------------------

// `calc_max_bytes(precision)` is what callers use to size padding, and
// `read_uint` picks its READ_BYTES arm from `U::MAX_BYTES`. These must agree:
// the arm chosen for a type must never read past that type's MAX_BYTES.
#[kani::proof]
fn proof_calc_max_bytes_covers_read_arm() {
  let precision: Bitlen = kani::any();
  // 64 is the widest latent this crate supports.
  kani::assume(precision > 0 && precision <= 64);

  let max_bytes = calc_max_bytes(precision);
  // The dispatch in `read_uint` must not fall through to its unreachable!().
  let read_bytes = match max_bytes {
    1..=4 => 4,
    5..=8 => 8,
    9..=15 => 15,
    _ => panic!("MAX_BYTES outside every read_uint arm"),
  };
  // A read of `precision` bits starting up to 7 bits into a byte must fit in
  // read_bytes bytes -- otherwise the chosen arm truncates the value.
  assert!(precision + 7 <= 8 * read_bytes as Bitlen);
}

// ---------------------------------------------------------------------------
// BitReader: refill establishes the invariant the raw loads depend on
// ---------------------------------------------------------------------------

// `read_uint` calls `refill()` and then hands `bits_past_byte` straight to a
// shift. This proves refill normalizes any bit position into < 8, and that
// the total bit index is preserved (i.e. refill only re-splits, never moves).
#[kani::proof]
fn proof_bit_reader_read_uint_u64() {
  let src: [u8; SRC_LEN] = kani::any();
  let bits_past_byte: Bitlen = kani::any();
  let n: Bitlen = kani::any();

  // A reader positioned anywhere in the first half of the buffer, so that
  // the 15-byte arm (the one u64 dispatches to) always has its padding.
  kani::assume(bits_past_byte < 8 * (SRC_LEN as Bitlen / 2));
  kani::assume(n <= 64);

  let mut reader = BitReader::new(&src, SRC_LEN, bits_past_byte);
  let before = reader.bit_idx();
  let _: u64 = unsafe { reader.read_uint(n) };
  assert_eq!(reader.bit_idx(), before + n as usize);
}

// ---------------------------------------------------------------------------
// ANS spec: the arithmetic behind the state-symbol table
// ---------------------------------------------------------------------------
//
// `Spec::spread_state_symbols` builds a table of `table_size = 1 << size_log`
// entries, indexing it with `(stride * step) & mod_table_size`. Every piece of
// that is Weight (u32) arithmetic on values derived from the wire. The loop
// itself is too large to unwind (table_size can be 2^14), so instead we prove
// the loop-free facts it depends on, for every admissible size_log:
//
//   * the shift in `mod_table_size` does not underflow,
//   * `stride * step` does not overflow Weight,
//   * the resulting index is always < table_size, so `res[..]` can't panic.
//
// This is also a regression gate: if MAX_ANS_BITS is ever raised, the
// multiplication overflows and this harness starts failing.
#[kani::proof]
fn proof_ans_spread_index_arithmetic() {
  let size_log: Bitlen = kani::any();
  kani::assume(size_log <= MAX_ANS_BITS);

  let table_size: Weight = 1 << size_log;

  // mirror of choose_stride
  let mut stride = (3 * table_size) / 5;
  if stride.is_multiple_of(2) {
    stride += 1;
  }

  let mod_table_size = Weight::MAX >> 1 >> (Weight::BITS as Bitlen - 1 - size_log);
  assert_eq!(mod_table_size, table_size - 1);

  // `step` runs over [0, table_size) across the whole nested loop, because the
  // weights are validated to sum to exactly table_size.
  let step: Weight = kani::any();
  kani::assume(step < table_size);

  let state_idx = (stride * step) & mod_table_size;
  assert!((state_idx as usize) < table_size as usize);
}

// `weight = reader.read_uint::<Weight>(ans_size_log) + 1` for each of up to
// `n_bins` bins, and the sum must be representable. Validation guarantees
// n_bins <= 1 << ans_size_log <= 1 << MAX_ANS_BITS.
#[kani::proof]
fn proof_ans_weight_sum_cannot_overflow() {
  let ans_size_log: Bitlen = kani::any();
  kani::assume(ans_size_log <= MAX_ANS_BITS);

  let n_bins: Weight = kani::any();
  kani::assume(n_bins as u64 <= 1u64 << ans_size_log);

  // largest value a single weight can take
  let max_weight: u64 = (1u64 << ans_size_log) - 1 + 1;
  let max_sum = n_bins as u64 * max_weight;
  assert!(max_sum <= Weight::MAX as u64);
}
