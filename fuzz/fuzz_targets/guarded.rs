//! A caller buffer with poison bands on both sides.
//!
//! The C API writes through raw pointers into memory the caller owns, so the
//! failure that matters is a write landing outside the capacity that was
//! handed over. libFuzzer builds without a memory sanitizer by default, and a
//! few bytes past the end of a `Vec` usually land in slack the allocator
//! already owns -- silently. Poison bands make that write observable.

use std::ffi::c_void;

/// Bytes of poison on each side. Wide enough that a length mistake in units
/// (elements vs bytes) has to hit it, rather than 8 bytes that a single
/// oversized word could jump over.
const GUARD: usize = 64;
const POISON: u8 = 0xA5;

pub struct Guarded {
  /// Backed by `u64` so the payload pointer is 8-aligned: every pco number
  /// type has an alignment of at most 8, and handing over a misaligned
  /// destination would be the *harness* violating the C contract, producing a
  /// finding that says nothing about the library.
  words: Vec<u64>,
  payload_bytes: usize,
}

impl Guarded {
  pub fn new(payload_bytes: usize) -> Self {
    let total = 2 * GUARD + payload_bytes;
    Self {
      words: vec![u64::from_ne_bytes([POISON; 8]); total.div_ceil(8)],
      payload_bytes,
    }
  }

  /// Pointer to the start of the payload -- the only region the callee is
  /// allowed to touch.
  pub fn ptr(&mut self) -> *mut c_void {
    unsafe { (self.words.as_mut_ptr() as *mut u8).add(GUARD) as *mut c_void }
  }

  fn bytes(&self) -> &[u8] {
    unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.words.len() * 8) }
  }

  pub fn payload(&self) -> &[u8] {
    &self.bytes()[GUARD..GUARD + self.payload_bytes]
  }

  /// Panics if anything outside the payload changed. Checked byte by byte
  /// rather than word by word, so a one-byte overrun cannot hide in the
  /// rounding up to whole `u64`s.
  pub fn check(&self, what: &str) {
    let bytes = self.bytes();
    for (i, b) in bytes.iter().enumerate() {
      let inside = i >= GUARD && i < GUARD + self.payload_bytes;
      assert!(
        inside || *b == POISON,
        "{what}: byte {i} of the guard band was overwritten with {b:#04x} \
         (payload is {} bytes at offset {GUARD})",
        self.payload_bytes
      );
    }
  }
}
