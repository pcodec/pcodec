use std::io;

use better_io::BetterBufRead;

use crate::bits;
use crate::constants::Bitlen;
use crate::errors::{PcoError, PcoResult};
use crate::read_write_uint::ReadWriteUint;

// Q: Why u64?
// A: It's the largest data type most instruction sets have support for (and
//    can do few-cycle/SIMD ops on). e.g. even 32-bit wasm has 64-bit ints and
//    opcodes.
#[inline]
pub unsafe fn u64_at(src: &[u8], byte_idx: usize) -> u64 {
  let raw_bytes = *(src.as_ptr().add(byte_idx) as *const [u8; 8]);
  u64::from_le_bytes(raw_bytes)
}

// Q: Why is there also a u32 version?
// A: This allows for better use of SIMD bandwidth when reading smaller latent
//    types compared to the u64 version.
#[inline]
pub unsafe fn u32_at(src: &[u8], byte_idx: usize) -> u32 {
  let raw_bytes = *(src.as_ptr().add(byte_idx) as *const [u8; 4]);
  u32::from_le_bytes(raw_bytes)
}

#[inline]
pub unsafe fn read_uint_at<U: ReadWriteUint, const READ_BYTES: usize>(
  src: &[u8],
  byte_idx: usize,
  bits_past_byte: Bitlen,
  n: Bitlen,
) -> U {
  // Q: Why is this fast?
  // A: The compiler removes branching allowing for fast SIMD.
  //
  // Q: Why does this work?
  // A: We set READ_BYTES so that,
  //    0  to 25  bit reads -> 4 bytes (1 u32)
  //    26 to 57  bit reads -> 8 bytes (1 u64)
  //    58 to 113 bit reads -> 15 bytes (almost 2 u64's)
  //    For the 1st u64, we read all bytes from the current u64. Due to our bit
  //    packing, up to the first 7 of these may be useless, so we can read up
  //    to (64 - 7) = 57 bits safely from a single u64. We right shift by only
  //    up to 7 bits, which is safe.
  //
  //    For the 2nd u64, we skip only 7 bytes forward. This will overlap with
  //    the 1st u64 by 1 byte, which seems useless, but allows us to avoid one
  //    nasty case: left shifting by U::BITS (a panic). This could happen e.g.
  //    with 64-bit reads when we start out byte-aligned (bits_past_byte=0).
  //
  //    For the 3rd u64 and onward (currently not implemented), we skip 8 bytes
  //    forward. Due to how we handled the 2nd u64, the most we'll ever need to
  //    shift by is U::BITS - 8, which is safe.

  match READ_BYTES {
    4 => read_u32_at(src, byte_idx, bits_past_byte, n),
    8 => read_u64_at(src, byte_idx, bits_past_byte, n),
    15 => read_almost_u64x2_at(src, byte_idx, bits_past_byte, n),
    _ => unreachable!("invalid read bytes: {}", READ_BYTES),
  }
}

#[inline]
unsafe fn read_u32_at<U: ReadWriteUint>(
  src: &[u8],
  byte_idx: usize,
  bits_past_byte: Bitlen,
  n: Bitlen,
) -> U {
  debug_assert!(n <= 25);
  U::from_u32(bits::lowest_bits_fast(
    u32_at(src, byte_idx) >> bits_past_byte,
    n,
  ))
}

#[inline]
unsafe fn read_u64_at<U: ReadWriteUint>(
  src: &[u8],
  byte_idx: usize,
  bits_past_byte: Bitlen,
  n: Bitlen,
) -> U {
  debug_assert!(n <= 57);
  U::from_u64(bits::lowest_bits_fast(
    u64_at(src, byte_idx) >> bits_past_byte,
    n,
  ))
}

#[inline]
unsafe fn read_almost_u64x2_at<U: ReadWriteUint>(
  src: &[u8],
  byte_idx: usize,
  bits_past_byte: Bitlen,
  n: Bitlen,
) -> U {
  debug_assert!(n <= 113);
  let first_word = U::from_u64(u64_at(src, byte_idx) >> bits_past_byte);
  let processed = 56 - bits_past_byte;
  let second_word = U::from_u64(u64_at(src, byte_idx + 7)) << processed;
  bits::lowest_bits(first_word | second_word, n)
}

pub struct BitReader<'a> {
  pub src: &'a [u8],
  unpadded_bit_size: usize,

  pub stale_byte_idx: usize,  // in current stream
  pub bits_past_byte: Bitlen, // in current stream
}

impl<'a> BitReader<'a> {
  pub fn new(src: &'a [u8], unpadded_byte_size: usize, bits_past_byte: Bitlen) -> Self {
    Self {
      src,
      unpadded_bit_size: unpadded_byte_size * 8,
      stale_byte_idx: 0,
      bits_past_byte,
    }
  }

  #[inline]
  pub fn bit_idx(&self) -> usize {
    self.stale_byte_idx * 8 + self.bits_past_byte as usize
  }

  fn byte_idx(&self) -> usize {
    self.bit_idx() / 8
  }

  // Returns the reader's current byte index. Will return an error if the
  // reader is at a misaligned position.
  fn aligned_byte_idx(&self) -> PcoResult<usize> {
    if self.bits_past_byte.is_multiple_of(8) {
      Ok(self.byte_idx())
    } else {
      Err(PcoError::invalid_argument(format!(
        "cannot get aligned byte index on misaligned bit reader (byte {} + {} bits)",
        self.stale_byte_idx, self.bits_past_byte,
      )))
    }
  }

  #[inline]
  fn refill(&mut self) {
    self.stale_byte_idx += (self.bits_past_byte / 8) as usize;
    self.bits_past_byte %= 8;
  }

  #[inline]
  fn consume(&mut self, n: Bitlen) {
    self.bits_past_byte += n;
  }

  pub fn read_aligned_bytes(&mut self, n: usize) -> PcoResult<&'a [u8]> {
    let byte_idx = self.aligned_byte_idx()?;
    let new_byte_idx = byte_idx + n;
    self.stale_byte_idx = new_byte_idx;
    self.bits_past_byte = 0;
    Ok(&self.src[byte_idx..new_byte_idx])
  }

  pub unsafe fn read_uint<U: ReadWriteUint>(&mut self, n: Bitlen) -> U {
    self.refill();
    let res = match U::MAX_BYTES {
      1..=4 => read_uint_at::<U, 4>(
        self.src,
        self.stale_byte_idx,
        self.bits_past_byte,
        n,
      ),
      5..=8 => read_uint_at::<U, 8>(
        self.src,
        self.stale_byte_idx,
        self.bits_past_byte,
        n,
      ),
      9..=15 => read_uint_at::<U, 15>(
        self.src,
        self.stale_byte_idx,
        self.bits_past_byte,
        n,
      ),
      _ => unreachable!(
        "[BitReader] unsupported max bytes: {}",
        U::MAX_BYTES
      ),
    };
    self.consume(n);
    res
  }

  pub unsafe fn read_usize(&mut self, n: Bitlen) -> usize {
    self.read_uint(n)
  }

  pub unsafe fn read_bitlen(&mut self, n: Bitlen) -> Bitlen {
    self.read_uint(n)
  }

  pub unsafe fn read_bool(&mut self) -> bool {
    self.read_uint::<u32>(1) > 0
  }

  // checks in bounds and returns bit idx
  #[inline]
  fn bit_idx_safe(&self) -> PcoResult<usize> {
    let bit_idx = self.bit_idx();
    if bit_idx > self.unpadded_bit_size {
      return Err(PcoError::insufficient_data(format!(
        "[BitReader] out of bounds at bit {} / {}",
        bit_idx, self.unpadded_bit_size
      )));
    }
    Ok(bit_idx)
  }

  pub fn check_in_bounds(&self) -> PcoResult<()> {
    self.bit_idx_safe()?;
    Ok(())
  }

  // Seek to the end of the byte, asserting it's all 0.
  // Used to terminate each section of the file, since they
  // always start and end byte-aligned.
  pub fn drain_empty_byte(&mut self, message: &str) -> PcoResult<()> {
    self.check_in_bounds()?;
    self.refill();
    if self.bits_past_byte != 0 {
      if (self.src[self.stale_byte_idx] >> self.bits_past_byte) > 0 {
        return Err(PcoError::corruption(message));
      }
      self.consume(8 - self.bits_past_byte);
    }
    Ok(())
  }
}

// High level idea behing a BitReaderBuilder: without fail, produce BitReaders
// with at least the requested number of bytes available. The user's provided
// BetterBufRead might not have enough capacity, so when we get to the end, we
// populate an eof_buffer with the last bit of data and some extra padding.
pub struct BitReaderBuilder<R: BetterBufRead> {
  inner: R,
  eof_buffer: Vec<u8>,
  bytes_into_eof_buffer: usize,
  bits_past_byte: Bitlen,
}

impl<R: BetterBufRead> BitReaderBuilder<R> {
  pub fn new(inner: R, bits_past_byte: Bitlen) -> Self {
    Self {
      inner,
      eof_buffer: vec![],
      bytes_into_eof_buffer: 0,
      bits_past_byte,
    }
  }

  fn build<'a>(&'a mut self, n_bytes: usize) -> io::Result<BitReader<'a>> {
    ensure_buf_read_capacity(&mut self.inner, n_bytes);

    let inner_src = self.inner.fill_or_eof(n_bytes)?;
    let src = if inner_src.len() >= n_bytes {
      // first, try to use the inner BetterBufRead if it has enough data left
      inner_src
    } else if self
      .eof_buffer
      .len()
      .saturating_sub(self.bytes_into_eof_buffer)
      >= n_bytes
    {
      // if not, try to use the current eof_buffer if it has enough data left
      &self.eof_buffer[self.bytes_into_eof_buffer..]
    } else {
      // if neither has enough capacity, make a new eof buffer with at least 2x
      // the requested bytes to avoid any antagonistic O(n^2) behavior
      self.eof_buffer = vec![0; 2 * n_bytes];
      self.eof_buffer[..inner_src.len()].copy_from_slice(inner_src);
      self.bytes_into_eof_buffer = 0;
      &self.eof_buffer
    };

    Ok(BitReader::new(
      src,
      inner_src.len(),
      self.bits_past_byte,
    ))
  }

  pub fn into_inner(self) -> R {
    self.inner
  }

  fn update(&mut self, reader_bit_idx: usize) {
    let bytes_consumed = reader_bit_idx / 8;
    self.inner.consume(bytes_consumed);
    self.bytes_into_eof_buffer += bytes_consumed;
    self.bits_past_byte = reader_bit_idx as Bitlen % 8;
  }

  /// Makes a BitReader that is guaranteed to be able to read at least
  /// `self.padding` bytes. Reading more than that many bytes can cause a
  /// segfault (very bad!).
  pub fn with_reader<Y, F: FnOnce(&mut BitReader) -> PcoResult<Y>>(
    &mut self,
    n_bytes: usize,
    f: F,
  ) -> PcoResult<Y> {
    let mut reader = self.build(n_bytes)?;
    let orig_bit_idx = reader.bit_idx();
    let res = f(&mut reader)?;
    let final_bit_idx = reader.bit_idx_safe()?;
    debug_assert!(final_bit_idx - orig_bit_idx <= 8 * n_bytes);
    self.update(final_bit_idx);
    Ok(res)
  }
}

fn ensure_buf_read_capacity<R: BetterBufRead>(src: &mut R, required: usize) {
  if let Some(current_capacity) = src.capacity() {
    if current_capacity < required {
      src.resize_capacity(required);
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::{
    constants::OVERSHOOT_PADDING,
    errors::{ErrorKind, PcoResult},
  };

  use super::*;

  // I find little endian confusing, hence all the comments.
  // All the bytes in comments are written backwards,
  // e.g. 00000001 = 2^7

  #[test]
  fn test_bit_reader() -> PcoResult<()> {
    // 10010001 01100100 00000000 11111111 10000010
    let mut src = vec![137, 38, 255, 65];
    src.resize(20, 0);
    let mut reader = BitReader::new(&src, 5, 0);

    unsafe {
      assert_eq!(reader.read_bitlen(4), 9);
      assert!(reader.read_aligned_bytes(1).is_err());
      assert_eq!(reader.read_bitlen(4), 8);
      assert_eq!(reader.read_aligned_bytes(1)?, vec![38]);
      assert_eq!(reader.read_usize(15), 255 + 65 * 256);
      reader.drain_empty_byte("should be empty")?;
      assert_eq!(reader.aligned_byte_idx()?, 4);
    }
    Ok(())
  }

  #[test]
  fn test_bit_reader_builder() -> PcoResult<()> {
    let src = (0..7).collect::<Vec<_>>();
    let capacity = 4 + OVERSHOOT_PADDING;
    let mut reader_builder = BitReaderBuilder::new(src.as_slice(), 1);
    reader_builder.with_reader(capacity, |reader| unsafe {
      assert_eq!(&reader.src[0..4], &vec![0, 1, 2, 3]);
      assert_eq!(reader.bit_idx(), 1);
      assert_eq!(reader.read_usize(16), 1 << 7); // not 1 << 8, because we started at bit_idx 1
      Ok(())
    })?;
    reader_builder.with_reader(capacity, |reader| unsafe {
      assert_eq!(&reader.src[0..4], &vec![2, 3, 4, 5]);
      assert_eq!(reader.bit_idx(), 1);
      assert_eq!(reader.read_usize(7), 1);
      assert_eq!(reader.bit_idx(), 8);
      assert_eq!(reader.read_aligned_bytes(3)?, &vec![3, 4, 5]);
      Ok(())
    })?;
    let err = reader_builder
      .with_reader(capacity, |reader| unsafe {
        assert!(reader.src.len() >= 4); // because of padding
        reader.read_usize(9); // this overshoots the end of the data by 1 bit
        Ok(())
      })
      .unwrap_err();
    assert!(matches!(
      err.kind,
      ErrorKind::InsufficientData
    ));

    Ok(())
  }
}
