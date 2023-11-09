use std::cmp::min;
use std::{io, mem};
use better_io::{BetterBufRead};

use crate::bits;
use crate::constants::Bitlen;
use crate::errors::{PcoError, PcoResult};
use crate::read_write_uint::ReadWriteUint;

// Q: Why u64?
// A: It's the largest data type most instruction sets have support for (and
//    can do few-cycle/SIMD ops on). e.g. even 32-bit wasm has 64-bit ints and
//    opcodes.
#[inline]
pub fn u64_at(src: &[u8], byte_idx: usize) -> u64 {
  let raw_bytes = unsafe { *(src.as_ptr().add(byte_idx) as *const [u8; 8]) };
  u64::from_le_bytes(raw_bytes)
}

#[inline]
pub fn read_uint_at<U: ReadWriteUint, const MAX_EXTRA_U64S: usize>(
  src: &[u8],
  mut byte_idx: usize,
  bits_past_byte: Bitlen,
  n: Bitlen,
) -> U {
  // Q: Why is this fast?
  // A: The 0..MAX_EXTRA_U64S can be unrolled at compile time and interact
  //    freely with an outer loop, allowing really fast SIMD stuff.
  //
  // Q: Why does this work?
  // A: We set MAX_EXTRA_U64S so that,
  //    0  to 57  bit reads -> 0 extra u64's
  //    58 to 113 bit reads -> 1 extra u64's
  //    113 to 128 bit reads -> 2 extra u64's
  //    During the 1st u64 (prior to the loop), we read all bytes from the
  //    current u64. Due to our bit packing, up to the first 7 of these may
  //    be useless, so we can read up to (64 - 7) = 57 bits safely from a
  //    single u64. We right shift by only up to 7 bits, which is safe.
  //
  //    For the 2nd u64, we skip only 7 bytes forward. This will overlap with
  //    the 1st u64 by 1 byte, which seems useless, but allows us to avoid one
  //    nasty case: left shifting by U::BITS (a panic). This could happen e.g.
  //    with 64-bit reads when we start out byte-aligned (bits_past_byte=0).
  //
  //    For the 3rd u64 and onward, we skip 8 bytes forward. Due to how we
  //    handled the 2nd u64, the most we'll ever need to shift by is
  //    U::BITS - 8, which is safe.
  let mut res = U::from_u64(u64_at(src, byte_idx) >> bits_past_byte);
  let mut processed = min(n, 56 - bits_past_byte);
  byte_idx += 7;

  for _ in 0..MAX_EXTRA_U64S {
    res |= U::from_u64(u64_at(src, byte_idx)) << processed;
    processed += 64;
    byte_idx += 8;
  }

  bits::lowest_bits(res, n)
}

struct BitReaderTombstone {
  bytes_read: usize,
  bits_past_byte: Bitlen,
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

  pub fn bit_idx(&self) -> usize {
    self.stale_byte_idx * 8 + self.bits_past_byte as usize
  }

  fn byte_idx(&self) -> usize {
    self.bit_idx() / 8
  }

  // Returns the reader's current byte index. Will return an error if the
  // reader is at a misaligned position.
  fn aligned_byte_idx(&self) -> PcoResult<usize> {
    if self.bits_past_byte % 8 == 0 {
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
    Ok(&self.src[byte_idx..new_byte_idx])
  }

  pub fn read_uint<U: ReadWriteUint>(&mut self, n: Bitlen) -> U {
    self.refill();
    let res = match U::MAX_EXTRA_U64S {
      0 => read_uint_at::<U, 0>(
        self.src,
        self.stale_byte_idx,
        self.bits_past_byte,
        n,
      ),
      1 => read_uint_at::<U, 1>(
        self.src,
        self.stale_byte_idx,
        self.bits_past_byte,
        n,
      ),
      2 => read_uint_at::<U, 2>(
        self.src,
        self.stale_byte_idx,
        self.bits_past_byte,
        n,
      ),
      _ => panic!(
        "[BitReader] data type too large (extra u64's {} > 2)",
        U::MAX_EXTRA_U64S
      ),
    };
    self.consume(n);
    res
  }
  pub fn read_usize(&mut self, n: Bitlen) -> usize {
    self.read_uint(n)
  }

  pub fn read_bitlen(&mut self, n: Bitlen) -> Bitlen {
    self.read_uint(n)
  }

  pub fn check_in_bounds(&self) -> PcoResult<()> {
    let bit_idx = self.bit_idx();
    if bit_idx > self.unpadded_bit_size {
      return Err(PcoError::insufficient_data(format!(
        "[BitReader] out of bounds at bit {} / {}",
        bit_idx, self.unpadded_bit_size
      )));
    }
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

  fn close(mut self) -> PcoResult<BitReaderTombstone> {
    self.check_in_bounds()?;
    self.refill();

    Ok(BitReaderTombstone {
      bytes_read: self.stale_byte_idx,
      bits_past_byte: self.bits_past_byte,
    })
  }
}

pub struct BitReaderBuilder<R: BetterBufRead> {
  padding: usize,
  inner: R,
  eof_buffer: Vec<u8>,
  reached_eof: bool,
  bytes_into_eof_buffer: usize,
  bits_past_byte: Bitlen,
}

impl<R: BetterBufRead> BitReaderBuilder<R> {
  pub fn new(inner: R, padding: usize, bits_past_byte: Bitlen) -> Self {
    Self {
      padding,
      inner,
      eof_buffer: vec![],
      reached_eof: false,
      bytes_into_eof_buffer: 0,
      bits_past_byte,
    }
  }

  fn read(&mut self, n_bytes: usize) -> io::Result<&[u8]> {
    if !self.reached_eof {
      self.inner.fill_or_eof(n_bytes)?;
      let inner_bytes = self.inner.buffer();

      if inner_bytes.len() >= n_bytes {
        return Ok(inner_bytes);
      } else {
        self.reached_eof = true;
        self.eof_buffer = vec![0; inner_bytes.len() + self.padding];
        self.eof_buffer[..inner_bytes.len()].copy_from_slice(inner_bytes);
      }
    }

    // we've reached the end of file buffer
    Ok(&self.eof_buffer[self.bytes_into_eof_buffer..])
  }

  fn build(&mut self) -> io::Result<BitReader> {
    let unpadded_bytes = if self.reached_eof {
      self.eof_buffer.len() - self.padding - self.bytes_into_eof_buffer
    } else {
      0
    };
    let bits_past_byte = self.bits_past_byte;
    let src = self.read(self.padding)?;
    Ok(BitReader::new(src, unpadded_bytes, bits_past_byte))
  }

  pub fn into_inner(self) -> R {
    self.inner
  }

  fn update(&mut self, tombstone: BitReaderTombstone) {
    self.inner.consume(tombstone.bytes_read);
    if self.reached_eof {
      self.bytes_into_eof_buffer += tombstone.bytes_read;
    }
    self.bits_past_byte = tombstone.bits_past_byte;
  }

  pub fn with_reader<Y, F: FnOnce(&mut BitReader) -> PcoResult<Y>>(&mut self, f: F) -> PcoResult<Y> {
    let mut reader = self.build()?;
    let res = f(&mut reader)?;
    let tombstone = reader.close()?;
    self.update(tombstone);
    Ok(res)
  }

  pub fn bits_past_byte(&self) -> Bitlen {
    self.bits_past_byte
  }
}

#[cfg(test)]
mod tests {
  use crate::errors::PcoResult;

  use super::*;

  // I find little endian confusing, hence all the comments.
  // All the bytes in comments are written backwards,
  // e.g. 00000001 = 2^7

  #[test]
  fn test_bit_reader() -> PcoResult<()> {
    // 10010001 01100100 00000000 11111111 10000010
    let src = vec![137, 38, 0, 255, 65, 0, 0, 0, 0];
    let mut reader = BitReader::new(&src, 5, 0);

    assert_eq!(reader.read_bitlen(4), 9);
    assert!(reader.read_aligned_bytes(1).is_err());
    assert_eq!(reader.read_bitlen(4), 8);
    assert_eq!(reader.read_aligned_bytes(1)?, vec![38]);
    assert_eq!(reader.read_usize(15), 255 + 65 * 256);
    reader.drain_empty_byte("should be empty")?;
    assert_eq!(reader.aligned_byte_idx()?, 5);
    Ok(())
  }
}
