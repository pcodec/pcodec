use std::cmp::min;
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

#[inline]
pub unsafe fn read_uint_at<U: ReadWriteUint, const MAX_U64S: usize>(
  src: &[u8],
  mut byte_idx: usize,
  bits_past_byte: Bitlen,
  n: Bitlen,
) -> U {
  // Q: Why is this fast?
  // A: The 0..MAX_U64S can be unrolled at compile time, allowing really
  // fast SIMD stuff at least in the MAX_U64S=1 (most common) case.
  //
  // Q: Why does this work?
  // A: We set MAX_U64S so that,
  //    0  to 57  bit reads -> 1 u64
  //    58 to 113 bit reads -> 2 u64's
  //    113 to 128 bit reads -> 3 u64's
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

  for _ in 0..MAX_U64S - 1 {
    res |= U::from_u64(u64_at(src, byte_idx)) << processed;
    processed += 64;
    byte_idx += 8;
  }

  if MAX_U64S * 64 <= U::BITS as usize {
    bits::lowest_bits_fast(res, n)
  } else {
    bits::lowest_bits(res, n)
  }
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
    self.bits_past_byte = 0;
    Ok(&self.src[byte_idx..new_byte_idx])
  }

  pub unsafe fn read_uint<U: ReadWriteUint>(&mut self, n: Bitlen) -> U {
    self.refill();
    let res = match U::MAX_U64S {
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
      3 => read_uint_at::<U, 3>(
        self.src,
        self.stale_byte_idx,
        self.bits_past_byte,
        n,
      ),
      0 => panic!("[BitReader] data type cannot have 0 bits"),
      _ => panic!(
        "[BitReader] data type too large (extra u64's {} > 2)",
        U::MAX_U64S
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

  fn build<'a>(&'a mut self) -> io::Result<BitReader<'a>> {
    // could make n_bytes configurably smaller if it matters for some reason
    let n_bytes_to_read = self.padding;
    if !self.reached_eof {
      self.inner.fill_or_eof(n_bytes_to_read)?;
      let inner_bytes = self.inner.buffer();

      if inner_bytes.len() < n_bytes_to_read {
        self.reached_eof = true;
        self.eof_buffer = vec![0; inner_bytes.len() + self.padding];
        self.eof_buffer[..inner_bytes.len()].copy_from_slice(inner_bytes);
      }
    }

    let src = if self.reached_eof {
      &self.eof_buffer[self.bytes_into_eof_buffer..]
    } else {
      self.inner.buffer()
    };

    // we've reached the end of file buffer
    let unpadded_bytes = if self.reached_eof {
      self.eof_buffer.len() - self.padding - self.bytes_into_eof_buffer
    } else {
      src.len()
    };
    let bits_past_byte = self.bits_past_byte;
    Ok(BitReader::new(
      src,
      unpadded_bytes,
      bits_past_byte,
    ))
  }

  pub fn into_inner(self) -> R {
    self.inner
  }

  fn update(&mut self, final_bit_idx: usize) {
    let bytes_consumed = final_bit_idx / 8;
    self.inner.consume(bytes_consumed);
    if self.reached_eof {
      self.bytes_into_eof_buffer += bytes_consumed;
    }
    self.bits_past_byte = final_bit_idx as Bitlen % 8;
  }

  pub fn with_reader<Y, F: FnOnce(&mut BitReader) -> PcoResult<Y>>(
    &mut self,
    f: F,
  ) -> PcoResult<Y> {
    let mut reader = self.build()?;
    let res = f(&mut reader)?;
    let final_bit_idx = reader.bit_idx_safe()?;
    self.update(final_bit_idx);
    Ok(res)
  }
}

pub fn ensure_buf_read_capacity<R: BetterBufRead>(src: &mut R, required: usize) {
  if let Some(current_capacity) = src.capacity() {
    if current_capacity < required {
      src.resize_capacity(required);
    }
  }
}

#[cfg(test)]
mod tests {
  use crate::constants::OVERSHOOT_PADDING;
  use crate::errors::{ErrorKind, PcoResult};

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
    let mut reader_builder = BitReaderBuilder::new(src.as_slice(), 4 + OVERSHOOT_PADDING, 1);
    reader_builder.with_reader(|reader| unsafe {
      assert_eq!(&reader.src[0..4], &vec![0, 1, 2, 3]);
      assert_eq!(reader.bit_idx(), 1);
      assert_eq!(reader.read_usize(16), 1 << 7); // not 1 << 8, because we started at bit_idx 1
      Ok(())
    })?;
    reader_builder.with_reader(|reader| unsafe {
      assert_eq!(&reader.src[0..4], &vec![2, 3, 4, 5]);
      assert_eq!(reader.bit_idx(), 1);
      assert_eq!(reader.read_usize(7), 1);
      assert_eq!(reader.bit_idx(), 8);
      assert_eq!(reader.read_aligned_bytes(3)?, &vec![3, 4, 5]);
      Ok(())
    })?;
    let err = reader_builder
      .with_reader(|reader| unsafe {
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
