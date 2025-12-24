use std::cmp::max;
use std::io::{BufReader, Error, ErrorKind, Read};

use crate::buf_read::BetterBufRead;

const DEFAULT_CAPACITY: usize = 8192;

/// An implementation of [`BetterBufRead`][crate::BetterBufRead] that wraps a
/// generic `Read`.
///
/// Use this to wrap things like files and network streams, but not data that's
/// already in memory.
/// This endows the `Read` with a buffer, unlocking a few benefits:
/// * better performance for repeated small reads
/// * doesn't lose data when passed around by programs that optimistically read
///   ahead
pub struct BetterBufReader<R: Read> {
  inner: R,
  buffer: Vec<u8>,
  desired_capacity: usize,
  pos: usize,
  filled: usize,
}

impl<R: Read> BetterBufRead for BetterBufReader<R> {
  fn fill_or_eof(&mut self, n_bytes: usize) -> std::io::Result<&[u8]> {
    // cycle the buffer if necessary
    let unfilled = self.buffer.len() - self.pos;
    let max_available = max(unfilled, self.desired_capacity);
    if n_bytes > max_available {
      return Err(Error::new(
        ErrorKind::InvalidInput,
        "requested reading more bytes than fit in buffer",
      ));
    }

    if n_bytes > unfilled {
      // we need to cycle the buffer
      // there's probably a more efficient way to do this
      for i in self.pos..self.filled {
        self.buffer[i - self.pos] = self.buffer[i];
      }
      self.buffer.truncate(self.desired_capacity);
      self.filled -= self.pos;
      self.pos = 0;
    }

    let target = self.pos + n_bytes;
    while self.filled < target {
      match self.inner.read(&mut self.buffer[self.filled..target]) {
        Ok(0) => break,
        Ok(n) => {
          self.filled += n;
        }
        Err(ref e) if e.kind() == ErrorKind::Interrupted => {}
        Err(e) => return Err(e),
      }
    }

    Ok(&self.buffer[self.pos..self.filled])
  }

  #[inline]
  fn consume(&mut self, n_bytes: usize) {
    self.pos += n_bytes;
  }

  #[inline]
  fn capacity(&self) -> Option<usize> {
    Some(self.desired_capacity)
  }

  fn resize_capacity(&mut self, desired: usize) {
    self.desired_capacity = desired;
    if desired >= self.filled {
      self.buffer.resize(desired, 0);
    }
  }
}

fn make_buffer(preloaded_data: &[u8], capacity: usize) -> (Vec<u8>, usize) {
  let mut buffer = vec![0; capacity];
  let filled = preloaded_data.len();
  buffer[0..filled].copy_from_slice(preloaded_data);
  (buffer, filled)
}

impl<R: Read> BetterBufReader<R> {
  /// Creates a `BetterBufReader` based on a `Read`.
  ///
  /// Providing preloaded data is optional, but can be useful if instantiating
  /// based on another abstraction that held a buffer and `Read`.
  ///
  /// Panics if `preloaded_data` is longer than `capacity`.
  pub fn new(preloaded_data: &[u8], inner: R, capacity: usize) -> Self {
    let (buffer, filled) = make_buffer(preloaded_data, capacity);
    Self {
      inner,
      buffer,
      desired_capacity: capacity,
      pos: 0,
      filled,
    }
  }

  /// Creates a `BetterBufReader` based on a `Read`, supplying sensible
  /// defaults.
  pub fn from_read_simple(inner: R) -> Self {
    Self::new(&[], inner, DEFAULT_CAPACITY)
  }

  /// Creates a `BetterBufReader` based on a `BufReader`.
  ///
  /// Panics if the `BufReader`'s buffer is longer than `capacity`.
  pub fn from_buf_reader(br: BufReader<R>, capacity: usize) -> Self {
    let (buffer, filled) = make_buffer(br.buffer(), capacity);
    Self {
      inner: br.into_inner(),
      buffer,
      desired_capacity: capacity,
      pos: 0,
      filled,
    }
  }

  /// Returns the inner `Read`, dropping the `BetterBufReader` and its buffer.
  ///
  /// To avoid losing data, be sure to read the last of the buffer before
  /// calling this.
  pub fn into_inner(self) -> R {
    self.inner
  }
}

#[cfg(test)]
mod tests {
  use std::io::Result;

  use super::*;

  use crate::BetterBufRead;

  #[test]
  fn test_better_buf_reader() -> Result<()> {
    let inner = (0..100_u8).skip(2).collect::<Vec<_>>();
    let mut reader = BetterBufReader::new(&[0, 1], inner.as_slice(), 5);

    // filling
    assert_eq!(reader.fill_or_eof(0)?, &[0, 1]);
    assert_eq!(reader.fill_or_eof(1)?, &[0, 1]);
    assert_eq!(reader.fill_or_eof(3)?, &[0, 1, 2]);
    assert!(reader.fill_or_eof(6).is_err());

    // consuming
    reader.consume(2);
    assert_eq!(reader.fill_or_eof(0)?, &[2]);
    assert_eq!(reader.fill_or_eof(2)?, &[2, 3]);
    assert_eq!(reader.fill_or_eof(5)?, &[2, 3, 4, 5, 6]);

    // resizing larger
    assert_eq!(reader.capacity(), Some(5));
    reader.resize_capacity(7);
    assert_eq!(reader.capacity(), Some(7));
    assert_eq!(
      reader.fill_or_eof(7)?,
      &[2, 3, 4, 5, 6, 7, 8]
    );

    // resizing smaller
    reader.resize_capacity(2);
    assert_eq!(reader.capacity(), Some(2));
    assert_eq!(
      reader.fill_or_eof(0)?,
      &[2, 3, 4, 5, 6, 7, 8]
    );
    reader.consume(6);
    assert_eq!(reader.fill_or_eof(2)?, &[8, 9]);

    // getting the read back
    assert_eq!(
      reader.into_inner(),
      &(10..100).collect::<Vec<_>>()
    );

    Ok(())
  }
}
