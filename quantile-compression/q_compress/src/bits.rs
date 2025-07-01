use crate::constants::WORD_SIZE;
use crate::data_types::UnsignedLike;

pub const BASE_BIT_MASK: usize = 1 << (WORD_SIZE - 1);
const BUMPY_MAX_K: usize = 129;
const BUMPY_LOG_TABLE: [(f64, f64); BUMPY_MAX_K] = {
  let mut res = [(0.0, 0.0); BUMPY_MAX_K];
  let mut k = 0;
  let exp_bit = 1_u64 << 52;
  while k < BUMPY_MAX_K {
    let mem_repr_exp = exp_bit * (k + 1023 + 1) as u64;
    res[k] = ((k + 2) as f64, unsafe {
      std::mem::transmute::<u64, f64>(mem_repr_exp)
    });
    k += 1;
  }
  res
};

pub fn bit_from_word(word: usize, j: usize) -> bool {
  (word & (BASE_BIT_MASK >> j)) > 0
}

pub fn bits_to_bytes(bits: Vec<bool>) -> Vec<u8> {
  let mut res = Vec::new();
  let mut i = 0;
  while i < bits.len() {
    let mut byte = 0_u8;
    for _ in 0..8 {
      byte <<= 1;
      if i < bits.len() {
        if bits[i] {
          byte |= 1;
        }
        i += 1;
      }
    }
    res.push(byte);
  }
  res
}

pub fn bytes_to_bits(bytes: Vec<u8>) -> Vec<bool> {
  let mut res = Vec::with_capacity(bytes.len() * 8);
  for b in bytes {
    for i in 0_usize..8 {
      res.push(b & (1 << (7 - i)) > 0);
    }
  }
  res
}

pub fn bits_to_usize(bits: &[bool]) -> usize {
  bits_to_usize_truncated(bits, bits.len())
}

pub fn bits_to_usize_truncated(bits: &[bool], max_depth: usize) -> usize {
  if max_depth < 1 {
    return 0;
  }

  let pow = 1_usize << (max_depth - 1);
  let mut res = 0;
  for (i, bit) in bits.iter().enumerate() {
    if *bit {
      res |= pow >> i;
    }
  }
  res
}

pub fn usize_truncated_to_bits(x: usize, max_depth: usize) -> Vec<bool> {
  if max_depth < 1 {
    return Vec::new();
  }

  let mut res = Vec::with_capacity(max_depth);
  for i in 0..max_depth {
    res.push((x >> (max_depth - i - 1)) & 1 > 0);
  }
  res
}

pub fn bits_to_string(bits: &[bool]) -> String {
  return bits
    .iter()
    .map(|b| if *b { "1" } else { "0" })
    .collect::<Vec<&str>>()
    .join("");
}

#[inline]
fn bumpy_log(x: f64) -> f64 {
  let k = x.log2() as usize;
  let (base, exp) = BUMPY_LOG_TABLE[k];
  base - exp / x
}

// This bumpy log gives a more accurate average number of offset bits used.
pub fn avg_offset_bits<U: UnsignedLike>(lower: U, upper: U, gcd: U) -> f64 {
  bumpy_log(((upper - lower) / gcd).to_f64() + 1.0)
}

// The true Huffman cost of course depends on the tree. We can statistically
// model this cost and get slightly different bumpy log formulas,
// but I haven't found
// anything that beats a simple log. Plus it's computationally cheap.
pub fn avg_depth_bits(weight: usize, total_weight: usize) -> f64 {
  (total_weight as f64 / weight as f64).log2()
}

pub fn ceil_div(x: usize, divisor: usize) -> usize {
  (x + divisor - 1) / divisor
}

pub fn words_to_bytes(words: &[usize]) -> Vec<u8> {
  // We can't just transmute because many machines are little-endian.
  words
    .iter()
    .flat_map(|w| w.to_be_bytes())
    .collect::<Vec<_>>()
}

pub fn bits_to_encode(max_number: usize) -> usize {
  ((max_number + 1) as f64).log2().ceil() as usize
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_bits_to_string() {
    assert_eq!(bits_to_string(&[]), "".to_string());
    assert_eq!(
      bits_to_string(&[true, false, false]),
      "100".to_string()
    );
  }

  #[test]
  fn test_depth_bits() {
    assert_eq!(avg_depth_bits(2, 2), 0.0);
    assert_eq!(avg_depth_bits(2, 4), 1.0);
    assert_eq!(avg_depth_bits(2, 8), 2.0);
    assert_eq!(avg_depth_bits(4, 8), 1.0);
  }

  #[test]
  fn test_avg_offset_bits() {
    assert_eq!(avg_offset_bits(0_u32, 0_u32, 1), 0.0);
    assert_eq!(avg_offset_bits(4_u32, 5_u32, 1), 1.0);
    assert!((avg_offset_bits(2_u32, 4_u32, 1) - 5.0 / 3.0).abs() < 1E-8);
    assert_eq!(avg_offset_bits(10_u32, 13_u32, 1), 2.0);
    assert_eq!(avg_offset_bits(10_u32, 13_u32, 3), 1.0);
    assert_eq!(avg_offset_bits(10_u32, 19_u32, 3), 2.0);
    assert!((avg_offset_bits(0_u64, 4_u64, 1) - 12.0 / 5.0).abs() < 1E-8);
  }

  #[test]
  fn test_bits_to_usize_truncated() {
    assert_eq!(bits_to_usize_truncated(&[], 0), 0);
    assert_eq!(bits_to_usize_truncated(&[true], 4), 8);
    assert_eq!(bits_to_usize_truncated(&[true], 3), 4);
    assert_eq!(
      bits_to_usize_truncated(&[true, false, true], 4),
      10
    );
    assert_eq!(
      bits_to_usize_truncated(&[true, false, true, true], 4),
      11
    );
  }

  #[test]
  fn test_bits_to_bytes_to_bits() {
    let bits_28 = vec![false, false, false, true, true, true, false, false];
    let byte_28 = bits_to_bytes(bits_28);
    assert_eq!(byte_28, vec![28]);

    let bits_28_128 = vec![false, false, false, true, true, true, false, false, true];
    let byte_28_128 = bits_to_bytes(bits_28_128);
    assert_eq!(byte_28_128, vec![28, 128]);
  }

  #[test]
  fn test_log_pows() {
    assert_eq!(BUMPY_LOG_TABLE[0], (2.0, 2.0));
    assert_eq!(BUMPY_LOG_TABLE[1], (3.0, 4.0));
    assert_eq!(BUMPY_LOG_TABLE[2], (4.0, 8.0));
    assert_eq!(BUMPY_LOG_TABLE[70], (72.0, 2.0_f64.powi(71)));
  }

  #[test]
  fn test_bumpy_log() {
    assert_eq!(bumpy_log(1.0), 0.0);
    assert_eq!(bumpy_log(2.0), 1.0);
    assert!((bumpy_log(3.0) - 1.667).abs() < 0.001);
    assert_eq!(bumpy_log(4.0), 2.0);
    assert!((bumpy_log(5.0) - 2.4).abs() < 0.001);
    assert_eq!(bumpy_log(2.0_f64.powi(128)), 128.0);
  }
}
