use crate::data_types::UnsignedLike;

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
  let mut res = 0;
  for (i, &b) in bits.iter().enumerate() {
    if b {
      res |= 1 << i;
    }
  }
  res
}

pub fn usize_to_bits(x: usize, n: usize) -> Vec<bool> {
  let mut res = Vec::with_capacity(n);
  for i in 0..n {
    res.push((x >> i) & 1 > 0);
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

// This bumpy log gives a more accurate average number of offset bits used.
pub fn avg_offset_bits<U: UnsignedLike>(lower: U, upper: U, gcd: U) -> f64 {
  (((upper - lower) / gcd).to_f64() + 1.0).log2().ceil()
}

// The true Huffman cost of course depends on the tree. We can statistically
// model this cost and get slightly different bumpy log formulas,
// but I haven't found
// anything that beats a simple log. Plus it's computationally cheap.
pub fn avg_depth_bits(weight: usize, total_weight: usize) -> f64 {
  (total_weight as f64 / weight as f64).log2()
}

pub const fn ceil_div(x: usize, divisor: usize) -> usize {
  (x + divisor - 1) / divisor
}

pub fn words_to_bytes(words: &[usize]) -> Vec<u8> {
  // We can't just transmute because many machines are little-endian.
  words
    .iter()
    .flat_map(|w| w.to_le_bytes())
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
    assert_eq!(avg_offset_bits(10_u32, 13_u32, 1), 2.0);
    assert_eq!(avg_offset_bits(10_u32, 13_u32, 3), 1.0);
    assert_eq!(avg_offset_bits(10_u32, 19_u32, 3), 2.0);
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
}
