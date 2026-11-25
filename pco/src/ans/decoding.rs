use crate::ans::spec::Spec;
use crate::ans::AnsState;
use crate::constants::Bitlen;

// Using smaller types to reduce the memory footprint of Node. This improves
// performance when the table gets large, likely due to fewer cache misses.
// Both these values fit within u16 cleanly:
// * next_state_idx_base < 2^16 since we encode ANS table size log2 with 4 bits
// * offset_bits <= the largest number size, currently 64 bits
//
// Also note that we include the bin's offset_bits in the node, even though it
// isn't a part of ANS coding; it just fits. This is another performance hack.
//
// We pack both into a u32 with next_state_idx_base in the low half, so that
// decoding loads the whole node at once and adds the base to the ANS value
// without a separate instruction to extract it.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Node(u32);

impl Node {
  fn new(next_state_idx_base: u16, offset_bits: u16) -> Self {
    Self(next_state_idx_base as u32 | (offset_bits as u32) << 16)
  }

  #[inline]
  pub fn next_state_idx_base(self) -> AnsState {
    self.0 as u16 as AnsState
  }

  #[inline]
  pub fn offset_bits(self) -> Bitlen {
    self.0 >> 16
  }
}

// bits_to_read (<= 16, the max ANS table size log2) lives in its own table
// rather than in Node. It's on the critical path of decoding, since each
// interleaved state must know how many bits it reads before the next state can
// find its own bits. Loading it directly from a byte array is faster than
// extracting it from a node.
#[derive(Clone, Debug)]
pub struct Decoder {
  pub bits_to_read: Vec<u8>,
  pub nodes: Vec<Node>,
}

impl Decoder {
  pub fn new(spec: &Spec, bin_offset_bits: &[Bitlen]) -> Self {
    let table_size = spec.table_size();
    let mut all_bits_to_read = Vec::with_capacity(table_size);
    let mut nodes = Vec::with_capacity(table_size);
    // x_s from Jarek Duda's paper
    let mut symbol_x_s = spec.symbol_weights.clone();
    for &symbol in &spec.state_symbols {
      let next_state_base = symbol_x_s[symbol as usize] as AnsState;
      let bits_to_read = next_state_base.leading_zeros() - (table_size as AnsState).leading_zeros();
      let next_state_base = next_state_base << bits_to_read;
      // In a degenerate case there are 0 bins, but the tANS table always has at
      // least one node, so we handle that by using 0 offset bits.
      let offset_bits = bin_offset_bits.get(symbol as usize).cloned().unwrap_or(0);
      all_bits_to_read.push(bits_to_read as u8);
      nodes.push(Node::new(
        (next_state_base - table_size as AnsState) as u16,
        offset_bits as u16,
      ));
      symbol_x_s[symbol as usize] += 1;
    }

    Self {
      bits_to_read: all_bits_to_read,
      nodes,
    }
  }
}
