use crate::ans::spec::Spec;
use crate::ans::AnsState;
use crate::constants::Bitlen;

// Using smaller types to reduce the memory footprint of Node. This improves
// performance when the table gets large, likely due to fewer cache misses.
// All these values fit within u16 cleanly:
// * symbol < 2^15 since we encode n_bins with 15 bits
// * next_state_idx_base < 2^16 since we encode ANS table size log2 with 4 bits
// * offset_bits <= the largest number size, currently 64 bits
// * bits_to_read <= 16, the max ANS table size log2.
//
// Also note that we include the bin's offset_bits in the struct, even though it
// isn't a part of ANS coding; it just fits. We still have to look up the bin's
// lower bound from a separate table.  This is another performance hack.
#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct Node {
  pub symbol: u16,
  pub next_state_idx_base: u16,
  pub offset_bits: u16,
  pub bits_to_read: u16,
}

#[derive(Clone, Debug)]
pub struct Decoder {
  pub nodes: Vec<Node>,
}

impl Decoder {
  pub fn new(spec: &Spec, bin_offset_bits: &[Bitlen]) -> Self {
    let table_size = spec.table_size();
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
      nodes.push(Node {
        symbol: symbol as u16,
        next_state_idx_base: (next_state_base - table_size as AnsState) as u16,
        offset_bits: offset_bits as u16,
        bits_to_read: bits_to_read as u16,
      });
      symbol_x_s[symbol as usize] += 1;
    }

    Self { nodes }
  }
}
