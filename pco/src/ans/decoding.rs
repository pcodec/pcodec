use crate::ans::spec::Spec;
use crate::ans::{AnsState, CompactAnsState, CompactSymbol};
use crate::data_types::Latent;
use crate::metadata::Bin;

// Using smallar types for AnsState and Bitlen to reduce the memory footprint
// of Node. This improves performance, likely due to fewer cache misses.
#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct Node {
  pub offset_bits_bits_to_read: u32,
  pub symbol: CompactSymbol,
  pub next_state_idx_base: CompactAnsState,
  // pub ans_part: CompactAnsState,
  // pub offset_bits: CompactBitlen,
  // pub bits_to_read: CompactBitlen,
}

#[derive(Clone, Debug)]
pub struct Decoder {
  pub nodes: Vec<Node>,
}

impl Decoder {
  pub fn new<L: Latent>(spec: &Spec, bins: &[Bin<L>]) -> Self {
    let table_size = spec.table_size();
    let mut nodes = Vec::with_capacity(table_size);
    // x_s from Jarek Duda's paper
    let mut symbol_x_s = spec.symbol_weights.clone();
    for &symbol in &spec.state_symbols {
      let next_state_base = symbol_x_s[symbol as usize] as AnsState;
      let bits_to_read = next_state_base.leading_zeros() - (table_size as AnsState).leading_zeros();
      let next_state_base = next_state_base << bits_to_read;
      let offset_bits = if bins.len() > symbol as usize {
        bins[symbol as usize].offset_bits
      } else {
        0
      };
      nodes.push(Node {
        symbol: symbol as CompactSymbol,
        next_state_idx_base: (next_state_base - table_size as AnsState) as CompactAnsState,
        // offset_bits: bins[symbol as usize].offset_bits as CompactBitlen,
        // bits_to_read: bits_to_read as CompactBitlen,
        // ans_part: ((1 << bits_to_read) - 1) as CompactAnsState,
        offset_bits_bits_to_read: (offset_bits << 16) | bits_to_read,
      });
      symbol_x_s[symbol as usize] += 1;
    }

    Self { nodes }
  }
}
