use crate::ans::spec::Spec;
use crate::ans::{AnsState, Symbol};
use crate::constants::Bitlen;

// Using smallar types for AnsState and Bitlen to reduce the memory footprint
// of Node. This improves performance, likely due to fewer cache misses.
#[derive(Clone, Debug)]
pub struct Node(u64);
// #[repr(align(8))]
// pub struct Node {
//   pub symbol: Symbol,
//   pub next_state_idx_base: CompactAnsState,
//   pub bits_to_read: CompactBitlen,
// }

const SYMBOL_MASK: usize = u32::MAX as usize;
const STATE_IDX_MASK: usize = (1 << 16) - 1;

impl Node {
  fn new(symbol: Symbol, next_state_idx_base: AnsState, bits_to_read: Bitlen) -> Self {
    Self((symbol as u64) | ((next_state_idx_base as u64) << 32) | ((bits_to_read as u64) << 48))
  }

  #[inline]
  pub fn symbol(&self) -> usize {
    self.0 as usize & SYMBOL_MASK
  }

  #[inline]
  pub fn next_state_idx_base(&self) -> usize {
    (self.0 >> 32) as usize & STATE_IDX_MASK
  }

  #[inline]
  pub fn bits_to_read(&self) -> Bitlen {
    (self.0 >> 48) as Bitlen
  }
}

#[derive(Clone, Debug)]
pub struct Decoder {
  pub nodes: Vec<Node>,
}

impl Decoder {
  pub fn new(spec: &Spec) -> Self {
    let table_size = spec.table_size();
    let mut nodes = Vec::with_capacity(table_size);
    // x_s from Jarek Duda's paper
    let mut symbol_x_s = spec.symbol_weights.clone();
    for &symbol in &spec.state_symbols {
      let next_state_base = symbol_x_s[symbol as usize] as AnsState;
      let bits_to_read = next_state_base.leading_zeros() - (table_size as AnsState).leading_zeros();
      let next_state_base = next_state_base << bits_to_read;
      nodes.push(Node::new(
        symbol,
        next_state_base - table_size as AnsState,
        bits_to_read,
      ));
      symbol_x_s[symbol as usize] += 1;
    }

    Self { nodes }
  }
}
