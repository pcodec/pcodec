use crate::ans::spec::Spec;
use crate::ans::{AnsState, CompactAnsState, CompactSymbol};
use crate::constants::CompactBitlen;

#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct Node {
  pub symbol: CompactSymbol,
  pub next_state_idx_base: CompactAnsState,
  pub bits_to_read: CompactBitlen,
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
      nodes.push(Node {
        symbol: symbol as CompactSymbol,
        next_state_idx_base: (next_state_base - table_size as AnsState) as CompactAnsState,
        bits_to_read: bits_to_read as CompactBitlen,
      });
      symbol_x_s[symbol as usize] += 1;
    }

    Self { nodes }
  }
}
