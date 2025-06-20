use crate::ans::spec::Spec;
use crate::ans::{AnsState, Symbol};
use crate::constants::Bitlen;

#[derive(Clone, Debug)]
#[repr(align(8))]
pub struct Node {
  // pub symbol_and_next_state_idx_base: u32,
  // pub offset_bits_and_bits_to_read: u32,
  pub symbol: u16,
  pub next_state_idx_base: u16,
  pub offset_bits: u16,
  pub bits_to_read: u16,
}

impl Node {
  #[inline]
  pub fn symbol(&self) -> Symbol {
    self.symbol as Symbol
    // self.symbol_and_next_state_idx_base >> 16
  }
  #[inline]
  pub fn next_state_idx_base(&self) -> AnsState {
    self.next_state_idx_base as AnsState
    // self.symbol_and_next_state_idx_base as u16 as u32
  }
  #[inline]
  pub fn offset_bits(&self) -> Bitlen {
    self.offset_bits as Bitlen
    // self.offset_bits_and_bits_to_read >> 16
  }
  #[inline]
  pub fn bits_to_read(&self) -> Bitlen {
    self.bits_to_read as Bitlen
    // self.offset_bits_and_bits_to_read as u16 as u32
  }
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
      let mut next_state_base = symbol_x_s[symbol as usize] as AnsState;
      let mut bits_to_read = 0;
      while next_state_base < table_size as AnsState {
        next_state_base *= 2;
        bits_to_read += 1;
      }
      let next_state_idx_base = next_state_base - table_size as AnsState;
      // let symbol_and_next_state_idx_base =
      //   (symbol << 16) + next_state_idx_base;
      let offset_bits = bin_offset_bits.get(symbol as usize).cloned().unwrap_or(0);
      // let offset_bits_and_bits_to_read =
      //   (offset_bits << 16) + bits_to_read;
      nodes.push(Node {
        symbol: symbol as u16,
        next_state_idx_base: next_state_idx_base as u16,
        offset_bits: offset_bits as u16,
        bits_to_read: bits_to_read as u16,
        // symbol_and_next_state_idx_base,
        // offset_bits_and_bits_to_read,
      });
      symbol_x_s[symbol as usize] += 1;
    }

    Self { nodes }
  }
}
