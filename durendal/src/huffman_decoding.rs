use std::cmp::min;

use crate::bin::{Bin, BinDecompressionInfo};
use crate::bit_reader::BitReader;
use crate::constants::MAX_BIN_TABLE_SIZE_LOG;
use crate::data_types::{NumberLike, UnsignedLike};
use crate::errors::{QCompressError, QCompressResult};

#[derive(Clone, Debug)]
pub enum HuffmanTable<U: UnsignedLike> {
  Leaf(BinDecompressionInfo<U>),
  NonLeaf {
    table_size_log: usize,
    children: Vec<HuffmanTable<U>>,
  },
}

impl<U: UnsignedLike> Default for HuffmanTable<U> {
  fn default() -> Self {
    HuffmanTable::Leaf(BinDecompressionInfo::default())
  }
}

impl<U: UnsignedLike> HuffmanTable<U> {
  pub fn search_with_reader(
    &self,
    reader: &mut BitReader,
  ) -> QCompressResult<BinDecompressionInfo<U>> {
    let mut node = self;
    let mut read_depth = 0;
    loop {
      match node {
        HuffmanTable::Leaf(decompression_info) => {
          reader.rewind_bin_overshoot(read_depth - decompression_info.depth);
          return Ok(*decompression_info);
        }
        HuffmanTable::NonLeaf {
          table_size_log,
          children,
        } => {
          let (bits_read, idx) = reader.read_bin_table_idx(*table_size_log)?;
          read_depth += bits_read;
          node = &children[idx];
          if bits_read != *table_size_log {
            return match node {
              HuffmanTable::Leaf(decompression_info) if decompression_info.depth == read_depth => {
                Ok(*decompression_info)
              }
              HuffmanTable::Leaf(_) => Err(QCompressError::insufficient_data(
                "search_with_reader(): ran out of data parsing Huffman bin (reached leaf)",
              )),
              HuffmanTable::NonLeaf {
                table_size_log: _,
                children: _,
              } => Err(QCompressError::insufficient_data(
                "search_with_reader(): ran out of data parsing Huffman bin (reached parent)",
              )),
            };
          }
        }
      }
    }
  }

  pub fn unchecked_search_with_reader(&self, reader: &mut BitReader) -> BinDecompressionInfo<U> {
    let mut node = self;
    let mut read_depth = 0;
    loop {
      match node {
        HuffmanTable::Leaf(decompression_info) => {
          reader.rewind_bin_overshoot(read_depth - decompression_info.depth);
          return *decompression_info;
        }
        HuffmanTable::NonLeaf {
          table_size_log,
          children,
        } => {
          let idx = reader.unchecked_read_bin_table_idx(*table_size_log);
          node = &children[idx];
          read_depth += table_size_log;
        }
      }
    }
  }
}

impl<T: NumberLike> From<&Vec<Bin<T>>> for HuffmanTable<T::Unsigned> {
  fn from(bins: &Vec<Bin<T>>) -> Self {
    if bins.is_empty() {
      HuffmanTable::default()
    } else {
      build_from_bins_recursive(bins, 0)
    }
  }
}

fn mask_to_n_bits(x: usize, n: usize) -> usize {
  x & (usize::MAX >> (usize::BITS as usize - n))
}

fn build_from_bins_recursive<T: NumberLike>(
  bins: &[Bin<T>],
  depth: usize,
) -> HuffmanTable<T::Unsigned> {
  if bins.len() == 1 {
    let bin = &bins[0];
    HuffmanTable::Leaf(BinDecompressionInfo::from(bin))
  } else {
    let max_depth = bins.iter().map(|bin| bin.code_len).max().unwrap();
    let table_size_log: usize = min(MAX_BIN_TABLE_SIZE_LOG, max_depth - depth);
    let table_size = 1 << table_size_log;

    let mut children = Vec::new();
    for idx in 0..table_size {
      // We put each bin into the table, possibly in multiple consecutive locations.
      // e.g. if the table size log is 7 and we have a 4-bit code, we'll put the bin in
      // 2^3=8 table indexes. We do this by iterating over all indices and finding the
      // bins that belong.
      let possible_bins = bins
        .iter()
        .filter(|&bin| {
          mask_to_n_bits(idx, bin.code_len - depth) == mask_to_n_bits(bin.code >> depth, table_size_log)
        }
        )
        .cloned()
        .collect::<Vec<Bin<T>>>();
      let child = build_from_bins_recursive(&possible_bins, depth + table_size_log);
      children.push(child);
    }
    HuffmanTable::NonLeaf {
      table_size_log,
      children,
    }
  }
}
