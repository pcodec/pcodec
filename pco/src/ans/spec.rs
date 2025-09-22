use crate::ans::Symbol;
use crate::constants::{Bitlen, Weight};
use crate::errors::{PcoError, PcoResult};

// Here and in encoding/decoding, state is between [0, table_size)

pub struct Spec {
  // log base 2 of the table size
  // e.g. the table states will be in [2^size_log, 2^(size_log + 1))
  pub size_log: Bitlen,
  // the ordered symbols in the table
  pub state_symbols: Vec<Symbol>,
  // the number of times each symbol appears in the table
  pub symbol_weights: Vec<Weight>,
}

// We use a relatively prime (odd) number near 3/5 of the table size. In this
// way, uncommon symbols with weight=2, 3, 4, 5 all get pretty reasonable
// spreads (in a slightly more balanced way than e.g. 4/7 would):
// * 2 -> [0, 0.6]
// * 3 -> [0, 0.2, 0.6]
// * 4 -> [0, 0.2, 0.6, 0.8]
// * 5 -> [0, 0.2, 0.4, 0.6, 0.8]
fn choose_stride(table_size: Weight) -> Weight {
  let mut res = (3 * table_size) / 5;
  if res.is_multiple_of(2) {
    res += 1;
  }
  res
}

impl Spec {
  // This needs to remain backward compatible.
  // The general idea is to spread the symbols out as much as possible,
  // deterministically, and ensuring each one gets as least one state.
  // Long runs of symbols are generally bad.
  fn spread_state_symbols(size_log: Bitlen, symbol_weights: &[Weight]) -> PcoResult<Vec<Symbol>> {
    let table_size = symbol_weights.iter().sum::<Weight>();
    if table_size != (1 << size_log) {
      return Err(PcoError::corruption(format!(
        "table size log of {} does not agree with total weight of {}",
        size_log, table_size,
      )));
    }

    let mut res = vec![0; table_size as usize];
    let mut step = 0;
    let stride = choose_stride(table_size);
    let mod_table_size = Weight::MAX >> 1 >> (Weight::BITS as Bitlen - 1 - size_log);
    for (symbol, &weight) in symbol_weights.iter().enumerate() {
      for _ in 0..weight {
        let state_idx = (stride * step) & mod_table_size;
        res[state_idx as usize] = symbol as Symbol;
        step += 1;
      }
    }

    Ok(res)
  }

  pub fn from_weights(size_log: Bitlen, symbol_weights: Vec<Weight>) -> PcoResult<Self> {
    let symbol_weights = if symbol_weights.is_empty() {
      vec![1]
    } else {
      symbol_weights
    };

    let state_symbols = Self::spread_state_symbols(size_log, &symbol_weights)?;

    Ok(Self {
      size_log,
      state_symbols,
      symbol_weights,
    })
  }

  pub fn table_size(&self) -> usize {
    1 << self.size_log
  }
}

#[cfg(test)]
mod tests {
  use crate::ans::spec::{Spec, Symbol};
  use crate::constants::Weight;
  use crate::errors::PcoResult;

  fn assert_state_symbols(weights: Vec<Weight>, expected: Vec<Symbol>) -> PcoResult<()> {
    let table_size_log = weights.iter().sum::<Weight>().ilog2();
    let spec = Spec::from_weights(table_size_log, weights)?;
    assert_eq!(spec.state_symbols, expected);
    Ok(())
  }

  #[test]
  fn ans_spec_new() -> PcoResult<()> {
    assert_state_symbols(
      vec![1, 1, 3, 11],
      vec![0, 3, 2, 3, 2, 3, 3, 3, 3, 1, 3, 2, 3, 3, 3, 3],
    )
  }

  #[test]
  fn ans_spec_new_trivial() -> PcoResult<()> {
    assert_state_symbols(vec![1], vec![0])?;
    assert_state_symbols(vec![2], vec![0, 0])
  }
}
