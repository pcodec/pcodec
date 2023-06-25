use crate::constants::{Bitlen, WORD_BITLEN};
use crate::errors::{QCompressError, QCompressResult};

// Here and in encoding/decoding, state is between [0, table_size)

pub type Token = u32;

pub struct Spec {
  // log base 2 of the table size
  // e.g. the table states will be in [2^size_log, 2^(size_log + 1))
  pub size_log: Bitlen,
  // the ordered tokens in the table
  pub state_tokens: Vec<Token>,
  // the number of times each token appears in the table
  pub token_weights: Vec<usize>,
}

// We use a relatively prime (odd) number near 3/5 of the table size. In this
// way, uncommon tokens with weight=2, 3, 4, 5 all get pretty reasonable
// spreads (in a slightly more balanced way than e.g. 4/7 would):
// * 2 -> [0, 0.6]
// * 3 -> [0, 0.2, 0.6]
// * 4 -> [0, 0.2, 0.6, 0.8]
// * 5 -> [0, 0.2, 0.4, 0.6, 0.8]
fn choose_stride(table_size: usize) -> usize {
  let mut res = (3 * table_size) / 5;
  if res % 2 == 0 {
    res += 1;
  }
  res
}

impl Spec {
  // TODO this can be slow when there are many tokens
  fn spread_state_tokens(size_log: Bitlen, token_weights: &[usize]) -> QCompressResult<Vec<Token>> {
    let table_size = token_weights.iter().sum::<usize>();
    if table_size != (1 << size_log) {
      return Err(QCompressError::corruption(format!(
        "table size log of {} does not agree with total weight of {}",
        size_log, table_size,
      )));
    }

    let mut res = vec![0; table_size];
    let mut step = 0;
    let stride = choose_stride(table_size);
    let mod_table_size = usize::MAX >> 1 >> (WORD_BITLEN - size_log - 1);
    for (token, &weight) in token_weights.iter().enumerate() {
      for _ in 0..weight {
        let state = (stride * step) & mod_table_size;
        res[state] = token as Token;
        step += 1;
      }
    }

    Ok(res)
  }
  // This needs to remain backward compatible.
  // The general idea is to spread the tokens out as much as possible,
  // deterministically, and ensuring each one gets as least one state.
  // Long runs of tokens are generally bad.
  // In the sparse case, it's best to have the very frequent tokens in the low
  // states and rarer tokens somewhat present in the high states, so for best
  // compression, we expect token_weights to be ordered from least frequent to
  // most frequent.
  pub fn from_weights(size_log: Bitlen, token_weights: Vec<usize>) -> QCompressResult<Self> {
    let token_weights = if token_weights.is_empty() {
      vec![1]
    } else {
      token_weights
    };

    let state_tokens = Self::spread_state_tokens(size_log, &token_weights)?;

    Ok(Self {
      size_log,
      state_tokens,
      token_weights,
    })
  }

  pub fn table_size(&self) -> usize {
    1 << self.size_log
  }
}

#[cfg(test)]
mod tests {
  use crate::ans::spec::{Spec, Token};
  use crate::errors::QCompressResult;

  fn assert_state_tokens(weights: Vec<usize>, expected: Vec<Token>) -> QCompressResult<()> {
    let table_size_log = weights.iter().sum::<usize>().ilog2();
    let spec = Spec::from_weights(table_size_log, weights)?;
    assert_eq!(spec.state_tokens, expected);
    Ok(())
  }

  #[test]
  fn ans_spec_new() -> QCompressResult<()> {
    assert_state_tokens(
      vec![1, 1, 3, 11],
      vec![3, 3, 3, 3, 2, 3, 3, 3, 3, 2, 3, 3, 3, 2, 1, 0],
    )
  }

  #[test]
  fn ans_spec_new_trivial() -> QCompressResult<()> {
    assert_state_tokens(vec![1], vec![0])?;
    assert_state_tokens(vec![2], vec![0, 0])
  }
}
