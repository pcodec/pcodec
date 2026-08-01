use std::cmp;
use std::collections::HashMap;

use crate::compression_intermediates::Bid;
use crate::data_types::{Latent, Number, SplitLatents};
use crate::dyn_slices::DynLatentSlice;
use crate::errors::{PcoError, PcoResult};
use crate::metadata::{DynLatents, Mode};

fn configure_less_specialized<L: Latent>(classic_nums: Vec<L>) -> (Mode, SplitLatents) {
  let mut count_by_unique = HashMap::new();
  for &num in &classic_nums {
    *count_by_unique.entry(num).or_insert(0_u32) += 1;
  }

  // We sort by frequency descending to maximize the benefit of binning. We do
  // this via an argsort instead of directly sorting a Vec<(L, u32)> to reduce
  // binary size.
  let unique_counts = count_by_unique
    .iter()
    .map(|(&val, &count)| (val, count))
    .collect::<Vec<(L, u32)>>();
  let mut sort_idx_counts = unique_counts
    .iter()
    .enumerate()
    .map(|(idx, &(_, count))| (idx as u32, count))
    .collect::<Vec<(u32, u32)>>();
  sort_idx_counts.sort_unstable_by_key(|&(_, count)| cmp::Reverse(count));
  let dict = sort_idx_counts
    .into_iter()
    .map(|(sort_idx, _)| unique_counts[sort_idx as usize].0)
    .collect::<Vec<_>>();

  // Here we reuse the hashmap we no longer need.
  let mut dict_idx_by_unique = count_by_unique;
  for (i, &val) in dict.iter().enumerate() {
    dict_idx_by_unique.insert(val, i as u32);
  }
  let mode = Mode::Dict(DynLatents::new(dict));
  let indices = classic_nums
    .into_iter()
    .map(|num| *dict_idx_by_unique.get(&num.to_latent_ordered()).unwrap())
    .collect();
  let latents = DynLatents::U32(indices);
  (
    mode,
    SplitLatents {
      primary: latents,
      secondary: None,
    },
  )
}

/// Unlike other modes, dict has to build its dictionary from all the numbers,
/// which produces the latents as a byproduct. We keep them in the split_fn so
/// that the bid interface stays uniform.
pub fn compute_bid<T: Number>(nums: &[T]) -> Bid<T> {
  let classic_nums = nums
    .iter()
    .map(|&num| num.to_latent_ordered())
    .collect::<Vec<_>>();
  let (mode, latents) = configure_less_specialized(classic_nums);
  Bid {
    mode,
    bits_saved_per_num: 0.0,
    split_fn: Box::new(move |_nums| latents),
  }
}

pub fn join_latents<T: Number>(
  dict: &DynLatents,
  primary: DynLatentSlice,
  dst: &mut [T],
) -> PcoResult<()> {
  let dict = dict.downcast_ref::<T::L>().unwrap();
  let idxs = primary.downcast::<u32>().unwrap();
  if idxs.iter().any(|idx| *idx >= dict.len() as u32) {
    // in some cases it is possible to prove the indices are in range from
    // looking at the bins ahead of time, but just keeping this simple for now
    return Err(PcoError::corruption(format!(
      "dict index exceeded dict length {}",
      dict.len()
    )));
  }

  for (idx, num) in idxs.iter().zip(dst.iter_mut()) {
    *num = T::from_latent_ordered(dict[*idx as usize]);
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::dyn_slices::DynLatentSlice;
  use crate::metadata::DynLatents;

  #[test]
  fn test_join_latents_oob_index_returns_err() {
    // dict has 3 entries; valid indices are 0, 1, 2
    let dict = DynLatents::new(vec![10_u32, 20_u32, 30_u32]);
    // index 3 == dict.len() — the off-by-one: currently passes the `> dict.len()`
    // guard and panics on dict[3]; after the fix it must return Err
    let idxs = [0_u32, 1_u32, 3_u32];
    let mut dst = vec![0_u32; 3];
    let result = join_latents::<u32>(&dict, DynLatentSlice::new(&idxs), &mut dst);
    assert!(
      result.is_err(),
      "expected Err for out-of-range dict index"
    );
  }
}
