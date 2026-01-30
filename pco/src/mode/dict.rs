use std::{cmp, collections::HashMap};

use crate::{
  data_types::{Latent, ModeAndLatents, Number, SplitLatents},
  dyn_slices::DynLatentSlice,
  errors::{PcoError, PcoResult},
  metadata::{DynLatents, Mode},
};

fn configure_less_specialized<L: Latent>(classic_nums: Vec<L>) -> PcoResult<ModeAndLatents> {
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
  Ok((
    mode,
    SplitLatents {
      primary: latents,
      secondary: None,
    },
  ))
}

pub fn configure_and_split_latents<T: Number>(nums: &[T]) -> PcoResult<ModeAndLatents> {
  let classic_nums = nums
    .iter()
    .map(|&num| num.to_latent_ordered())
    .collect::<Vec<_>>();
  configure_less_specialized(classic_nums)
}

pub fn join_latents<T: Number>(
  dict: &DynLatents,
  primary: DynLatentSlice,
  dst: &mut [T],
) -> PcoResult<()> {
  let dict = dict.downcast_ref::<T::L>().unwrap();
  let idxs = primary.downcast::<u32>().unwrap();
  if idxs.iter().any(|idx| *idx > dict.len() as u32) {
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
