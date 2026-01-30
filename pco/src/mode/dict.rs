use std::{cmp, collections::HashMap};

use crate::{
  data_types::{Latent, ModeAndLatents, Number, SplitLatents},
  dyn_slices::DynLatentSlice,
  errors::{PcoError, PcoResult},
  metadata::{DynLatents, Mode},
};

fn configure_less_specialized<L: Latent>(classic_nums: Vec<L>) -> PcoResult<ModeAndLatents> {
  let mut counts = HashMap::new();
  for &num in &classic_nums {
    *counts.entry(num).or_insert(0_u32) += 1;
  }
  let mut counts = counts.into_iter().collect::<Vec<(L, u32)>>();
  counts.sort_by_key(|&(_, count)| cmp::Reverse(count));
  let ordered_unique = counts.into_iter().map(|(x, _)| x).collect::<Vec<_>>();
  let mut index_hashmap = HashMap::new();
  for (i, &val) in ordered_unique.iter().enumerate() {
    index_hashmap.insert(val, i as u32);
  }
  let mode = Mode::Dict(DynLatents::new(ordered_unique));
  let indices = classic_nums
    .into_iter()
    .map(|num| *index_hashmap.get(&num.to_latent_ordered()).unwrap())
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
