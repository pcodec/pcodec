use std::{cmp, collections::HashMap};

use crate::{
  data_types::{Latent, ModeAndLatents, Number, SplitLatents},
  errors::PcoResult,
  metadata::{DynLatents, Mode},
};

pub fn configure_and_split_latents<T: Number>(nums: &[T]) -> PcoResult<ModeAndLatents> {
  let mut counts = HashMap::new();
  for &num in nums {
    *counts.entry(num.to_latent_ordered()).or_insert(0_u32) += 1;
  }
  let mut counts = counts.into_iter().collect::<Vec<(T::L, u32)>>();
  counts.sort_by_key(|&(_, count)| cmp::Reverse(count));
  let ordered_unique = counts.into_iter().map(|(x, _)| x).collect::<Vec<_>>();
  let mut index_hashmap = HashMap::new();
  for (i, &val) in ordered_unique.iter().enumerate() {
    index_hashmap.insert(val, i as u32);
  }
  let mode = Mode::Dict(DynLatents::new(ordered_unique));
  let indices = nums
    .iter()
    .map(|&num| T::L::from_u32(*index_hashmap.get(&num.to_latent_ordered()).unwrap()))
    .collect();
  let latents = DynLatents::new(indices);
  Ok((
    mode,
    SplitLatents {
      primary: latents,
      secondary: None,
    },
  ))
}

pub fn join_latents<L: Latent>(primary: &mut [L], dict: &DynLatents) {
  let dict = dict.downcast_ref::<L>().unwrap();
  for latent in primary.iter_mut() {
    let index = latent.to_u64() as usize;
    *latent = dict[index];
  }
}
