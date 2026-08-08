use std::cmp::max;
use std::collections::HashMap;

use crate::macros::match_latent_enum;
use crate::{constants::CLASSIC_MEMORIZABLE_BINS_LOG, metadata::DynLatents};
use rand_xoshiro::rand_core::{RngCore, SeedableRng};

use crate::data_types::Latent;

pub const MIN_SAMPLE: usize = 10;
// 1 in this many nums get put into sample
const SAMPLE_RATIO: usize = 40;
// Int mults will be considered infrequent if they occur less than 1/this of
// the time.
const CLASSIC_MEMORIZABLE_BINS: f64 = (1 << CLASSIC_MEMORIZABLE_BINS_LOG) as f64;
// how many times over to try collecting samples without replacement before
// giving up
const SAMPLING_PERSISTENCE: usize = 4;
const DELTA_TARGET_GROUP_N: usize = 200;

fn calc_sample_n(n: usize) -> Option<usize> {
  if n >= MIN_SAMPLE {
    Some(MIN_SAMPLE + (n - MIN_SAMPLE) / SAMPLE_RATIO)
  } else {
    None
  }
}

// extracted for testing
fn choose_delta_sample_inner(
  primary_latents: &DynLatents,
  group_n: usize,
  n_groups: usize,
) -> DynLatents {
  let n = primary_latents.len();
  let nominal_sample_size = n_groups * group_n;
  let group_stride = group_n + (n.saturating_sub(nominal_sample_size) / (n_groups.max(2) - 1));

  match_latent_enum!(
    primary_latents,
    DynLatents<L>(primary_latents) => {
      let mut sample = Vec::<L>::with_capacity(nominal_sample_size);
      for i in 0..n_groups {
        let group_start = group_stride * i;
        sample.extend(&primary_latents[group_start..group_start + group_n]);
      }
      DynLatents::new(sample)
    }
  )
}

pub(crate) fn choose_delta_sample(primary_latents: &DynLatents) -> Option<DynLatents> {
  // We select some large contiguous groups of latents so that delta encodings
  // can both sample from a variety of places in the chunk while minimizing the
  // number of unnatural jumps.
  let n = primary_latents.len();
  let target_sample_n = calc_sample_n(n)?;
  Some(choose_delta_sample_inner(
    primary_latents,
    DELTA_TARGET_GROUP_N.min(n),
    target_sample_n.div_ceil(DELTA_TARGET_GROUP_N),
  ))
}

#[inline(never)]
pub fn choose_mode_sample<T, S, Filter: Fn(&T) -> Option<S>>(
  nums: &[T],
  filter: Filter,
) -> Option<Vec<S>> {
  // We can't modify the list, and copying it may be expensive, but we want to
  // sample a small fraction from it without replacement, so we keep a
  // bitpacked vector representing whether each one is used yet and just keep
  // resampling.
  // Maybe this is a bad idea, but it works for now.
  let target_sample_size = calc_sample_n(nums.len())?;

  let mut rng = rand_xoshiro::Xoroshiro128PlusPlus::seed_from_u64(0);
  let mut visited = vec![0_u8; nums.len().div_ceil(8)];
  let mut res = Vec::with_capacity(target_sample_size);
  let mut n_iters = 0;
  while res.len() < target_sample_size && n_iters < SAMPLING_PERSISTENCE * target_sample_size {
    let rand_idx = rng.next_u64() as usize % nums.len();
    let visited_idx = rand_idx / 8;
    let visited_bit = rand_idx % 8;
    let mask = 1 << visited_bit;
    let is_visited = visited[visited_idx] & mask;
    if is_visited == 0 {
      if let Some(x) = filter(&nums[rand_idx]) {
        res.push(x);
      }
      visited[visited_idx] |= mask;
    }
    n_iters += 1;
  }

  if res.len() >= MIN_SAMPLE {
    Some(res)
  } else {
    None
  }
}

pub struct PrimaryLatentAndSavings<L: Latent> {
  pub primary: L,
  pub bits_saved: f64,
}

#[inline(never)]
pub fn est_bits_saved_per_num<L: Latent, S: Copy, F: Fn(S) -> PrimaryLatentAndSavings<L>>(
  sample: &[S],
  primary_fn: F,
) -> f64 {
  let mut primary_counts_and_savings = HashMap::<L, (usize, f64)>::with_capacity(sample.len());
  for &x in sample {
    let PrimaryLatentAndSavings {
      primary: primary_latent,
      bits_saved,
    } = primary_fn(x);
    let entry = primary_counts_and_savings
      .entry(primary_latent)
      .or_default();
    entry.0 += 1;
    entry.1 += bits_saved;
  }

  let infrequent_cutoff = max(
    1,
    (sample.len() as f64 / CLASSIC_MEMORIZABLE_BINS) as usize,
  );

  // Maybe this should be made fuzzy instead of a hard cutoff because it's just
  // a sample.
  let sample_bits_saved = primary_counts_and_savings
    .values()
    .filter(|&&(count, _)| count <= infrequent_cutoff)
    .map(|&(_, bits_saved)| bits_saved)
    .sum::<f64>();
  sample_bits_saved / sample.len() as f64
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_sample_n() {
    assert_eq!(calc_sample_n(9), None);
    assert_eq!(calc_sample_n(10), Some(10));
    assert_eq!(calc_sample_n(100), Some(12));
    assert_eq!(calc_sample_n(1000010), Some(25010));
  }

  #[test]
  fn test_choose_delta_sample() {
    let latents = DynLatents::new(vec![0_u32, 1]);
    assert_eq!(
      choose_delta_sample_inner(&latents, 2, 1)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1]
    );
    assert_eq!(
      choose_delta_sample_inner(&latents, 1, 2)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1]
    );

    let latents = DynLatents::new((0..300).collect::<Vec<u32>>());
    let sample = choose_delta_sample_inner(&latents, 100, 2)
      .downcast::<u32>()
      .unwrap();
    assert_eq!(sample.len(), 200);
    assert_eq!(&sample[..3], &[0, 1, 2]);
    assert_eq!(&sample[197..], &[297, 298, 299]);

    let latents = DynLatents::new((0..8).collect::<Vec<u32>>());
    assert_eq!(
      choose_delta_sample_inner(&latents, 2, 3)
        .downcast::<u32>()
        .unwrap(),
      vec![0, 1, 3, 4, 6, 7]
    );
  }

  #[test]
  fn test_choose_mode_sample() {
    let mut nums = Vec::new();
    for i in 0..150 {
      nums.push(-i as f32);
    }
    let mut sample = choose_mode_sample(&nums, |&num| {
      if num == 0.0 {
        None
      } else {
        Some(num)
      }
    })
    .unwrap();
    sample.sort_unstable_by(f32::total_cmp);
    assert_eq!(sample.len(), 13);
    assert_eq!(&sample[0..3], &[-147.0, -142.0, -119.0]);
  }
}
